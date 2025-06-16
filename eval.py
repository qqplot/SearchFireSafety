import os, json, argparse, pathlib, tqdm, numpy as np, torch, concurrent.futures
from collections import defaultdict
import random        
import orjson

def _loads(line: str):
    return orjson.loads(line) if orjson else json.loads(line)

def load_doc_db(path):
    if path is None:
        return {}

    db = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            j = _loads(ln)

            doc_id      = j["doc_id"]
            semantic_id = j.get("semantic_id", "").strip()
            body = (
                j.get("chapter_body")
                or j.get("text")
                or j.get("article_body")
                or j.get("body")
                or ""
            )

            text = f"법령: {semantic_id}\n{body}" if semantic_id else body
            db[str(doc_id)] = text
    return db

from transformers import PreTrainedTokenizerFast
import torch.nn as nn

from bert_score import BERTScorer

def build_bertscorer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BERTScorer(
        model_type="beomi/kcbert-base",
        num_layers=4,
        lang="ko",
        device=device,
        rescale_with_baseline=False,
    )

def bert_calc(item, scorer):
    q, ref, hyp = item["question"], item["answer"], item["model_answer_clean"]
    P, R, F = scorer.score([hyp], [ref], verbose=False)
    _, _, Fqh = scorer.score([hyp], [q], verbose=False)
    prec, rec, f_sc, faith = P[0].item(), R[0].item(), F[0].item(), Fqh[0].item()
    return {
        "bert_faithfulness": faith,
        "bert_precision":    prec,
        "bert_recall":       rec,
        "bert_fscore":       f_sc,
        "bertscore":         0.5 * (faith + f_sc)
    }

from rouge_score import rouge_scorer
rouge_scorer_ = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    use_stemmer=True
)

def rouge_calc(item):
    ref, hyp = item["answer"], item["model_answer_clean"]
    scores = rouge_scorer_.score(ref, hyp)
    return {
        "rouge1_precision":   scores["rouge1"].precision,
        "rouge1_recall":      scores["rouge1"].recall,
        "rouge1_f1":          scores["rouge1"].fmeasure,
        "rouge2_precision":   scores["rouge2"].precision,
        "rouge2_recall":      scores["rouge2"].recall,
        "rouge2_f1":          scores["rouge2"].fmeasure,
        "rougeL_precision":   scores["rougeL"].precision,
        "rougeL_recall":      scores["rougeL"].recall,
        "rougeL_f1":          scores["rougeL"].fmeasure,
        "rougeLsum_precision": scores["rougeLsum"].precision,
        "rougeLsum_recall":    scores["rougeLsum"].recall,
        "rougeLsum_f1":        scores["rougeLsum"].fmeasure
    }

import openai    # pip install openai
from openai import OpenAI

def build_openai_client(api_key=None, timeout=20):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not available.")
    return OpenAI(api_key=api_key, timeout=timeout)

def llm_judge(item, client, model="gpt-4o"):
    """
    Returns 1 if model_answer_clean is factually correct & sufficiently comprehensive,
            0 otherwise (strict binary).
    """
    q, ref, hyp = item["question"], item["answer"], item["model_answer_clean"]

    messages = [
        {"role": "system",
         "content":
         "You are an expert grader. "
         "Return ONLY a single character: '1' (if the model answer is factually correct and sufficiently comprehensive "
         "relative to the gold answer) or '0' (otherwise). No explanation, no punctuation."},
        {"role": "user",
         "content":
f"""### Question
{q}

### Gold Answer
{ref}

### Model Answer
{hyp}

### Task
Judge the model answer. Respond with 1 or 0 only."""}
    ]
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1,
        )
        txt = rsp.choices[0].message.content.strip()
        return 1 if txt.startswith("1") else 0
    except Exception as e:
        return 0

def llm_pairwise_judge(item,
                       client,
                       id2text,
                       model="gpt-4o",
                       seed=None):
                         
    rng = random.Random(seed)   

    q   = item["question"]
    ans = item["answer"]
    hyp = item["model_answer_clean"]

    ctx = ""
    doc_ids = [str(d) for d in item.get("matched_doc_id_merged", [])]
    if doc_ids and id2text:
        ctx = "\n\n".join(id2text[d] for d in doc_ids if d in id2text)

    if rng.random() < 0.5:
        A, B              = ans, hyp
        model_is_A        = False
    else:
        A, B              = hyp, ans
        model_is_A        = True

    prompt = f"""### Question
{q}

### Relevant Documents
{ctx if ctx else '(없음)'}

### Answer A
{A}

### Answer B
{B}

### Task
Assess which answer is **more factually correct and comprehensive** given the question and the documents.
Reply with *only* `A` or `B`."""

    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an expert grader. Reply with a single character: A or B."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1,
        )
        choice = rsp.choices[0].message.content.strip().upper()
        if (choice == "A" and model_is_A) or (choice == "B" and not model_is_A):
            return 1  
        return 0     
    except Exception:
        return 0      

def main(args):
    metrics = set(m.strip().lower() for m in args.metrics.split(","))
    bert_scorer = build_bertscorer()     if "bert"    in metrics else None

    id2text     = load_doc_db(args.oracle_docs) if "winrate" in metrics else {}

    need_openai = {"llm", "winrate"} & metrics and not args.no_llm
    openai_cl   = build_openai_client(api_key=args.openai_api_key) if need_openai else None

    infile   = pathlib.Path(args.input)
    with infile.open(encoding="utf-8") as f:
        items = [json.loads(ln) for ln in tqdm.tqdm(f, desc="Load")]

    out_rows = []

    openai_results = defaultdict(list)     # metric → list(scores)
    if need_openai:
        def work_openai(item):
            res = {}
            if "llm" in metrics:
                res["llm_score"] = llm_judge(
                    item, client=openai_cl, model=args.llm_model
                )
            if "winrate" in metrics:
                seed = item.get("question_id")         
                res["win"] = llm_pairwise_judge(
                    item,
                    client=openai_cl,
                    id2text=id2text,
                    model=args.llm_model,
                    seed=seed,
                )
            return res

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as exe:
            for d in exe.map(work_openai, items):
                for k, v in d.items():
                    openai_results[k].append(v)

    for k in ("llm_score", "win"):
        if k not in openai_results and k in {"llm_score", "win"}:
            openai_results[k] = [0] * len(items)

    for idx, item in enumerate(items):
        if "bert"    in metrics: item.update(bert_calc(item, bert_scorer))
        if "rouge"   in metrics: item.update(rouge_calc(item))
        if "llm"     in metrics:
            item["llm_score"] = openai_results["llm_score"][idx]
        if "winrate" in metrics:
            item["win"] = openai_results["win"][idx]

        out_rows.append(item)

    outfile = pathlib.Path(args.output)
    with outfile.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    msg = ["Dataset-mean ▶"]

    if not out_rows:
        print("No data rows – nothing to score.")
        return

    mean = lambda k: float(np.mean([r[k] for r in out_rows if k in r]))

    def add(key, label):
        if key in out_rows[0]:
            msg.append(f"{label} = {mean(key):.4f}")

    add("bertscore",    "BERTScore")
    add("rouge1_f1",    "ROUGE-1 F1")
    add("rouge2_f1",    "ROUGE-2 F1")
    add("rougeL_f1",    "ROUGE-L F1")
    add("rougeLsum_f1", "ROUGE-Lsum F1")
    add("llm_score",    "LLM-Score")
    add("win",          "Win-Rate")

    print("  ·  ".join(msg))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--openai_api_key", default=None)
    ap.add_argument("--llm_model", default="gpt-4o")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--no_llm", action='store_true')
    ap.add_argument("--metrics", default="bart,bert,rouge,korouge,llm,winrate")
    ap.add_argument("--oracle_docs", default=None)                
    args = ap.parse_args()
    main(args)
