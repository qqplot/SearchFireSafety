from __future__ import annotations
import argparse, json, orjson, os, pickle, sys, re, time, random
from pathlib import Path
from typing import List, Dict, Tuple, Any

import openai
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import gc         
import traceback   

def load_docs(path: str) -> Tuple[List[int], List[str], Dict[int, List[int]]]:
    doc_ids, texts, link_dict = [], [], {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = orjson.loads(line)

            doc_id       = j["doc_id"]
            semantic_id  = j.get("semantic_id", "").strip()       
            body         = j.get("chapter_body") or j.get("text") or ""

            if semantic_id:
                text = f"법령: {semantic_id}\n{body}"
            else:
                text = body

            doc_ids.append(doc_id)
            texts.append(text)
            link_dict[doc_id] = j.get("matched_doc_id_merged") or []

    return doc_ids, texts, link_dict

def load_queries(path: str) -> List[Dict]:
    lst = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = orjson.loads(line)
            if j.get("has_matched_docs"):  
                lst.append(j)
    return lst

def build_retriever(kind: str, docs: List[str], cache_dir: Path):
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
            max_features=120_000,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        tfidf = vect.fit_transform(docs)
        return {"vect": vect, "tfidf": tfidf}
    elif kind == "bm25":
        from rank_bm25 import BM25Okapi
        tokenized = [d.replace("\n", " ").split() for d in docs]
        bm25 = BM25Okapi(tokenized)
        return {"bm25": bm25, "tokenized": tokenized}
    elif kind == "bge":
        pkl = cache_dir / "bge_index.pkl"
        if pkl.exists():
            print("· BGE 인덱스 로드:", pkl)
            with open(pkl, "rb") as f:
                return pickle.load(f)

        from sentence_transformers import SentenceTransformer
        import faiss
        model = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
        model.max_seq_length = 512

        all_embs = model.encode(
            docs,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(all_embs.shape[1])
        index.add(all_embs)
        obj = {"index": index, "model": model}
        with open(pkl, "wb") as f:
            pickle.dump(obj, f)
        return obj
    else:
        raise ValueError(f"Unknown retriever: {kind}")

def search(retriever: Dict, kind: str, query: str, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    if kind == "tfidf":
        q_vec = retriever["vect"].transform([query])
        row = (q_vec @ retriever["tfidf"].T).toarray().ravel()
        idx = np.argpartition(-row, range(topk))[:topk]
        idx = idx[np.argsort(-row[idx])]
        return idx, row[idx]
    elif kind == "bm25":
        q_tok = query.split()
        scores = retriever["bm25"].get_scores(q_tok)
        idx = np.argpartition(-scores, range(topk))[:topk]
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]
    elif kind == "bge":
        import faiss
        q_emb = retriever["model"].encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, idx = retriever["index"].search(q_emb, topk)
        return idx.flatten(), scores.flatten()
    else:
        raise ValueError

def expand_with_links(
    retrieved_ids: List[int],
    link_dict: Dict[int, List[int]],
) -> List[int]:
    seen = set(retrieved_ids)
    expanded = list(retrieved_ids)
    for did in retrieved_ids:
        for linked in link_dict.get(did, []):
            if linked not in seen:
                expanded.append(linked)
                seen.add(linked)
    return expanded

def load_llm(model_name: str, bf16: bool = True):
    if is_openai_model(model_name):        
        return None, None                 
    print(f"· Loading model: {model_name}")
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

def safe_run_generation(model, tokenizer, messages,
                        max_new_tokens=32768,
                        placeholder="[OOM‑SKIPPED]") -> str:
    try:
        return run_generation(model, tokenizer, messages, max_new_tokens)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            print("⚠️  OOM detected ‑ skipping this query", file=sys.stderr)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

            return placeholder
        raise

def build_prompt(                     
    model_name: str,
    user_q: str,
    context: str,
    include_rag_instr: bool = True,     # NEW
) -> List[Dict]:

    if "hyperclovax" in model_name.lower():
        sys_msg = "- 당신은 네이버가 만든 \"CLOVA X\" 라는 AI 언어모델입니다.\n-"
    elif "qwen" in model_name.lower():
        sys_msg = "You are Qwen3, a helpful Korean legal assistant."
    elif "gpt" in model_name.lower():
        sys_msg = "You are Chat-GPT from OpenAI, a helpful Korean legal assistant."
    else:
        sys_msg = "You are EXAONE model from LG AI Research, a helpful Korean legal assistant."

    rag_instr = (
        "다음은 사용자의 질문과 관련 문서입니다. "
        "문서를 충분히 활용해 정확하고 간결하게 한국어로 답변하십시오."
    )

    messages = [{"role": "system", "content": sys_msg}]

    if include_rag_instr and context:
        ctx_block = "-----\n" + context.strip() + "\n-----"
        messages.append({"role": "system", "content": rag_instr})
        messages.append({"role": "user",   "content": f"{user_q}\n\n{ctx_block}"})
    else:   
        messages.append({"role": "user",   "content": user_q})

    return messages

def run_generation(model, tokenizer, messages, max_new_tokens=32768) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    out_text = tokenizer.decode(output[0], skip_special_tokens=True)
    m = re.search(r"<\|assistant\|>(.*)", out_text, re.S)
    return m.group(1).strip() if m else out_text.strip()

def is_openai_model(name: str) -> bool:
    return name.lower().startswith("openai:")


def ensure_openai_key(key_file: str):
    key = None
    if key_file and Path(key_file).exists():
        key = Path(key_file).read_text().strip()
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌  Cannot find OpenAI API. Please check ‑‑openai_key_file")
    openai.api_key = key


def safe_openai_generation(
    model_name: str,
    messages: List[Dict],
    max_new_tokens: int = 2048,
    min_backoff: float = 1.0,
    max_backoff: float = 60.0,
) -> str:

    _retryable_names = [
        "RateLimitError",       
        "APIConnectionError",   
        "InternalServerError",  
        "APITimeoutError",     
        "Timeout",             
        "APIError",           
    ]
    RETRYABLE_EXC: tuple[type[BaseException], ...] = tuple(
        exc
        for name in _retryable_names
        if (exc := getattr(openai, name, None))             
        if isinstance(exc, type) and issubclass(exc, BaseException)  
    )

    NON_RETRYABLE = (
        getattr(openai, "AuthenticationError", BaseException),
        getattr(openai, "PermissionDeniedError", BaseException),
        getattr(openai, "BadRequestError", BaseException),      
        getattr(openai, "InvalidRequestError", BaseException),  
        getattr(openai, "NotFoundError", BaseException),
        getattr(openai, "UnprocessableEntityError", BaseException),
    )

    backoff = min_backoff
    attempt = 1

    while True:
        try:
            resp = openai.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_new_tokens,
            )
            return resp.choices[0].message.content.strip()

        except RETRYABLE_EXC as e:
            wait = random.uniform(backoff * 0.8, backoff * 1.2)  # jitter
            print(
                f"⚠️  {type(e).__name__} ‑ retry #{attempt} after {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
            backoff = min(backoff * 2, max_backoff)         
            attempt += 1

        except NON_RETRYABLE as e:
            raise RuntimeError(f"❌  OpenAI call failed: {e}") from e
        except Exception:
            raise

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("1) Loading Documents...")
    doc_ids, docs, link_dict = load_docs(args.docs)
    print(f"   · {len(docs):,} docs")

    print("2) Loading Query...")
    queries = load_queries(args.queries)
    print(f"   · {len(queries):,} queries (has_matched_docs==True)")

    print("3) Building Index...")
    retriever = build_retriever(args.retriever, docs, out_dir)

    id2text = {d: t for d, t in zip(doc_ids, docs)}        

    if any(is_openai_model(m) for m in args.models):
        if openai is None:
            raise RuntimeError("openai package not installed. `pip install openai>=1.0`")
        ensure_openai_key(args.openai_key_file)

    for model_name in args.models:
        use_openai = is_openai_model(model_name)

        if use_openai:
                base  = model_name.split(":", 1)[1]   
                model, tokenizer = None, None          
        else:
            model, tokenizer = load_llm(model_name)
            base  = model_name.split("/")[-1]
            
        filename = f"{base}_{args.retriever}_k{args.topk}"      
        if args.oracle:                                     
            filename += "_oracle"
        filename += ".jsonl"
        outfile = out_dir / filename

        with open(outfile, "w", encoding="utf-8") as fw, \
             tqdm(queries, desc=f"Inference [{model_name}]") as pbar:

            for j in pbar:
                qtext = j["question"]

                if args.oracle:                                 
                    init_ids = j.get("matched_doc_id", [])
                elif args.topk == 0:                             
                    init_ids = []
                else:                                     
                    idx, _ = search(retriever, args.retriever, qtext, args.topk)
                    init_ids = [doc_ids[i] for i in idx]

                if args.expand_with_links and init_ids:
                    final_ids = expand_with_links(init_ids, link_dict)
                else:
                    final_ids = init_ids

                ctx_docs = "\n\n".join(
                    id2text[did] for did in final_ids if did in id2text
                )

                include_rag = args.topk != 0                     
                prompt_msgs = build_prompt(
                    model_name, qtext, ctx_docs, include_rag_instr=include_rag
                )
                if use_openai:
                    answer = safe_openai_generation(
                        base, prompt_msgs,
                        max_new_tokens=args.max_new_tokens
                    )
                else:
                    answer = safe_run_generation(
                        model, tokenizer, prompt_msgs,
                        max_new_tokens=args.max_new_tokens
                    )

                new_obj = dict(j)
                new_obj["model_answer"] = answer
                fw.write(orjson.dumps(
                    new_obj, option=orjson.OPT_APPEND_NEWLINE
                ).decode("utf-8"))
        print("· Saved:", outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out_dir", default="rag_outputs")
    parser.add_argument("--retriever", choices=["tfidf", "bm25", "bge"],
                        default="bge")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--expand_with_links", action="store_true")

    parser.add_argument(
        "--oracle", action="store_true")
    parser.add_argument(
    "--openai_key_file", default="openai_api_key.txt")

    args = parser.parse_args()
    main(args)
