from __future__ import annotations
import argparse, orjson, json, os, statistics, itertools
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import faiss
import torch        
import math, statistics

@dataclass
class ModelSpec:
    key: str
    arg_model_name_attr: str
    results_key: str

specs: List[ModelSpec] = [
    ModelSpec(
        key="bge",
        arg_model_name_attr="bge_model_name",
        results_key="bge-m3",
    ),
    ModelSpec(
        key="nomic",
        arg_model_name_attr="nomic_model_name",
        results_key="nomic-embed-v2",
    ),
    ModelSpec(
        key="qwen",
        arg_model_name_attr="qwen_model_name",
        results_key="qwen3-embedding-0.6B",
    ),
    ModelSpec(
        key="snow",
        arg_model_name_attr="snow_model_name",
        results_key="snowflake-arctic-embed-v2"
    ),
    ModelSpec(
        key="kure",
        arg_model_name_attr="kure_model_name",
        results_key="kure-v1"
    )
]

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

def load_queries(path: str) -> Tuple[List[str], List[List[int]]]:
    qs, rel_lists = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = orjson.loads(line)
            if j.get("has_matched_docs"):
                qs.append(j["question"])
                rel_lists.append(j["matched_doc_id"])
    return qs, rel_lists

def load_multihop_queries(path: str) -> Tuple[List[str], List[List[int]]]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            if ex.get("QA_Appropriateness") == "O":
                items.append(ex)

    questions: List[str] = []
    qrels_list: List[List[int]] = []

    for ex in items:
        q = ex.get("question")
        if not isinstance(q, str):
            continue
        q = q.strip()
        if not q:
            continue

        rel: List[int] = []
        for k in ("doc_id_from", "doc_id_to"):
            v = ex.get(k)
            if isinstance(v, int):
                rel.append(v)

        if rel:
            questions.append(q)
            qrels_list.append(rel)

    return questions, qrels_list

def build_tfidf(docs: List[str], max_feats: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
        max_features=max_feats,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    tfidf = vect.fit_transform(docs)
    return vect, tfidf

def search_tfidf(vect, tfidf, query: str, topk: int):
    q_vec = vect.transform([query])
    scores = q_vec @ tfidf.T              # [1, n_docs]
    row = scores.toarray().ravel()
    idx = np.argpartition(-row, range(topk))[:topk]
    idx = idx[np.argsort(-row[idx])]
    return idx, row[idx]

def build_bm25(docs: List[str], k1: float, b: float):
    from rank_bm25 import BM25Okapi
    tok_docs = [d.replace("\n", " ").split() for d in docs]
    bm25 = BM25Okapi(tok_docs, k1=k1, b=b)
    return bm25, tok_docs

def search_bm25(bm25, tokens, query: str, topk: int):
    q_tok = query.split()
    scores = bm25.get_scores(q_tok)
    idx = np.argpartition(-scores, range(topk))[:topk]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

def build_others_index(
    docs: List[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[faiss.IndexFlatIP, np.ndarray, "SentenceTransformer"]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    model.max_seq_length = 2048
    dim = model.get_sentence_embedding_dimension()

    embs = np.empty((len(docs), dim), dtype="float32")
    for s in tqdm(range(0, len(docs), batch_size), desc=f"Embedding docs ({model_name})"):
        batch = docs[s : s + batch_size]
        embs[s : s + batch_size] = model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index, embs, model

def search_others(index, model, query: str, topk: int):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q_emb, topk)
    return idx.flatten(), scores.flatten()

def expand_with_links(
    retrieved_ids: List[int],
    link_dict: Dict[int, List[int]],
    topk: int
) -> List[int]:
    seen = set()
    expanded: List[int] = []
    for did in retrieved_ids:
        if len(expanded) >= topk:
            break
        if did not in seen:
            expanded.append(did)
            seen.add(did)
        for linked in link_dict.get(did, []):
            if len(expanded) >= topk:
                break
            if linked not in seen:
                expanded.append(linked)
                seen.add(linked)
    return expanded

# def evaluate(
#     run: Dict[int, List[int]],
#     qrels: List[List[int]],
#     ks=(1, 2, 3, 5, 10, 20, 100),
# ) -> Dict[str, float]:
#     recalls = {k: [] for k in ks}
#     rr = []

#     for qid, rel in enumerate(qrels):
#         rel_set = set(rel)
#         retrieved = run[qid]

#         rank = next((i + 1 for i, d in enumerate(retrieved) if d in rel_set), None)
#         rr.append(0 if rank is None else 1 / rank)

#         for k in ks:
#             hit = len([d for d in retrieved[:k] if d in rel_set])
#             recalls[k].append(hit / len(rel_set))

#     metrics = {f"R@{k}": statistics.mean(recalls[k]) for k in ks}
#     metrics["MRR"] = statistics.mean(rr)
#     return metrics

def evaluate(
    run: Dict[int, List[int]],
    qrels: List[List[int]],
) -> Dict[str, float]:
    """
    Metrics:
      - R@K for K in {1, 10, 50, 100}  (fractional recall: hit/len(rel_set))
      - Both@K for K in {1, 10, 50, 100} (1 if all relevant docs appear within top-K, else 0)
      - MRR (first relevant rank reciprocal)
      - nDCG@K for K in {10, 20} (binary gain)
    """
    recall_ks = (1, 10, 50, 100)
    both_ks   = (1, 10, 50, 100)
    ndcg_ks   = (10, 20)

    # collectors
    recalls = {k: [] for k in recall_ks}
    boths   = {k: [] for k in both_ks}
    ndcgs   = {k: [] for k in ndcg_ks}
    rr_list = []

    for qid, rel in enumerate(qrels):
        rel_set = set(rel)
        if not rel_set:
            continue

        ranked = run.get(qid, []) or []

        # MRR
        first_rank = None
        for i, d in enumerate(ranked):
            if d in rel_set:
                first_rank = i + 1
                break
        rr_list.append(0.0 if first_rank is None else 1.0 / first_rank)

        # Recall@K / Both@K
        for k in recall_ks:
            topk = ranked[:k]
            hit_cnt = sum(1 for d in topk if d in rel_set)
            # fractional recall (히트 수 / 정답 수)
            recalls[k].append(hit_cnt / len(rel_set))
        for k in both_ks:
            topk = ranked[:k]
            both_hit = 1.0 if rel_set.issubset(set(topk)) else 0.0
            boths[k].append(both_hit)

        # nDCG@K (binary gain)
        for k in ndcg_ks:
            topk = ranked[:k]
            hits = [1 if d in rel_set else 0 for d in topk]
            # DCG = sum(h_i / log2(i+2))
            dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
            # IDCG: 상위 k에 올 수 있는 최대 관련문서 수
            R = min(len(rel_set), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(R)) if R > 0 else 1.0
            ndcgs[k].append(0.0 if idcg == 0 else dcg / idcg)

    def mean_or_zero(xs):
        return statistics.mean(xs) if xs else 0.0

    metrics = {}
    for k in recall_ks:
        metrics[f"R@{k}"] = mean_or_zero(recalls[k])
    for k in both_ks:
        metrics[f"Both@{k}"] = mean_or_zero(boths[k])
    for k in ndcg_ks:
        metrics[f"nDCG@{k}"] = mean_or_zero(ndcgs[k])
    metrics["MRR"] = mean_or_zero(rr_list)

    return metrics

def run_one_model(
    spec: ModelSpec,
    args,
    doc_texts: List[str],
    doc_ids: List[str],
    queries: List[str],
    link_dict: Dict[str, List[str]],
    rel_lists: Dict[int, List[str]],
    results: Dict[str, Dict],
    rows: List[Dict],
):
    model_name = getattr(args, spec.arg_model_name_attr)
    print(f"\n🟢 {model_name} Embedding")

    index, _, model = build_others_index(
        doc_texts,
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
    )

    run: Dict[int, List[str]] = {}
    for qid, q in enumerate(tqdm(queries, desc=f"{model_name} Search")):
        idx, _ = search_others(index, model, q, args.topk)
        retrieved = [doc_ids[i] for i in idx]
        if args.expand_links:
            retrieved = expand_with_links(retrieved, link_dict, args.topk)
        run[qid] = retrieved

    metrics = evaluate(run, rel_lists)
    results[spec.results_key] = metrics
    rows.append(metrics)
    print(f"{spec.results_key}:", json.dumps(metrics, ensure_ascii=False, indent=2))

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--docs",        required=True)
    p.add_argument("--queries",     required=True)
    p.add_argument("--topk",        type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--methods", default="tfidf,bm25,bge,nomic,qwen,snow")
    p.add_argument("--expand_links", action="store_true")
    p.add_argument("--tfidf_max_features", type=int, default=120_000)
    p.add_argument("--bm25_k1", type=float, default=1.5)
    p.add_argument("--bm25_b",  type=float, default=0.75)
    p.add_argument("--bge_model_name", default="BAAI/bge-m3")
    p.add_argument("--nomic_model_name", default="nomic-ai/nomic-embed-text-v2-moe")
    p.add_argument("--qwen_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--snow_model_name", default="Snowflake/snowflake-arctic-embed-l-v2.0")
    p.add_argument("--kure_model_name", default="nlpai-lab/KURE-v1")

    args = p.parse_args()

    doc_ids, doc_texts, link_dict = load_docs(args.docs)
    queries, rel_lists = load_multihop_queries(args.queries)
    print(f"Loaded {len(doc_ids)} documents, and {len(queries)} queries.")

    results, rows = {}, []

    wanted = [m.strip().lower() for m in args.methods.split(",")]

    if "tfidf" in wanted:
        print("\n🟢 TF‑IDF Indexing")
        tf_vect, tf_mat = build_tfidf(doc_texts, args.tfidf_max_features)
        tf_run = {}
        for qid, q in enumerate(tqdm(queries, desc="TF‑IDF Search")):
            idx, _ = search_tfidf(tf_vect, tf_mat, q, args.topk)
            retrieved = [doc_ids[i] for i in idx]
            if args.expand_links:
                retrieved = expand_with_links(retrieved, link_dict, args.topk)
            tf_run[qid] = retrieved
        metrics = evaluate(tf_run, rel_lists)
        results["TF‑IDF"] = metrics
        rows.append(metrics)
        print("TF‑IDF:", json.dumps(metrics, ensure_ascii=False, indent=2))

    if "bm25" in wanted:
        print("\n🟢 BM25 Indexing (k1={:.2f}, b={:.2f})".format(args.bm25_k1, args.bm25_b))
        bm25, tok_docs = build_bm25(doc_texts, args.bm25_k1, args.bm25_b)
        bm_run = {}
        for qid, q in enumerate(tqdm(queries, desc="BM25 Search")):
            idx, _ = search_bm25(bm25, tok_docs, q, args.topk)
            retrieved = [doc_ids[i] for i in idx]
            if args.expand_links:
                retrieved = expand_with_links(retrieved, link_dict, args.topk)
            bm_run[qid] = retrieved
        metrics = evaluate(bm_run, rel_lists)
        results["BM25"] = metrics
        rows.append(metrics)
        print("BM25:", json.dumps(metrics, ensure_ascii=False, indent=2))

    for spec in specs:
        if spec.key in wanted:
            run_one_model(
                spec, args, doc_texts, doc_ids, queries, link_dict, rel_lists, results, rows
            )      

    df = pd.DataFrame(rows, index=list(results))
    print("\n=== Final Results ===")
    print(df.round(4).to_markdown())

    out = Path("ir_metrics.csv")
    df.to_csv(out, index=True)
    print(f"\n➡️ Results saved to {out}.")

if __name__ == "__main__":
    main()
