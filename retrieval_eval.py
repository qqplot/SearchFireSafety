from __future__ import annotations
import argparse, orjson, json, os, statistics, itertools
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import faiss
import torch                        
from transformers import (         
    AutoModel,
    AutoTokenizer,
)

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

def build_bge_index(
    docs: List[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[faiss.IndexFlatIP, np.ndarray, "SentenceTransformer"]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
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

def search_bge(index, model, query: str, topk: int):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idx = index.search(q_emb, topk)
    return idx.flatten(), scores.flatten()

def _encode_passages(
    texts: List[str],
    model: "torch.nn.Module",
    tokenizer: "AutoTokenizer",
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    all_embs = []
    with torch.no_grad():
        try:
            for s in tqdm(range(0, len(texts), batch_size), desc="Embedding docs (DPR)"):
                batch = texts[s : s + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)
                emb = model(**inputs).last_hidden_state[:, 0, :]          # CLS
                all_embs.append(emb.cpu())
        except:
            for s in tqdm(range(0, len(texts), batch_size), desc="Embedding docs (DPR)"):
                batch = texts[s : s + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                emb = model(**inputs).last_hidden_state[:, 0, :]          # CLS
                all_embs.append(emb.cpu())
    embs = torch.cat(all_embs, dim=0).contiguous().numpy().astype("float32")
    faiss.normalize_L2(embs)     # cosine ≒ inner‑product
    return embs

def build_dpr_index(
    docs: List[str],
    context_encoder_path: str,
    device: str,
    batch_size: int,
) -> Tuple[faiss.IndexFlatIP, "AutoModel", "AutoTokenizer"]:
    if not os.path.isdir(context_encoder_path):
        raise FileNotFoundError(f"[DPR] context_encoder_path not found: {context_encoder_path}")
    ctx_tok   = AutoTokenizer.from_pretrained(context_encoder_path, local_files_only=True)
    ctx_model = AutoModel.from_pretrained(context_encoder_path, local_files_only=True).to(device)

    dim = ctx_model.config.hidden_size
    embs = _encode_passages(docs, ctx_model, ctx_tok, device, batch_size)

    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    del ctx_model
    torch.cuda.empty_cache()

    return index

def search_dpr(
    index: faiss.IndexFlatIP,
    query: str,
    q_model: "AutoModel",
    q_tok: "AutoTokenizer",
    device: str,
    topk: int,
):
    q_model.eval()
    with torch.no_grad():
        inp = q_tok(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        q_emb = q_model(**inp).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        faiss.normalize_L2(q_emb)
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

def evaluate(
    run: Dict[int, List[int]],
    qrels: List[List[int]],
    ks=(1, 2, 3, 5, 10, 20, 100),
) -> Dict[str, float]:
    recalls = {k: [] for k in ks}
    rr = []

    for qid, rel in enumerate(qrels):
        rel_set = set(rel)
        retrieved = run[qid]

        rank = next((i + 1 for i, d in enumerate(retrieved) if d in rel_set), None)
        rr.append(0 if rank is None else 1 / rank)

        for k in ks:
            hit = len([d for d in retrieved[:k] if d in rel_set])
            recalls[k].append(hit / len(rel_set))

    metrics = {f"R@{k}": statistics.mean(recalls[k]) for k in ks}
    metrics["MRR"] = statistics.mean(rr)
    return metrics

def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--docs",        required=True)
    p.add_argument("--queries",     required=True)
    p.add_argument("--topk",        type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--methods", default="tfidf,bm25,bge,dpr")
    p.add_argument("--expand_links", action="store_true")
    p.add_argument("--tfidf_max_features", type=int, default=120_000)
    p.add_argument("--bm25_k1", type=float, default=1.5)
    p.add_argument("--bm25_b",  type=float, default=0.75)
    p.add_argument("--bge_model_name", default="BAAI/bge-m3")

    p.add_argument("--dpr_context_encoder_path")
    p.add_argument("--dpr_question_encoder_path")

    args = p.parse_args()

    doc_ids, doc_texts, link_dict = load_docs(args.docs)
    queries, rel_lists = load_queries(args.queries)
    print(f"Loaded {len(doc_ids)} documents, and {len(queries)} queries.")
  
    id2idx = {d: i for i, d in enumerate(doc_ids)}

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

    if "bge" in wanted:
        print(f"\n🟢 {args.bge_model_name} Embedding")
        bge_index, _, bge_model = build_bge_index(
            doc_texts,
            model_name=args.bge_model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
        bge_run = {}
        for qid, q in enumerate(tqdm(queries, desc=f"{args.bge_model_name} Search")):
            idx, _ = search_bge(bge_index, bge_model, q, args.topk)
            retrieved = [doc_ids[i] for i in idx]
            if args.expand_links:
                retrieved = expand_with_links(retrieved, link_dict, args.topk)
            bge_run[qid] = retrieved
        metrics = evaluate(bge_run, rel_lists)
        results["BGE‑m3"] = metrics
        rows.append(metrics)
        print("BGE‑m3:", json.dumps(metrics, ensure_ascii=False, indent=2))

    if "dpr" in wanted:
        print("\n🟢 DPR Indexing")
        dpr_index = build_dpr_index(
            doc_texts,
            context_encoder_path=args.dpr_context_encoder_path,
            device=args.device,
            batch_size=args.batch_size,
        )
        if not os.path.isdir(args.dpr_question_encoder_path):
            raise FileNotFoundError(f"[DPR] question_encoder_path not found: {args.dpr_question_encoder_path}")
        dpr_q_tok   = AutoTokenizer.from_pretrained(args.dpr_question_encoder_path, local_files_only=True)
        dpr_q_model = AutoModel.from_pretrained(args.dpr_question_encoder_path, local_files_only=True).to(args.device)

        dpr_run = {}
        for qid, q in enumerate(tqdm(queries, desc="DPR Search")):
            idx, _ = search_dpr(
                dpr_index,
                q,
                dpr_q_model,
                dpr_q_tok,
                device=args.device,
                topk=args.topk,
            )
            retrieved = [doc_ids[i] for i in idx]
            if args.expand_links:
                retrieved = expand_with_links(retrieved, link_dict, args.topk)
            dpr_run[qid] = retrieved

        metrics = evaluate(dpr_run, rel_lists)
        results["DPR"] = metrics
        rows.append(metrics)
        print("DPR:", json.dumps(metrics, ensure_ascii=False, indent=2))

    df = pd.DataFrame(rows, index=list(results))
    print("\n=== Final Results ===")
    print(df.round(4).to_markdown())

    out = Path("ir_metrics.csv")
    df.to_csv(out, index=True)
    print(f"\n➡️ Results saved to {out}.")

if __name__ == "__main__":
    main()
