#!/usr/bin/env python3
"""
Evaluate Baseline, Rocchio, and SAR(Explicit) on real-world QA.

Expected graph pickle format:

    {
        "nodes": list[dict] or pandas.DataFrame,
        "graph": {doc_id: [neighbor_doc_id, ...]}
                 or {doc_id: [(neighbor_doc_id, weight), ...]},
        "metadata": {...}
    }

The graph can be generated with:

    python3 scripts/build_legal_explicit_graph.py \
      --output legal_explicit_graph.pkl
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GRAPH_PKL = Path("legal_explicit_graph.pkl")
DEFAULT_QA_FILE = Path("data/realworld_qa.jsonl")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_METRIC_K = [10, 20, 50]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def dedupe_ints(values: list[Any]) -> list[int]:
    out = []
    seen = set()
    for value in values or []:
        if value is None:
            continue
        doc_id = int(value)
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def load_graph_payload(path: Path) -> tuple[list[int], list[str], dict[int, list[int]], dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    nodes = payload["nodes"]
    graph_raw = payload["graph"]
    metadata = payload.get("metadata", {})

    if hasattr(nodes, "iterrows"):
        rows = []
        for doc_id, row in nodes.iterrows():
            item = row.to_dict()
            item["doc_id"] = int(doc_id)
            rows.append(item)
    elif isinstance(nodes, list):
        rows = [dict(row) for row in nodes]
    else:
        raise TypeError("Unsupported nodes format. Expected list[dict] or pandas.DataFrame.")

    rows.sort(key=lambda row: int(row["doc_id"]))
    idx_to_docid = [int(row["doc_id"]) for row in rows]
    corpus_texts = [row.get("full_text") or row.get("text") or "" for row in rows]

    graph: dict[int, list[int]] = {}
    for raw_doc_id, raw_neighbors in graph_raw.items():
        doc_id = int(raw_doc_id)
        neighbors = []
        for item in raw_neighbors:
            if isinstance(item, (tuple, list)):
                neighbor = item[0]
            else:
                neighbor = item
            neighbors.append(int(neighbor))
        graph[doc_id] = neighbors

    return idx_to_docid, corpus_texts, graph, metadata


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (x / denom).astype("float32", copy=False)


def encode_texts(model_name: str, texts: list[str], batch_size: int, max_seq_len: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is required for embedding retrieval. "
            "Install it in the environment used to run evaluation."
        ) from exc

    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.astype("float32", copy=False)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.array([], dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64, copy=False)


def calculate_metrics(retrieved_ids: list[int], true_ids: list[int], metric_k: list[int]) -> dict[str, float]:
    true_set = set(true_ids)
    metrics = {}
    for k in metric_k:
        topk = retrieved_ids[:k]
        if not true_set:
            metrics[f"R@{k}"] = 0.0
            metrics[f"nDCG@{k}"] = 0.0
            metrics[f"MRR@{k}"] = 0.0
            continue

        hits = sum(1 for doc_id in topk if doc_id in true_set)
        metrics[f"R@{k}"] = hits / len(true_set)

        dcg = 0.0
        for rank, doc_id in enumerate(topk, start=1):
            if doc_id in true_set:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(true_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        metrics[f"nDCG@{k}"] = dcg / idcg if idcg > 0 else 0.0

        rr = 0.0
        for rank, doc_id in enumerate(topk, start=1):
            if doc_id in true_set:
                rr = 1.0 / rank
                break
        metrics[f"MRR@{k}"] = rr
    return metrics


def run_rocchio(
    query_vec: np.ndarray,
    corpus_emb: np.ndarray,
    init_idx: np.ndarray,
    idx_to_docid: list[int],
    top_k: int,
    alpha: float,
    beta: float,
    mode: str,
    top_k_freeze: int,
    return_k: int,
) -> list[int]:
    feedback = corpus_emb[init_idx[:top_k]]
    avg_doc = feedback.mean(axis=0)
    new_query = (alpha * query_vec) + (beta * avg_doc)
    new_query = new_query.reshape(1, -1).astype("float32", copy=False)
    new_query = normalize_rows(new_query)[0]

    if mode == "candidate":
        candidate_idx = init_idx[:return_k]
        candidate_scores = np.maximum(corpus_emb[candidate_idx] @ new_query, 0.0)
        if top_k_freeze > 0:
            freeze_n = min(top_k_freeze, candidate_scores.shape[0])
            candidate_scores[:freeze_n] = candidate_scores[:freeze_n] + 100.0
        order = np.argsort(-candidate_scores)
        reranked_idx = candidate_idx[order]
    elif mode == "full":
        scores = np.maximum(corpus_emb @ new_query, 0.0)
        reranked_idx = topk_indices(scores, return_k)
    else:
        raise ValueError(f"Unsupported Rocchio mode: {mode}")

    return [idx_to_docid[int(idx)] for idx in reranked_idx]


def run_sar_explicit(
    scores: np.ndarray,
    init_idx: np.ndarray,
    idx_to_docid: list[int],
    docid_to_idx: dict[int, int],
    graph: dict[int, list[int]],
    indegree: Counter,
    sar_initial_k: int,
    sar_beta: float,
    top_k_freeze: int,
    max_pool_size: int,
    return_k: int,
) -> list[int]:
    candidate_pool = {int(idx): float(scores[int(idx)]) for idx in init_idx}
    bonuses: dict[int, float] = {}

    for p_idx_raw in init_idx[:sar_initial_k]:
        p_idx = int(p_idx_raw)
        p_id = idx_to_docid[p_idx]
        p_score = float(scores[p_idx])
        neighbors = graph.get(p_id, [])
        if not neighbors:
            continue

        penalty_out = math.log(len(neighbors) + 1) if len(neighbors) > 1 else 1.0
        vote_power = p_score / penalty_out

        for n_id in neighbors:
            c_idx = docid_to_idx.get(int(n_id))
            if c_idx is None or c_idx == p_idx:
                continue

            pop = indegree.get(int(n_id), 1)
            penalty_in = math.log(pop + 1) if pop > 1 else 1.0
            bonuses[c_idx] = bonuses.get(c_idx, 0.0) + (vote_power / penalty_in)

            if c_idx not in candidate_pool and len(candidate_pool) < max_pool_size:
                candidate_pool[c_idx] = 0.0

    top_freeze_idx = {int(idx) for idx in init_idx[:top_k_freeze]}
    reranked = []
    for idx in candidate_pool:
        base_s = float(scores[idx])
        bonus = float(bonuses.get(idx, 0.0))
        if idx in top_freeze_idx:
            final_s = base_s + 100.0
        else:
            final_s = base_s + (sar_beta * bonus * (1.0 - base_s))
        reranked.append((idx, final_s))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return [idx_to_docid[idx] for idx, _ in reranked[:return_k]]


def indegree_from_graph(graph: dict[int, list[int]]) -> Counter:
    indegree = Counter()
    for neighbors in graph.values():
        for neighbor in neighbors:
            indegree[int(neighbor)] += 1
    return indegree


def print_results(results: dict[str, dict[str, float]], metric_keys: list[str]) -> None:
    method_width = max(len("Method"), *(len(method) for method in results))
    col_width = 10
    header = " ".join(["Method".ljust(method_width)] + [key.rjust(col_width) for key in metric_keys])
    print("\n" + header)
    print("-" * len(header))
    for method, values in results.items():
        row = [method.ljust(method_width)]
        row.extend(f"{values[key] * 100.0:10.2f}" for key in metric_keys)
        print(" ".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Baseline, Rocchio, and SAR(Explicit) on realworld QA."
    )
    parser.add_argument("--graph-pkl", type=Path, default=DEFAULT_GRAPH_PKL)
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--metric-k", type=int, nargs="+", default=DEFAULT_METRIC_K)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--corpus-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--init-retrieval-k", type=int, default=100)
    parser.add_argument("--return-k", type=int, default=100)
    parser.add_argument("--rocchio-top-k", type=int, default=15)
    parser.add_argument("--rocchio-alpha", type=float, default=0.9)
    parser.add_argument("--rocchio-beta", type=float, default=0.1)
    parser.add_argument(
        "--rocchio-top-k-freeze",
        type=int,
        default=0,
        help="Keep the initial top-N dense results above other Rocchio-reranked candidates.",
    )
    parser.add_argument(
        "--rocchio-mode",
        choices=["candidate", "full"],
        default="candidate",
        help=(
            "candidate reranks only the initial top-k candidate set; "
            "full reruns retrieval over the whole corpus after query expansion."
        ),
    )
    parser.add_argument("--sar-initial-k", "--cir-initial-k", dest="sar_initial_k", type=int, default=15)
    parser.add_argument("--sar-beta", "--cir-beta", dest="sar_beta", type=float, default=0.3)
    parser.add_argument("--top-k-freeze", type=int, default=5)
    parser.add_argument(
        "--max-pool-size",
        type=int,
        default=100,
        help="Candidate-pool cap. The default 100 gives strict top-100 reranking.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    idx_to_docid, corpus_texts, graph, metadata = load_graph_payload(args.graph_pkl)
    docid_to_idx = {doc_id: idx for idx, doc_id in enumerate(idx_to_docid)}
    indegree = indegree_from_graph(graph)

    qa_rows = read_jsonl(args.qa_file)
    questions = [row["question"] for row in qa_rows]
    true_doc_ids = [dedupe_ints(row.get("related_doc_ids") or []) for row in qa_rows]

    print(f"Graph: {args.graph_pkl}")
    print(f"Graph source: {metadata.get('source', '(unknown)')}")
    print(f"Docs: {len(idx_to_docid)}")
    print(f"Graph nodes with edges: {len(graph)}")
    print(f"Directed adjacency entries: {sum(len(v) for v in graph.values())}")
    print(f"QA rows: {len(qa_rows)}")
    print(f"Model: {args.model_name}")

    print("\nEncoding corpus...")
    corpus_emb = encode_texts(
        args.model_name,
        corpus_texts,
        batch_size=args.corpus_batch_size,
        max_seq_len=args.max_seq_len,
    )

    print("\nEncoding queries...")
    query_emb = encode_texts(
        args.model_name,
        questions,
        batch_size=args.query_batch_size,
        max_seq_len=args.max_seq_len,
    )

    metric_keys = []
    for k in args.metric_k:
        metric_keys.extend([f"R@{k}", f"nDCG@{k}", f"MRR@{k}"])

    per_method = {
        "Baseline": {key: [] for key in metric_keys},
        "Rocchio": {key: [] for key in metric_keys},
        "SAR(Explicit)": {key: [] for key in metric_keys},
    }

    for i, q_vec in enumerate(query_emb):
        scores = np.maximum(corpus_emb @ q_vec, 0.0)
        init_idx = topk_indices(scores, args.init_retrieval_k)

        baseline_ids = [idx_to_docid[int(idx)] for idx in init_idx[: args.return_k]]
        rocchio_ids = run_rocchio(
            q_vec,
            corpus_emb,
            init_idx,
            idx_to_docid,
            top_k=args.rocchio_top_k,
            alpha=args.rocchio_alpha,
            beta=args.rocchio_beta,
            mode=args.rocchio_mode,
            top_k_freeze=args.rocchio_top_k_freeze,
            return_k=args.return_k,
        )
        sar_ids = run_sar_explicit(
            scores,
            init_idx,
            idx_to_docid,
            docid_to_idx,
            graph,
            indegree,
            sar_initial_k=args.sar_initial_k,
            sar_beta=args.sar_beta,
            top_k_freeze=args.top_k_freeze,
            max_pool_size=args.max_pool_size,
            return_k=args.return_k,
        )

        for method, retrieved in [
            ("Baseline", baseline_ids),
            ("Rocchio", rocchio_ids),
            ("SAR(Explicit)", sar_ids),
        ]:
            metrics = calculate_metrics(retrieved, true_doc_ids[i], args.metric_k)
            for key, value in metrics.items():
                per_method[method][key].append(value)

    results = {
        method: {key: float(np.mean(values[key])) for key in metric_keys}
        for method, values in per_method.items()
    }
    print_results(results, metric_keys)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "graph_pkl": str(args.graph_pkl),
                    "qa_file": str(args.qa_file),
                    "model_name": args.model_name,
                    "metric_k": args.metric_k,
                    "rocchio_mode": args.rocchio_mode,
                    "rocchio_top_k": args.rocchio_top_k,
                    "rocchio_alpha": args.rocchio_alpha,
                    "rocchio_beta": args.rocchio_beta,
                    "rocchio_top_k_freeze": args.rocchio_top_k_freeze,
                    "sar_initial_k": args.sar_initial_k,
                    "sar_beta": args.sar_beta,
                    "top_k_freeze": args.top_k_freeze,
                    "max_pool_size": args.max_pool_size,
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nSaved JSON results to {args.save_json}")


if __name__ == "__main__":
    main()
