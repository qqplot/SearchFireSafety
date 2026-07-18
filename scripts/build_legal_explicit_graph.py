#!/usr/bin/env python3
"""
Build an explicit legal-document graph from corpus annotations.

The graph uses only `data/legal_docs.jsonl`. Each document's
`related_doc_ids` field defines explicit document-to-document links:

    source doc_id = a, related_doc_ids = [b, c]
    edges: a -> b, a -> c

Default output is unweighted:

    {
        "nodes": [{... legal doc row ...}, ...],
        "graph": {doc_id: [neighbor_doc_id, ...]},
        "metadata": {...}
    }

If you need compatibility with older retrieval code that expects
`nodes` as a pandas DataFrame and `(neighbor, weight)` graph tuples,
pass `--nodes-format dataframe --edge-format weighted`.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LEGAL_DOCS = Path("data/legal_docs.jsonl")
DEFAULT_OUTPUT = Path("legal_explicit_graph.pkl")


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


def make_full_text(row: dict[str, Any]) -> str:
    semantic_id = (row.get("semantic_id") or "").strip()
    law_name = (row.get("law_name") or "").strip()
    chapter = (row.get("chapter") or "").strip()
    chapter_description = (row.get("chapter_description") or "").strip()
    body = row.get("text") or row.get("chapter_body") or ""

    header_parts = [p for p in [semantic_id, law_name, chapter, chapter_description] if p]
    if header_parts:
        return f"{' | '.join(header_parts)}\n{body}"
    return str(body)


def build_node_rows(legal_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for item in legal_docs:
        doc_id = int(item["doc_id"])
        if doc_id in seen:
            raise ValueError(f"Duplicate doc_id value in legal docs: {doc_id}")
        seen.add(doc_id)

        row = dict(item)
        row["doc_id"] = doc_id
        row["full_text"] = make_full_text(item)
        rows.append(row)

    return sorted(rows, key=lambda row: row["doc_id"])


def format_nodes(node_rows: list[dict[str, Any]], nodes_format: str) -> Any:
    if nodes_format == "records":
        return node_rows

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "--nodes-format dataframe requires pandas. Install pandas in the "
            "environment used to build the graph, or use --nodes-format records."
        ) from exc

    return pd.DataFrame(node_rows).set_index("doc_id").sort_index()


def build_legal_reference_graph(
    legal_docs: list[dict[str, Any]],
    valid_doc_ids: set[int],
    undirected: bool,
) -> tuple[dict[int, set[int]], dict[str, Any]]:
    graph: dict[int, set[int]] = defaultdict(set)
    unknown_doc_ids = Counter()
    source_with_edges = 0
    relation_len_counter = Counter()
    referenced_targets = set()
    self_loops = 0

    for item in legal_docs:
        source_id = int(item["doc_id"])
        doc_ids = dedupe_ints(item.get("related_doc_ids") or [])
        relation_len_counter[len(doc_ids)] += 1
        if doc_ids:
            source_with_edges += 1

        for target_id in doc_ids:
            if target_id == source_id:
                self_loops += 1
                continue
            if target_id in valid_doc_ids:
                graph[source_id].add(target_id)
                referenced_targets.add(target_id)
                if undirected:
                    graph[target_id].add(source_id)
            else:
                unknown_doc_ids[target_id] += 1

    adjacency_entries = sum(len(neighbors) for neighbors in graph.values())
    stats = {
        "legal_doc_count": len(legal_docs),
        "docs_with_related_doc_ids": source_with_edges,
        "related_doc_count_histogram": dict(sorted(relation_len_counter.items())),
        "referenced_target_doc_count": len(referenced_targets),
        "graph_node_count": len(graph),
        "adjacency_entry_count": adjacency_entries,
        "directed": not undirected,
        "self_loop_count": self_loops,
        "unknown_doc_id_count": sum(unknown_doc_ids.values()),
        "unknown_doc_ids_top20": unknown_doc_ids.most_common(20),
    }
    return graph, stats


def format_graph(
    graph_sets: dict[int, set[int]],
    edge_format: str,
    weight: float,
) -> dict[int, list[int] | list[tuple[int, float]]]:
    out = {}
    for doc_id in sorted(graph_sets):
        neighbors = sorted(graph_sets[doc_id])
        if edge_format == "weighted":
            out[doc_id] = [(neighbor, weight) for neighbor in neighbors]
        else:
            out[doc_id] = neighbors
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit graph from legal_docs.jsonl related_doc_ids."
    )
    parser.add_argument("--legal-docs", type=Path, default=DEFAULT_LEGAL_DOCS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Add reverse edges. By default, legal related_doc_ids are kept directed.",
    )
    parser.add_argument(
        "--nodes-format",
        choices=["records", "dataframe"],
        default="records",
        help="Use dataframe for compatibility with older eval_retrieval.py code.",
    )
    parser.add_argument(
        "--edge-format",
        choices=["unweighted", "weighted"],
        default="unweighted",
        help="Use weighted if older retrieval code expects (neighbor, weight) tuples.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Weight to use only when --edge-format weighted is selected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    legal_docs = read_jsonl(args.legal_docs)

    node_rows = build_node_rows(legal_docs)
    valid_doc_ids = set(int(row["doc_id"]) for row in node_rows)
    nodes = format_nodes(node_rows, args.nodes_format)

    graph_sets, stats = build_legal_reference_graph(legal_docs, valid_doc_ids, args.undirected)
    graph = format_graph(graph_sets, args.edge_format, args.weight)

    payload = {
        "nodes": nodes,
        "graph": graph,
        "metadata": {
            "source": "legal_docs.related_doc_ids",
            "legal_docs": str(args.legal_docs),
            "nodes_format": args.nodes_format,
            "edge_format": args.edge_format,
            "directed": not args.undirected,
            "edge_weight": args.weight if args.edge_format == "weighted" else None,
            "stats": stats,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved graph to {args.output}")
    print(f"Legal docs: {len(node_rows)}")
    print(f"Docs with related_doc_ids: {stats['docs_with_related_doc_ids']}")
    print(f"Referenced target docs: {stats['referenced_target_doc_count']}")
    print(f"Graph nodes with edges: {stats['graph_node_count']}")
    print(f"Adjacency entries: {stats['adjacency_entry_count']}")
    print(f"Directed: {stats['directed']}")
    print(f"Edge format: {args.edge_format}")
    if stats["unknown_doc_id_count"]:
        print(f"Unknown doc id references: {stats['unknown_doc_id_count']}")


if __name__ == "__main__":
    main()
