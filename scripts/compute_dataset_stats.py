#!/usr/bin/env python3
"""Compute dataset statistics for the SearchFireSafety release files."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            yield row


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def text_len(value: Any) -> int:
    return len(value) if isinstance(value, str) else 0


def word_len(value: Any) -> int:
    return len(value.split()) if isinstance(value, str) else 0


def list_len(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    return len([item for item in value if item is not None])


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def doc_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    bodies = [
        row.get("text") or row.get("chapter_body") or row.get("provision_text") or ""
        for row in rows
    ]
    related_counts = [list_len(row.get("related_doc_ids")) for row in rows]
    missing_related_key = [row for row in rows if "related_doc_ids" not in row]
    return {
        "total": len(rows),
        "avg_len": avg(text_len(body) for body in bodies),
        "avg_words": avg(word_len(body) for body in bodies),
        "avg_related": avg(related_counts),
        "missing_related_key": len(missing_related_key),
        "empty_related": sum(
            1 for row in rows if "related_doc_ids" in row and not row.get("related_doc_ids")
        ),
        "with_related": sum(1 for count in related_counts if count > 0),
    }


def qa_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    related_counts = [list_len(row.get("related_doc_ids")) for row in rows]
    return {
        "total": len(rows),
        "with_docs": sum(1 for count in related_counts if count > 0),
        "avg_q_len": avg(text_len(row.get("question")) for row in rows),
        "avg_a_len": avg(text_len(row.get("answer")) for row in rows),
        "avg_related": avg(related_counts),
    }


def multihop_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    related_counts = [list_len(row.get("related_doc_ids")) for row in rows]
    return {
        "total": len(rows),
        "avg_q_len": avg(text_len(row.get("question")) for row in rows),
        "avg_related": avg(related_counts),
    }


def print_missing_related(rows: List[Dict[str, Any]]) -> None:
    missing = [row for row in rows if "related_doc_ids" not in row]
    print()
    print("Legal documents missing related_doc_ids")
    print(f"Count: {len(missing)}")
    print("line_no\tdoc_id\tsemantic_id\tlaw_name\tchapter\tchapter_description")
    for row in missing:
        print(
            "\t".join(
                str(row.get(key, ""))
                for key in [
                    "_line_no",
                    "doc_id",
                    "semantic_id",
                    "law_name",
                    "chapter",
                    "chapter_description",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute current SearchFireSafety release statistics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing legal_docs.jsonl, realworld_qa.jsonl, and multihop_qa_mcq.jsonl.",
    )
    parser.add_argument(
        "--show-missing-related",
        action="store_true",
        help="Print legal document rows that omit the related_doc_ids field.",
    )
    args = parser.parse_args()

    legal_docs = load_rows(args.data_dir / "legal_docs.jsonl")
    realworld_qa = load_rows(args.data_dir / "realworld_qa.jsonl")
    multihop_qa = load_rows(args.data_dir / "multihop_qa_mcq.jsonl")

    docs = doc_stats(legal_docs)
    qa = qa_stats(realworld_qa)
    mh = multihop_stats(multihop_qa)

    print(f"Data directory: {args.data_dir}")
    print()
    print("Legal Documents")
    print(f"Total documents: {docs['total']:.0f}")
    print(f"Avg. document length: {docs['avg_len']:.2f} characters")
    print(f"Avg. words per document: {docs['avg_words']:.2f}")
    print(f"Avg. related documents: {docs['avg_related']:.2f}")
    print(f"Rows with related documents: {docs['with_related']:.0f}")
    print(f"Rows with empty related_doc_ids: {docs['empty_related']:.0f}")
    print(f"Rows missing related_doc_ids key: {docs['missing_related_key']:.0f}")
    print()
    print("Real-World Expert QA")
    print(f"Total pairs: {qa['total']:.0f}")
    print(f"Pairs with mapped documents: {qa['with_docs']:.0f}")
    print(f"Avg. question length: {qa['avg_q_len']:.2f} characters")
    print(f"Avg. answer length: {qa['avg_a_len']:.2f} characters")
    print(f"Avg. relevant docs per question: {qa['avg_related']:.2f}")
    print()
    print("Multi-Hop QA (MCQ)")
    print(f"Total pairs: {mh['total']:.0f}")
    print(f"Avg. question length: {mh['avg_q_len']:.2f} characters")
    print(f"Avg. relevant docs per question: {mh['avg_related']:.2f}")

    if args.show_missing_related:
        print_missing_related(legal_docs)


if __name__ == "__main__":
    main()
