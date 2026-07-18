# SearchFireSafety (ACL 2026)

Official dataset repository for the ACL 2026 paper:
**Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA**

- Preprint: https://arxiv.org/abs/2604.06173
- Paper: https://aclanthology.org/2026.acl-long.2112/

## Overview

SearchFireSafety is a benchmark for statute-centric legal QA in the Korean fire-safety domain.
The dataset is designed to evaluate:

- Structure-aware retrieval over citation-linked legal documents
- Multi-hop reasoning across delegated statutory provisions
- Safe abstention behavior under partial/incomplete context

## Repository Scope

This repository is organized as a **dataset archive**.
The core release is under `data/`:

- `data/legal_docs.jsonl`: legal corpus (article-level units) + citation links
- `data/realworld_qa.jsonl`: real-world expert QA pairs
- `data/multihop_qa_mcq.jsonl`: synthetic multi-hop MCQ for safety evaluation

## Installation

The dataset statistics script and default graph builder use only the Python
standard library. To run dense retrieval evaluation, install the minimal runtime
dependencies:

```bash
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install PyTorch for your platform
first, then install the requirements above.

## Dataset Statistics

Recompute these statistics with:

```bash
python scripts/compute_dataset_stats.py
```

| Category | Statistic | Number |
|---|---|---:|
| Legal Documents | Total documents | 4,468 |
| Legal Documents | Avg. document length | 477.9 characters |
| Legal Documents | Avg. words per document | 103.2 |
| Legal Documents | Avg. related documents | 1.8 |
| Real-World Expert QA | Total pairs | 876 |
| Real-World Expert QA | Avg. question length | 90.7 characters |
| Real-World Expert QA | Avg. answer length | 278.1 characters |
| Real-World Expert QA | Avg. relevant docs per question | 1.5 |
| Multi-Hop QA (MCQ) | Total pairs | 3,395 |
| Multi-Hop QA (MCQ) | Avg. question length | 51.1 characters |
| Multi-Hop QA (MCQ) | Relevant docs per question | 2.0 |

## File Formats

### 1) `legal_docs.jsonl`

Article-level legal corpus entries.

| Field | Type | Description |
|---|---|---|
| `doc_id` | int | Unique document unit ID |
| `semantic_id` | string | Human-readable legal identifier |
| `collection_name` | string | Parent legal collection |
| `law_level` | string | Legal hierarchy level (e.g., Act, Decree, Rule) |
| `law_name` | string | Law title |
| `chapter` | string | Article/appendix label |
| `chapter_description` | string | Article heading |
| `text` | string | Legal text |
| `related_doc_ids` | int[] | Citation/delegation-linked `doc_id` list |

Notes:

- 1,728 rows contain at least one outgoing `related_doc_ids` entry.
- 2,740 rows have an empty `related_doc_ids` list.
- `related_doc_ids` defines graph edges used for structure-aware retrieval.

### 2) `realworld_qa.jsonl`

Real-world public petition questions with official NFA answers.

| Field | Type | Description |
|---|---|---|
| `question_id` | int | Question ID |
| `question` | string | User question |
| `answer` | string | Official expert answer |
| `related_doc_ids` | int[] | Supporting legal document IDs |
| `semantic_ids` | string[] | Supporting semantic identifiers |

### 3) `multihop_qa_mcq.jsonl`

Synthetic multiple-choice QA designed to test strict multi-hop dependency.

| Field | Type | Description |
|---|---|---|
| `question_id` | int | Question ID |
| `related_doc_ids` | int[] | Source document IDs used to construct the question |
| `related_semantic_ids` | string[] | Semantic identifiers for source docs |
| `question` | string | MCQ question |
| `option_1` ~ `option_5` | string | Five answer options |
| `answer_full` | int (1-5) | Correct option under full context |
| `answer_partial` | int (1-5) | Correct option under partial context |

Notes:

- For all 3,395 rows, `answer_partial = 5` ("Cannot be answered with the given information").
- This setup explicitly evaluates safe abstention under missing evidence.

## Structure-Aware Reranking Evaluation

This repository also includes lightweight scripts for evaluating
**Structure-Aware Reranking (SAR)** on the real-world QA split.

SAR is evaluated as a strict top-100 reranker in this release. The dense
retriever first retrieves the top 100 documents. SAR then keeps the same
candidate set and reranks it using explicit document links from
`legal_docs.jsonl`. The real-world QA labels are used only for evaluation, not
for graph construction.

### Build the Explicit Graph

Directed graph:

```bash
python scripts/build_legal_explicit_graph.py \
  --output legal_explicit_graph.pkl
```

Undirected graph:

```bash
python scripts/build_legal_explicit_graph.py \
  --undirected \
  --output legal_explicit_graph_undirected.pkl
```

The directed graph contains:

- 4,468 legal document nodes
- 1,728 documents with at least one `related_doc_ids` entry
- 1,650 referenced target documents
- 1,725 graph nodes with outgoing edges
- 8,114 directed adjacency entries

The undirected graph contains:

- 4,468 legal document nodes
- 1,728 documents with at least one `related_doc_ids` entry
- 1,650 referenced target documents
- 2,276 graph nodes with edges
- 15,262 undirected adjacency entries

### Run Retrieval Evaluation

```bash
python scripts/eval_realworld_graph_retrieval.py \
  --graph-pkl legal_explicit_graph.pkl \
  --qa-file data/realworld_qa.jsonl \
  --model-name BAAI/bge-m3 \
  --save-json results_legal_graph_sar_top100.json
```

For the undirected graph, replace `--graph-pkl` and `--save-json`:

```bash
python scripts/eval_realworld_graph_retrieval.py \
  --graph-pkl legal_explicit_graph_undirected.pkl \
  --qa-file data/realworld_qa.jsonl \
  --model-name BAAI/bge-m3 \
  --save-json results_legal_graph_sar_top100_undirected.json
```

### Results

The table below reports retrieval performance on `data/realworld_qa.jsonl`
using `BAAI/bge-m3`. Values are percentages.

| Method | R@10 | nDCG@10 | MRR@10 | R@20 | nDCG@20 | MRR@20 | R@50 | nDCG@50 | MRR@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 53.14 | 37.35 | 35.26 | 61.46 | 39.61 | 35.83 | 72.66 | 42.07 | 36.16 |
| Rocchio | 46.66 | 31.22 | 28.64 | 57.18 | 34.06 | 29.40 | 69.57 | 36.78 | 29.78 |
| SAR (Directed) | 55.45 | 38.22 | 35.59 | 63.62 | 40.47 | 36.15 | 73.46 | 42.65 | 36.45 |
| SAR (Undirected) | 54.18 | 37.77 | 35.37 | 62.97 | 40.18 | 35.97 | 73.40 | 42.48 | 36.30 |


## Citation

If you use this dataset, please cite the ACL 2026 paper.

```bibtex
@inproceedings{chae-etal-2026-evaluating,
    title = "Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal {QA}",
    author = "Chae, Kyubyung  and
      Yeom, Jewon  and
      Park, Jeongjae  and
      Bae, Seunghyun  and
      Jang, Ijun  and
      Jin, Hyunbin  and
      Jang, Jinkwan  and
      Kim, Taesup",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2112/",
    doi = "10.18653/v1/2026.acl-long.2112",
    pages = "45553--45573",
    ISBN = "979-8-89176-390-6"
}
```

## Contact

For questions about the dataset release, please open an issue in this repository.
