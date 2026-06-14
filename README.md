# SearchFireSafety (ACL 2026)

Official dataset repository for the ACL 2026 paper:
**Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA**

Preprint: https://arxiv.org/abs/2604.06173

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

## Dataset Statistics

Current file-level counts:

- `legal_docs.jsonl`: 4,468
- `realworld_qa.jsonl`: 876
- `multihop_qa_mcq.jsonl`: 3,395

Additional summary statistics:

- Legal docs avg text length: 477.9 characters
- Real-world QA avg question length: 90.7 characters
- Real-world QA avg answer length: 278.1 characters
- Multi-hop MCQ avg question length: 51.1 characters

Note: Table 1 in the paper reports 4,467 legal documents. The current release file contains 4,468 rows.

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
| `related_doc_ids` | int[] (optional) | Citation/delegation-linked `doc_id` list |

Notes:

- Most rows include `related_doc_ids`; 54 rows do not.
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

## Citation

If you use this dataset, please cite the ACL 2026 paper.
For now, you may cite the arXiv preprint:

```bibtex
@article{chae2026beyond,
  title={Beyond Case Law: Evaluating Structure-Aware Retrieval and Safety in Statute-Centric Legal QA},
  author={Chae, Kyubyung and Yeom, Jewon and Park, Jeongjae and Bae, Seunghyun and Jang, Ijun and Jin, Hyunbin and Jang, Jinkwan and Kim, Taesup},
  journal={arXiv preprint arXiv:2604.06173},
  year={2026}
}
```

(We will update this section with the ACL Anthology entry once the proceedings version is available.)

## Contact

For questions about the dataset release, please open an issue in this repository.
