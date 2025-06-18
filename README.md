# SearchFireSafety

## Introduction

Building fire safety is a critical public concern, especially in dense urban areas. In South Korea, however, the legal framework governing fire safety is exceptionally complex, marked by fragmented legislation, extensive cross-references, and often vague decrees. This intricate web of regulations creates significant challenges for interpretation, particularly for non-experts, and can lead to real-world safety failures. This underscores an urgent need for computational systems, particularly retrieval-augmented generation (RAG) models, to provide accurate and trustworthy access to legal information.
To address this, we introduce **SearchFireSafety**, the first question-answering (QA) dataset designed for RAG in the Korean fire safety legal domain. It comprises (1) 941 real-world QA pairs from public inquiries (2013-2015), (2) 4,437 supporting legal documents from 117 cited statutes, and (3) a comprehensive legal citation graph. 

We benchmarked six retrieval methods, finding that dense retrievers, especially multilingual embeddings, consistently outperformed sparse and language-specific baselines. Our evaluation of five Korean-capable large language models (LLMs) across various RAG strategies, using metrics like ROUGE, BERTScore, and GPT-4o for human-aligned scoring, demonstrates that grounding LLMs with documents retrieved from SearchFireSafety significantly improves the accuracy and human alignment of generated responses. 

## Data Format


### 1. Question-Answering Dataset (/data/qna.jsonl)

| Key                |  Example                             | Notes                             |
| ------------------ |  ----------------------------------- | --------------------------------- |
| `question_id`      |  `0`                                 | Unique identifier.                |
| `qna_doc_id`       |  `1AA‑2304‑0225487-…`                | Government archive ID.            |
| `original_text`    |  Full civil‑petition reply text.     |                                   |
| `law_references`   |  `["소방시설 … 시행령 별표4", …]`      | Manually extracted citations.     |
| `question_raw`     | Original citizen question.          |                                   |
| `question`         |  GPT-4o Cleansed question text.           |                                   |
| `answer`           | Gold answer by NFA official.    |                                   |
| `matched_doc_id`   |  `[2, 1259, 1292]`                   | IDs of supporting statute chunks. |
| `semantic_ids`     |  Machine‑readable statute IDs.       |                                   |
| `has_matched_docs` |  Whether supporting docs were found. |                                   |


### 2. Legal Documents
| Key                                              | Example                              | Explanation                          |
| ------------------------------------------------ | ------------------------------------ | ------------------------------------ |
| `doc_id`                                         | `0`                                  | Sequential integer.                  |
| `semantic_id`                                    | `NFPC‑101_1조`                        | Law‑code + article slug.             |
| `collection_name`                                | `소방시설 설치 및 관리에 관한 법률`                | Parent statute.                      |
| `law_level`                                      | `행정규칙`                               | 법률 / 시행령 / 행정규칙 등.                     |
| `law_name`                                       | Full statute title.                  |                                      |
| `chapter`, `chapter_description`, `chapter_body` | Text content.                        |                                      |
| `deleted`                                        | `false`                              | `true` if the provision is repealed. |
| `related_chapters`                               | Cross‑links to related statutes. |                                      |
| `matched_doc_id_merged`                          | [1291, 1292]               |   IDs of related document ids                                   |

## Evaluation Prompts
