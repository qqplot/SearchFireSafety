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


### 2. Legal Documents (/data/doc.jsonl)
| Key                                              | Example                              | Explanation                          |
| ------------------------------------------------ | ------------------------------------ | ------------------------------------ |
| `doc_id`                                         | `0`                                  | Sequential integer.                  |
| `semantic_id`                                    | `NFPC‑101_1조`                        | Law‑code + article slug.             |
| `collection_name`                                | `소방시설 설치 및 관리에 관한 법률`                | Parent statute.                      |
| `law_level`                                      | `행정규칙`                               | Law level (법률 / 시행령 / 행정규칙 etc)                     |
| `law_name`                                       | Full statute title.                  |                                      |
| `chapter`, `chapter_description`, `chapter_body` | Text content.                        |                                      |
| `deleted`                                        | `false`                              | `true` if the provision is repealed. |
| `related_chapters`                               | Cross‑links to related statutes. |                                      |
| `matched_doc_id_merged`                          | [1291, 1292]               |   IDs of related document ids                                   |

## Evaluation Prompts

## Tutorial
---

### 1. Crawl Korean legislation (`Crawl_Law.ipynb`)

Open the notebook and replace `url_list` with **법령 페이지** from [https://www.law.go.kr/LSW/main.html](https://www.law.go.kr/LSW/main.html).
Run all cells—each URL is downloaded, parsed chapter‑by‑chapter, and the result is printed:

```python
url_list = [
    "https://www.law.go.kr/법령/소방기본법",
    "https://www.law.go.kr/법령/주택법",
]
```

Example output

```
[OK]  https://www.law.go.kr/...소방기본법  →  132 item(s)
```

The notebook writes a **newline‑delimited JSON (`*.jsonl`)** file containing:

```json
{"doc_id":0, "semantic_id":"소방기본법 제1장 1조", "chapter_body":"..."}
```

---

### 2. Evaluate retrievers (`retrieval_eval.py`)

Benchmark TF‑IDF, BM25, BGE‑m3 (or any supported by SentenceTransformer in HuggingFace), DPR (or any subset) on a *docs* / *queries* pair.

```bash
python retrieval_eval.py \
    --docs    data/law_docs.jsonl \
    --queries data/train_queries.jsonl \
    --methods tfidf,bm25,bge,dpr \
    --topk 100 \
    --expand_links            # RAG‑style link expansion
    --device cuda:0           # cpu / cuda:<id> / mps
```

| Key flags                                                   | Purpose                                          |
| ----------------------------------------------------------- | ------------------------------------------------ |
| `--tfidf_max_features`                                      | vocabulary size (default 120 000)                |
| `--bm25_k1`, `--bm25_b`                                     | BM25 hyper‑params                                |
| `--bge_model_name`                                          | any Sentence‑Transformer (default `BAAI/bge-m3`) |
| `--dpr_context_encoder_path`, `--dpr_question_encoder_path` | local DPR checkpoints                            |
| `--batch_size`                                              | GPU memory vs. speed trade‑off                   |

The script prints per‑method **Recall@{1,2,3,5,10,20,100} & MRR**, then saves `ir_metrics.csv`.

---

### 3. Generate answers (`inference.py`)

Run one or many LLMs with optional retrieval augmentation.

```bash
python inference.py \
    --docs    data/doc.jsonl \
    --queries data/dev_queries.jsonl \
    --retriever bge          \        # tfidf / bm25 / bge
    --topk 5                 \        # 0 = no RAG
    --models  LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
              openai:gpt-4o \
    --out_dir rag_outputs \
    --max_new_tokens 8192 \
    --expand_with_links       \
    --openai_key_file ~/.openai_key
```

Open‑source models are loaded via `AutoModelForCausalLM`; OpenAI models are indicated by the prefix `openai:`.
`--oracle` makes the generator see **gold** document ids instead of retrieved ones (upper‑bound).

Each model creates one JSONL file:

```
rag_outputs/
  exaone-3.5-7.8b-instruct_bge_k5.jsonl
  gpt-4o_bge_k5.jsonl
```

Every line copies the original query row and adds:

```json
"model_answer": "..."
```

---

### 4. Clean generations (`data_postprocessing.py`)

Useful for LLMs that wrap answers with proprietary tags (e.g. `<think> … </think>`).

```bash
python data_postprocessing.py rag_outputs/llama-3-70b_bge_k5.jsonl \
    --inplace           # overwrite file
    --strip-think       # drop <think> blocks
    --extra-regex "<noise>.*?</noise>"
```

| Option                                   | Explanation                                    |
| ---------------------------------------- | -------------------------------------------------  |
| `--strip-prompts` / `--no-strip-prompts` | remove \`assistant  :\` etc. |
| `--new-field cleaned_answer`             | keep raw & add cleaned field instead of overwrite         |
| `--backup`                               | save `.bak` before modifying                   |

---

### 5. Score answers (`eval.py`)

Compute **ROUGE, BERTScore, LLM Judge, and Win‑Rate** (model vs. gold) in one go.

```bash
python eval.py \
    --input  rag_outputs/exaone-3.5-7.8b-instruct_bge_k5.jsonl \
    --output results/exaone-3.5-7.8b-instruct_bge_k5_scores.jsonl \
    --metrics bert,rouge,llm,winrate \
    --oracle_docs data/doc.jsonl \
    --openai_api_key $OPENAI_API_KEY
```

| Metric flag | Description                                                |
| ----------- | ----------------------------------------------------------  |
| `bert`      | `beomi/kcbert-base` on faithfulness & similarity |
| `rouge`     | Rouge‑1/2/L/Lsum (stemming)                        |
| `llm`       | GPT‑4o discrete pass/fail (0 or 1) |
| `winrate`   | GPT‑4o pairwise A/B comparison                   |

The script appends metric columns to each row, writes a new JSONL, and prints corpus‑level means:

```
Dataset-mean ▶ BERTScore=0.8423  ·  ROUGE-1 F1=0.6710  ·  LLM‑Score=0.79  ·  Win‑Rate=0.62
```

---
