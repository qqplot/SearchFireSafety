# SearchFireSafety

## Introduction

Building fire safety is a critical public concern, especially in dense urban areas. In South Korea, however, the legal framework governing fire safety is exceptionally complex, marked by fragmented legislation, extensive cross-references, and often vague decrees. This intricate web of regulations creates significant challenges for interpretation, particularly for non-experts, and can lead to real-world safety failures. This underscores an urgent need for computational systems, particularly retrieval-augmented generation (RAG) models, to provide accurate and trustworthy access to legal information.
To address this, we introduce **SearchFireSafety**, the first question-answering (QA) dataset designed for RAG in the Korean fire safety legal domain. It comprises (1) 941 real-world QA pairs from public inquiries (2013-2015), (2) 4,437 supporting legal documents from 117 cited statutes, and (3) a comprehensive legal citation graph. 

We benchmarked six retrieval methods, finding that dense retrievers, especially multilingual embeddings, consistently outperformed sparse and language-specific baselines. Our evaluation of five Korean-capable large language models (LLMs) across various RAG strategies, using metrics like ROUGE, BERTScore, and GPT-4o for human-aligned scoring, demonstrates that grounding LLMs with documents retrieved from SearchFireSafety significantly improves the accuracy and human alignment of generated responses. 

## Data Format


### 1. Question-Answering Dataset


### 2. Legal Documents


## Evaluation Prompts
