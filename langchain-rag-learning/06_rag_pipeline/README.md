# Module 6 — Complete RAG Pipeline

## Objective

Combine document retrieval and LLM generation into a complete Retrieval-Augmented Generation (RAG) pipeline.

This module connects everything learned in the previous modules.

---

## What is RAG?

RAG stands for:

```text
Retrieval-Augmented Generation
```

It combines:

```text
Retrieval
+
Augmentation
+
Generation
```

### Retrieval

Find relevant document chunks.

```text
Question
 ↓
Retriever
 ↓
Relevant Documents
```

### Augmentation

Add the retrieved information to the LLM prompt.

```text
Question
+
Retrieved Context
```

### Generation

Generate an answer using the supplied context.

```text
Question + Context
       ↓
      LLM
       ↓
     Answer
```

---

## Complete Architecture

The project now contains two major pipelines.

### Offline Ingestion Pipeline

```text
Cardiac Arrest.pdf
        ↓
PyMuPDF4LLM
        ↓
Markdown
        ↓
LangChain Document
        ↓
Chunking
        ↓
Chunks
        ↓
nomic-embed-text
        ↓
Embeddings
        ↓
Chroma Vector Store
```

This pipeline prepares and stores the knowledge.

---

### Online RAG Pipeline

```text
User Question
      ↓
Embedding Model
      ↓
Retriever
      ↓
Relevant Chunks
      ↓
Context Formatting
      ↓
Question + Context
      ↓
Prompt
      ↓
Gemma
      ↓
Grounded Answer
```

The existing Chroma database is reused.

Document embeddings are not recreated for every question.

---

## 1. Manual RAG

File:

```text
01_manual_rag.py
```

The first RAG pipeline is implemented manually to understand every step.

```text
Question
 ↓
Retriever
 ↓
Documents
 ↓
Join Document Content
 ↓
Prompt
 ↓
LLM
 ↓
Answer
```

This demonstrates exactly where retrieval, augmentation, and generation happen.

---

## 2. RAG Prompt

File:

```text
02_rag_prompt.py
```

`ChatPromptTemplate` is used to structure the RAG prompt.

The prompt receives two important variables:

```text
{context}
{question}
```

They come from different sources:

```text
User
 ↓
question

Retriever
 ↓
context
```

The prompt instructs the model to answer using the provided document context.

---

## 3. LCEL RAG Chain

File:

```text
03_rag_chain.py
```

LangChain Expression Language is used to combine retrieval and generation.

Example structure:

```python
rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## RunnablePassthrough in RAG

The same question follows two paths.

```text
                    ┌→ Retriever
                    │      ↓
Question ───────────┤  Documents
                    │      ↓
                    │   Context
                    │
                    └→ RunnablePassthrough
                           ↓
                     Original Question
```

The result becomes:

```python
{
    "context": "retrieved document text",
    "question": "user's original question"
}
```

This dictionary is passed to the prompt.

---

## 4. RAG With Sources

File:

```text
04_rag_with_sources.py
```

Retrieved metadata is included with the context.

Example:

```text
Source: Cardiac Arrest.pdf
Chunk ID: 42

[Retrieved text]
```

This improves traceability.

Metadata created during document ingestion now becomes useful for identifying where retrieved information came from.

---

## 5. Grounding and Guardrails

File:

```text
05_rag_guardrails.py
```

The LLM is instructed to:

```text
Use only retrieved context
Do not invent information
Do not rely on outside knowledge
Return a fallback when information is unavailable
```

Example fallback:

```text
I could not find this information in the provided document.
```

---

## Why Guardrails Matter

Vector retrieval always attempts to find the closest chunks.

Therefore an unrelated question may still retrieve documents.

```text
Unrelated Question
       ↓
Retriever
       ↓
Nearest Available Chunks
```

The retrieved chunks are not necessarily relevant.

The generation layer should therefore be instructed not to invent an answer from insufficient context.

More advanced retrieval evaluation and threshold calibration can be introduced later.

---

## Retrieval vs Generation

These responsibilities must remain separate.

### Retriever

```text
Question
 ↓
Relevant Evidence
```

### LLM

```text
Question
+
Evidence
 ↓
Answer
```

The retriever finds information.

The LLM generates an answer from that information.

---

## RAG Does Not Train the LLM

Adding documents to the vector database does not change the LLM's model weights.

```text
RAG ≠ Training
RAG ≠ Fine-Tuning
```

Instead, retrieved information is supplied at query time.

```text
Retrieved Context
       +
Question
       ↓
Prompt
       ↓
LLM
```

---

## Components and Responsibilities

```text
PyMuPDF4LLM
      ↓
Document extraction


RecursiveCharacterTextSplitter
      ↓
Chunking


nomic-embed-text
      ↓
Text → Vector


Chroma
      ↓
Vector storage + search


Retriever
      ↓
Relevant Documents


ChatPromptTemplate
      ↓
Context + Question


Gemma
      ↓
Answer Generation


StrOutputParser
      ↓
Final String
```

---

## Testing the RAG Pipeline

The pipeline should be tested with different categories of questions.

### Direct Question

```text
Which machine learning model performed best?
```

Tests whether relevant evidence can be retrieved and used.

### Paraphrased Question

```text
Which classifier showed the strongest performance in the experiments?
```

Tests semantic retrieval.

### Out-of-Scope Question

```text
Who invented the telephone?
```

Tests grounding and fallback behavior.

---

## Module Files

```text
06_rag_pipeline/
│
├── 01_manual_rag.py
├── 02_rag_prompt.py
├── 03_rag_chain.py
├── 04_rag_with_sources.py
├── 05_rag_guardrails.py
├── 06_complete_rag.py
└── README.md
```

---

## Complete Learning Pipeline

After Module 6:

```text
MODULE 1
LangChain Fundamentals
        ↓
MODULE 2
LLM Chains
        ↓
MODULE 3
Document Loading + Chunking
        ↓
MODULE 4
Embeddings + Vector Stores
        ↓
MODULE 5
Retrievers
        ↓
MODULE 6
Retrieval-Augmented Generation
```

---

## Key Takeaway

A basic RAG system consists of two separate stages:

```text
INGESTION

Documents
 ↓
Chunks
 ↓
Embeddings
 ↓
Vector Store
```

and:

```text
QUERY

Question
 ↓
Retriever
 ↓
Relevant Context
 ↓
Prompt
 ↓
LLM
 ↓
Grounded Answer
```

Module 6 connects all previously learned LangChain components into the first complete document-based RAG application.