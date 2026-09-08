# Module 5 — Retrievers

## Objective

Learn how LangChain retrieves relevant document chunks from a vector store before passing them to an LLM.

This module focuses only on **retrieval**.

No LLM is used for answer generation yet.

---

## RAG Pipeline Progress

So far, the project pipeline is:

```text
PDF
 ↓
PyMuPDF4LLM
 ↓
Markdown
 ↓
LangChain Document
 ↓
Chunking
 ↓
Embeddings
 ↓
Chroma Vector Store
 ↓
Retriever
 ↓
Relevant Documents
```

The next stage will use these retrieved documents as context for an LLM.

---

## What is a Retriever?

A retriever is a component that accepts a query and returns relevant LangChain `Document` objects.

```text
User Query
    ↓
Retriever
    ↓
Relevant Documents
```

A retriever does not generate an answer.

Its responsibility is to find the most useful context for the query.

---

## Vector Store vs Retriever

### Vector Store

Stores and searches:

```text
Embedding Vector
+
Chunk Text
+
Metadata
```

In this project:

```python
Chroma
```

is used as the vector store.

### Retriever

Provides a standard interface for retrieving relevant documents.

```python
retriever = vector_store.as_retriever()
```

Documents can then be retrieved using:

```python
documents = retriever.invoke(query)
```

---

## 1. Basic Retriever

File:

```text
01_basic_retriever.py
```

The existing Chroma vector store from Module 4 is loaded and converted into a retriever.

```python
retriever = vector_store.as_retriever()
```

The retriever accepts a natural-language query and returns relevant `Document` objects.

---

## 2. Top-K Retrieval

File:

```text
02_top_k_retrieval.py
```

The number of retrieved documents can be controlled using `k`.

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)
```

Here:

```text
k = 3
```

means return the three most relevant document chunks.

Choosing `k` involves a trade-off.

```text
Small k
 ↓
Less context
 ↓
Possible missing information


Large k
 ↓
More context
 ↓
Possible irrelevant/redundant information
```

More retrieved documents do not automatically produce better RAG results.

---

## 3. Similarity Retrieval

Similarity retrieval returns chunks that are semantically closest to the user query.

```text
Query
 ↓
Query Embedding
 ↓
Vector Similarity
 ↓
Top-K Chunks
```

Example:

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)
```

Similarity retrieval primarily focuses on relevance.

---

## 4. MMR Retrieval

File:

```text
03_mmr_retriever.py
```

MMR stands for:

```text
Maximal Marginal Relevance
```

MMR attempts to balance:

```text
Relevance
+
Diversity
```

This can reduce repetitive results caused by similar or overlapping chunks.

Example:

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)
```

### `k`

Final number of documents returned.

### `fetch_k`

Number of candidate documents initially retrieved before MMR selects the final results.

```text
Vector Store
     ↓
10 candidate chunks
     ↓
MMR
     ↓
3 relevant + diverse chunks
```

---

## 5. Metadata Filtering

File:

```text
04_metadata_filtering.py
```

Metadata can be used to restrict the documents searched by the retriever.

Example metadata:

```python
{
    "source": "Cardiac Arrest.pdf",
    "domain": "healthcare",
    "document_type": "research_paper",
    "topic": "cardiac_arrest_prediction",
    "chunk_id": 42
}
```

Example filter:

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {
            "domain": "healthcare"
        }
    }
)
```

Metadata filtering becomes especially important when the vector store contains multiple documents.

---

## 6. Similarity vs MMR

File:

```text
05_retrieval_comparison.py
```

Both retrieval strategies were compared using the same query.

### Similarity Search

Focuses mainly on:

```text
Relevance
```

It may return multiple highly similar chunks.

### MMR Search

Focuses on:

```text
Relevance
+
Diversity
```

It can reduce redundant retrieved context.

Neither method is always better.

The correct retrieval strategy depends on the data and the type of questions being asked.

---

## Important Concepts Learned

### Retrieval is not generation

A retriever returns documents.

```text
Question
 ↓
Retriever
 ↓
Relevant Documents
```

It does not generate the final answer.

An LLM will later use the retrieved documents as context.

---

### Embedding Model vs Vector Store vs Retriever

```text
Embedding Model
Text → Vector

Vector Store
Stores and searches vectors

Retriever
Query → Relevant Documents
```

These are separate components with different responsibilities.

---

### Retrieval Quality Matters

A powerful LLM cannot reliably answer a document-based question if the retriever provides the wrong context.

```text
Poor Retrieval
     ↓
Wrong / Irrelevant Context
     ↓
Poor RAG Answer
```

Therefore:

```text
Good RAG
starts with
Good Retrieval
```

---

## Module 5 Files

```text
05_retrievers/
│
├── 01_basic_retriever.py
│   └── Convert Chroma into a LangChain retriever
│
├── 02_top_k_retrieval.py
│   └── Experiment with number of retrieved chunks
│
├── 03_mmr_retriever.py
│   └── Retrieve relevant but diverse chunks
│
├── 04_metadata_filtering.py
│   └── Restrict retrieval using metadata
│
├── 05_retrieval_comparison.py
│   └── Compare similarity and MMR retrieval
│
└── README.md
```

---

## Key Takeaway

The complete mental model after this module is:

```text
                INGESTION

Document
   ↓
Chunks
   ↓
Embedding Model
   ↓
Vector Store


                RETRIEVAL

User Question
   ↓
Retriever
   ↓
Relevant Documents


                GENERATION
                  (Next)

Question + Relevant Documents
             ↓
            LLM
             ↓
           Answer
```

Module 5 establishes the retrieval layer required for building a complete RAG pipeline.