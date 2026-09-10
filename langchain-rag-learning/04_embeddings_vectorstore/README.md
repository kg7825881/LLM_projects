# Module 4 — Embeddings & Vector Stores

## Objective

Learn how document chunks are converted into numerical vectors and stored so that information can be searched semantically.

---

## Pipeline Progress

```text
PDF
 ↓
Markdown
 ↓
Document
 ↓
Chunks
 ↓
Embedding Model
 ↓
Vectors
 ↓
Chroma Vector Store
```

Module 4 adds the embedding and storage layers to the ingestion pipeline.

---

## What is an Embedding?

An embedding is a numerical representation of text.

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

Example concept:

```text
"Random Forest achieved high accuracy"

↓

[0.12, -0.43, 0.82, ...]
```

Semantically similar text should generally produce vectors that are closer together in the embedding space.

---

## Embedding Model vs LLM

An embedding model and an LLM perform different tasks.

### LLM

```text
Text
 ↓
LLM
 ↓
Generated Text
```

Example:

```text
Gemma
Qwen
```

### Embedding Model

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

This project uses:

```text
nomic-embed-text
```

through Ollama.

---

## OllamaEmbeddings

LangChain connects to the embedding model using:

```python
from langchain_ollama import OllamaEmbeddings
```

Example:

```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
```

---

## embed_query vs embed_documents

### embed_query()

Used to embed a query.

```python
embeddings.embed_query(
    "Which model performed best?"
)
```

### embed_documents()

Used to embed multiple document texts.

```python
embeddings.embed_documents(
    [
        "Chunk 1",
        "Chunk 2",
        "Chunk 3"
    ]
)
```

Both document and query vectors need to exist in the same embedding space for meaningful similarity comparison.

---

## What is a Vector Store?

A vector store manages:

```text
Embedding
+
Original Text
+
Metadata
```

Conceptually:

```text
Record
│
├── Vector
├── Chunk Text
└── Metadata
```

This project uses Chroma.

---

## Chroma

Chroma is used as a persistent local vector store.

```python
from langchain_chroma import Chroma
```

The vector database is stored locally under:

```text
04_embeddings_vectorstore/
└── chroma_db/
```

---

## Creating the Vector Store

Chunks are inserted using:

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(VECTOR_DB_DIR)
)
```

Conceptually:

```text
Chunk
 ↓
Embedding Model
 ↓
Vector
 ↓
Chroma

Stored:
Vector + Text + Metadata
```

---

## Similarity Search

A user query is also converted into an embedding.

```text
Question
 ↓
Embedding Model
 ↓
Query Vector
```

The query vector is compared with stored vectors:

```text
Query Vector
      ↓
Vector Similarity
      ↓
Closest Chunk Vectors
      ↓
Relevant Documents
```

Example:

```python
results = vector_store.similarity_search(
    query,
    k=3
)
```

`k=3` means return the three most relevant results.

---

## Similarity Scores

Similarity search can also expose scores.

Example:

```python
vector_store.similarity_search_with_score(
    query,
    k=5
)
```

A similarity or distance score should not automatically be interpreted as:

```text
Probability
or
Confidence Percentage
```

Different vector stores and distance metrics may expose scores differently.

---

## Unrelated Queries

Vector search can still return results for an unrelated query.

Example:

```text
Query:
"What is the capital of Japan?"

Vector Store:
Cardiac Arrest research paper
```

The search system still has nearest vectors.

Therefore:

```text
Nearest Result
≠
Relevant Answer
```

This becomes an important issue when building RAG guardrails.

---

## Persistent Embeddings

Document embeddings should normally be created during ingestion.

```text
INGESTION

Documents
 ↓
Chunks
 ↓
Embeddings
 ↓
Chroma
```

They should not be recreated for every question.

During querying:

```text
QUERY

Question
 ↓
Query Embedding
 ↓
Search Existing Chroma DB
```

Only the question needs to be embedded.

---

## Module Files

```text
04_embeddings_vectorstore/
│
├── 01_embedding_basics.py
├── 02_ollama_embeddings.py
├── 03_vector_store.py
├── 04_similarity_search.py
├── 05_similarity_scores.py
├── 06_complete_vector_pipeline.py
├── chroma_db/
└── README.md
```

---

## Key Takeaway

The ingestion side now looks like:

```text
Document
 ↓
Chunks
 ↓
Embedding Model
 ↓
Vectors
 ↓
Vector Store
```

The query side looks like:

```text
Question
 ↓
Embedding Model
 ↓
Query Vector
 ↓
Vector Search
 ↓
Relevant Chunks
```

Module 4 provides the semantic-search foundation used by retrievers in Module 5.