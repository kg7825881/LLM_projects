# Module 8 — Advanced Retrieval: BM25, Hybrid Search & Reranking

## Objective

Module 8 improves the retrieval stage of the RAG pipeline.

Until now, the project primarily used semantic vector retrieval:

```text
Question
   ↓
Embedding Model
   ↓
Query Vector
   ↓
Chroma
   ↓
Top-K Similar Chunks
```

Semantic retrieval is powerful, but it is not always sufficient.

Some queries depend heavily on:

- Exact terminology
- Model names
- Technical keywords
- Abbreviations
- Identifiers
- Specific metrics
- Rare domain terms

Module 8 introduces multiple retrieval signals and combines them into a stronger retrieval pipeline.

The final architecture becomes:

```text
                     User Query
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       BM25 Retriever          Vector Retriever
             │                       │
       Lexical Search          Semantic Search
             │                       │
             └───────────┬───────────┘
                         ▼
               Reciprocal Rank Fusion
                         │
                         ▼
                  Candidate Chunks
                         │
                         ▼
                     Reranker
                         │
                         ▼
                    Top Context
                         │
                         ▼
                        LLM
                         │
                         ▼
                      Answer
```

---

# Module Files

```text
08_advanced_retrieval/
│
├── 01_bm25_basics.py
├── 02_bm25_retriever.py
├── 03_vector_vs_bm25.py
├── 04_hybrid_retrieval.py
├── 05_reciprocal_rank_fusion.py
├── 06_reranking.py
├── 07_complete_advanced_retrieval.py
└── README.md
```

Each file introduces one additional retrieval concept.

---

# 1. Why Advanced Retrieval?

The vector retriever from previous modules searches based primarily on semantic similarity.

For example:

```text
Document:

Random Forest achieved the best classification performance.
```

A query such as:

```text
Which classifier showed the strongest predictive performance?
```

may still retrieve the correct chunk even though the exact words are different.

This works because:

```text
best classification performance

≈

strongest predictive performance
```

The embedding model represents these sentences as semantically related vectors.

However, semantic search can sometimes be weaker when exact terminology matters.

Examples:

```text
Random Forest

PostgreSQL

Qdrant

Apache Spark

CA26000248

2A:14-1
```

For these kinds of queries, lexical retrieval can provide another useful signal.

---

# 2. Lexical vs Semantic Retrieval

Module 8 uses two major retrieval approaches.

## Lexical Retrieval

Implemented using:

```text
BM25
```

It focuses primarily on matching important words and terms.

Conceptually:

```text
Query
  ↓
Tokenization
  ↓
Term Matching
  ↓
Term Statistics
  ↓
BM25 Ranking
```

A useful mental model is:

```text
BM25 asks:

"Which documents contain important words
from my query?"
```

---

## Semantic Retrieval

Implemented using:

```text
nomic-embed-text
+
Chroma
```

Conceptually:

```text
Query
  ↓
Embedding Model
  ↓
Query Vector
  ↓
Vector Similarity
  ↓
Chroma Ranking
```

A useful mental model is:

```text
Vector search asks:

"Which documents mean something similar
to my query?"
```

---

# 3. BM25

BM25 stands for:

```text
Best Matching 25
```

It is a classical information retrieval ranking algorithm.

BM25 uses ideas including:

```text
Term Frequency
+
Inverse Document Frequency
+
Document Length Normalization
```

Its goal is to rank documents according to how relevant their terms are to the query.

---

# 4. Term Frequency

Term Frequency considers how often a query term appears in a document.

Example:

```text
Document A:

Random Forest achieved high accuracy.


Document B:

Random Forest Random Forest Random Forest
achieved high accuracy.
```

The repeated occurrence provides additional evidence that the document may be related to the query.

However, BM25 does not treat unlimited repetition as unlimited relevance.

Conceptually:

```text
1 occurrence
     ↓
Useful

2 occurrences
     ↓
More evidence

20 occurrences
     ↓
Not twenty times more useful
```

BM25 applies saturation to term frequency.

---

# 5. Inverse Document Frequency

Some words appear almost everywhere.

For example:

```text
the
and
model
data
```

These words are usually not very useful for distinguishing one document from another.

Other terms may be much rarer:

```text
Random Forest

cardiac arrest

Support Vector Machine
```

Rare terms can carry more retrieval information.

Conceptually:

```text
Very common term
       ↓
Lower importance


Rare term
       ↓
Higher importance
```

This is the basic intuition behind Inverse Document Frequency.

---

# 6. Document Length Normalization

Suppose:

```text
Chunk A
=
100 words


Chunk B
=
2000 words
```

The larger chunk naturally has more opportunities to contain query terms.

BM25 compensates for document length so that large documents do not automatically dominate simply because they contain more words.

---

# 7. BM25 Basics

File:

```text
01_bm25_basics.py
```

This file introduces BM25 using small manually created documents.

Example:

```python
documents = [

    Document(
        page_content=(
            "Random Forest achieved the best "
            "classification performance."
        )
    ),

    Document(
        page_content=(
            "Logistic Regression was used as "
            "a baseline classification model."
        )
    ),
]
```

A BM25 retriever is created using:

```python
BM25Retriever.from_documents(
    documents
)
```

No embedding model is required.

Therefore:

```text
BM25

does NOT require

Ollama embeddings
Chroma
Vector database
LLM
```

This is an important distinction.

---

# 8. BM25 on the Research Paper

File:

```text
02_bm25_retriever.py
```

The Cardiac Arrest Markdown document is loaded and chunked using the same general chunking configuration used earlier:

```text
chunk_size = 1000

chunk_overlap = 200
```

The chunks are then indexed by BM25.

```text
Cardiac_Arrest.md
        ↓
LangChain Document
        ↓
RecursiveCharacterTextSplitter
        ↓
Chunks
        ↓
BM25Retriever
```

Example query:

```text
Random Forest accuracy
```

The BM25 retriever returns chunks based on lexical relevance.

---

# 9. Why BM25 Is Rebuilt in Memory

The existing Chroma database stores vector information.

```text
Chroma
  ↓
Embedding vectors
+
Text
+
Metadata
```

It is not automatically a BM25 lexical index.

Therefore, in this learning implementation:

```text
Markdown
   ↓
Chunk again
   ↓
Build BM25 retriever
```

while the vector retriever loads the already existing:

```text
04_embeddings_vectorstore/
└── chroma_db/
```

In a larger production system, both indexes could be created and persisted as part of an ingestion architecture.

---

# 10. Vector Search vs BM25

File:

```text
03_vector_vs_bm25.py
```

This experiment sends the same query to both retrievers.

```text
                       Query
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
           BM25                  Vector Search
             │                       │
      Keyword Matching          Semantic Meaning
             │                       │
             ▼                       ▼
        Top-K Chunks             Top-K Chunks
```

The retrieved chunk IDs are compared.

Example:

```text
BM25:

[66, 67, 31]


Vector:

[66, 33, 76]


Common:

[66]
```

Different results are not necessarily a problem.

They show that the retrievers are capturing different relevance signals.

---

# 11. Useful Retrieval Experiments

Several query styles should be tested.

## Exact terminology

```text
Random Forest
```

## Exact terminology + metric

```text
Random Forest accuracy
```

## Semantic paraphrase

```text
Which classifier showed the strongest
predictive performance?
```

## Broader conceptual query

```text
What methods were used to predict
cardiac arrest?
```

The goal is not simply to decide:

```text
Which retriever is better?
```

A better question is:

```text
For which types of queries does each
retriever succeed or fail?
```

---

# 12. Hybrid Retrieval

File:

```text
04_hybrid_retrieval.py
```

Instead of choosing:

```text
BM25
```

or:

```text
Vector Search
```

hybrid retrieval uses both.

```text
                 Query
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
        BM25              Vector
          │                 │
          ▼                 ▼
    Lexical Results    Semantic Results
          │                 │
          └────────┬────────┘
                   ▼
             Hybrid Results
```

This provides:

```text
Exact-term matching
+
Semantic matching
```

---

# 13. Simple Hybrid Merge

The first hybrid implementation combines results from both retrievers and removes duplicate chunks.

Suppose:

```text
BM25:

66
31
67
45
20
```

and:

```text
Vector:

66
33
76
31
42
```

The unique candidate set becomes:

```text
66
31
67
45
20
33
76
42
```

Chunks `66` and `31` are not duplicated.

---

# 14. Problem With Simple Hybrid Merging

Simple merging gives us more candidate documents.

But it does not provide a strong final ranking.

For example:

```text
BM25:

1. Chunk 66
2. Chunk 31
3. Chunk 45


Vector:

1. Chunk 33
2. Chunk 66
3. Chunk 31
```

Chunk `66` ranks highly in both retrievers.

That agreement should contribute to its final ranking.

Simple deduplication does not fully capture this information.

This motivates:

```text
Reciprocal Rank Fusion
```

---

# 15. Reciprocal Rank Fusion

File:

```text
05_reciprocal_rank_fusion.py
```

Reciprocal Rank Fusion, or RRF, combines multiple ranked result lists.

A common form is:

```text
RRF(document)
=
Σ 1 / (C + rank)
```

where:

```text
rank
=
position of the document in a result list


C
=
rank constant
```

In this module:

```text
C = 60
```

---

# 16. RRF Example

Suppose:

```text
BM25

1. Chunk 66
2. Chunk 31
3. Chunk 45
4. Chunk 33
```

and:

```text
Vector

1. Chunk 33
2. Chunk 66
3. Chunk 76
4. Chunk 31
```

Chunk `66` appears:

```text
BM25 rank = 1

Vector rank = 2
```

Its RRF contribution is:

```text
1 / (60 + 1)
+
1 / (60 + 2)
```

Chunk `45` only appears in BM25:

```text
1 / (60 + 3)
```

Therefore, documents appearing highly across multiple retrievers can accumulate stronger fused scores.

---

# 17. Why Not Add Raw Scores?

BM25 and vector search use different scoring systems.

Therefore, doing this blindly:

```text
BM25 score
+
Vector similarity score
```

can be misleading.

The score scales may not be directly comparable.

RRF instead uses:

```text
Rank position
```

rather than directly combining unrelated raw score scales.

Conceptually:

```text
BM25 Rank
     +
Vector Rank
     ↓
RRF Ranking
```

---

# 18. Retrieval Fusion

After RRF, the architecture becomes:

```text
                  Query
                    │
           ┌────────┴────────┐
           ▼                 ▼
         BM25              Vector
           │                 │
           ▼                 ▼
      Ranked List       Ranked List
           │                 │
           └────────┬────────┘
                    ▼
                   RRF
                    │
                    ▼
             Fused Ranking
```

At this point we have a stronger candidate set.

But we can add another stage.

---

# 19. Reranking

File:

```text
06_reranking.py
```

Retrieval and reranking perform different jobs.

## Retrieval

Retrieval quickly finds candidate documents.

Conceptually:

```text
Large document collection
        ↓
Retriever
        ↓
Small candidate set
```

For example:

```text
50,000 chunks
     ↓
Retriever
     ↓
10 chunks
```

---

## Reranking

Reranking examines the small candidate set more carefully.

```text
10 candidates
      ↓
Reranker
      ↓
Top 3 candidates
```

Therefore:

```text
Retriever
=
Candidate Discovery


Reranker
=
Candidate Reordering
```

---

# 20. Why Not Rerank Everything?

A more sophisticated relevance evaluation is usually more expensive than initial retrieval.

We do not want:

```text
100,000 chunks
      ↓
LLM evaluates every chunk
```

Instead:

```text
100,000 chunks
      ↓
Fast Retrieval
      ↓
10–20 candidates
      ↓
Reranking
      ↓
3–5 chunks
```

This creates a multi-stage retrieval architecture.

---

# 21. LLM-Based Reranking

For learning purposes, this module uses the local Gemma model as a simple reranker.

The model evaluates:

```text
Question
+
Candidate Chunk
```

and returns one of:

```text
HIGH

MEDIUM

LOW
```

The meanings are:

```text
HIGH
=
Directly useful for answering the question


MEDIUM
=
Related but only partially useful


LOW
=
Mostly unrelated
```

---

# 22. Why Labels Instead of Fake Probabilities?

The LLM is not asked to generate:

```text
87.4% relevant
```

and that value is not treated as a calibrated probability.

Instead, the experiment uses coarse relevance labels:

```text
HIGH
MEDIUM
LOW
```

Internally:

```text
HIGH   → 3

MEDIUM → 2

LOW    → 1
```

These values are used only for ordering the candidate documents in this learning implementation.

---

# 23. RRF as a Tie-Breaker

Suppose multiple chunks receive:

```text
HIGH
```

Then the RRF score can remain useful as a secondary ranking signal.

The final ordering can conceptually use:

```text
Primary:
Reranker relevance


Secondary:
RRF score
```

Therefore:

```text
HIGH + strong RRF
```

can rank above:

```text
HIGH + weaker RRF
```

when both receive the same relevance label.

---

# 24. Dedicated Rerankers

The LLM-based reranker in this module is primarily intended to teach the architecture.

A more advanced system could use dedicated reranking models such as cross-encoders.

Conceptually:

```text
Query
+
Document
     ↓
Cross-Encoder
     ↓
Relevance Signal
```

This differs from bi-encoder/vector retrieval.

---

# 25. Bi-Encoder Retrieval

The embedding approach used earlier works approximately like:

```text
Document
   ↓
Embedding Model
   ↓
Document Vector
```

and separately:

```text
Query
  ↓
Embedding Model
  ↓
Query Vector
```

Then:

```text
Query Vector
     ↓
Compare
     ↓
Document Vectors
```

Because documents can be embedded ahead of time, this is efficient for retrieval.

---

# 26. Cross-Encoder Reranking

A cross-encoder conceptually processes:

```text
Query + Document
```

together.

```text
Query
   +
Document
   ↓
Model
   ↓
Relevance
```

This can provide a more detailed relevance judgment but is more expensive to apply across very large collections.

Therefore:

```text
Bi-Encoder / BM25
        ↓
Candidate Retrieval
        ↓
Cross-Encoder
        ↓
Reranking
```

is a useful multi-stage pattern.

---

# 27. Complete Advanced Retrieval

File:

```text
07_complete_advanced_retrieval.py
```

This file combines all Module 8 concepts.

The complete retrieval flow is:

```text
Question
   │
   ├───────────────┐
   │               │
   ▼               ▼
 BM25            Vector
   │               │
   ▼               ▼
Top-K            Top-K
   │               │
   └───────┬───────┘
           ▼
          RRF
           │
           ▼
   Candidate Chunks
           │
           ▼
       Reranker
           │
           ▼
      Final Top-K
           │
           ▼
       RAG Context
           │
           ▼
          LLM
           │
           ▼
         Answer
```

---

# 28. Complete Pipeline Responsibilities

Each component has a different responsibility.

## BM25

```text
Find exact and lexically relevant text
```

## Embedding Model

```text
Convert meaning into vector representations
```

## Chroma

```text
Store and search document vectors
```

## Vector Retriever

```text
Find semantically similar chunks
```

## Reciprocal Rank Fusion

```text
Combine BM25 and vector rankings
```

## Reranker

```text
Re-evaluate candidate relevance
```

## Final LLM

```text
Generate an answer from selected evidence
```

---

# 29. Final Retrieval Context

Only the strongest reranked chunks are sent to the final answer model.

For example:

```text
BM25 retrieves 5

Vector retrieves 5

        ↓

Merge + RRF

        ↓

7 unique candidates

        ↓

Rerank

        ↓

Top 3

        ↓

LLM Context
```

This is better than automatically passing every candidate into the answer prompt.

---

# 30. Observability

The complete script prints each retrieval stage.

Example:

```text
BM25:

1. Chunk 66
2. Chunk 31
3. Chunk 45


Vector:

1. Chunk 33
2. Chunk 66
3. Chunk 76


RRF:

1. Chunk 66 | Score ...
2. Chunk 33 | Score ...
3. Chunk 31 | Score ...


Reranked:

1. Chunk 66 | HIGH
2. Chunk 33 | HIGH
3. Chunk 76 | MEDIUM


Final Sources:

1. Cardiac Arrest.pdf (Chunk 66)
2. Cardiac Arrest.pdf (Chunk 33)
3. Cardiac Arrest.pdf (Chunk 76)
```

This makes retrieval easier to debug.

---

# 31. Debugging Retrieval

If the final answer is wrong, do not immediately blame the LLM.

Inspect the pipeline in order.

```text
Wrong Answer
     ↓
Was the relevant chunk retrieved by BM25?
     ↓
Was it retrieved by Vector Search?
     ↓
What happened after RRF?
     ↓
What happened during reranking?
     ↓
Was it included in final context?
     ↓
How did the answer LLM use the context?
```

This allows retrieval failures and generation failures to be separated.

---

# 32. Retrieval Failure vs Generation Failure

These are different problems.

## Retrieval Failure

The correct evidence never reaches the LLM.

```text
Question
   ↓
Retriever
   ↓
Wrong Chunks
   ↓
LLM
```

Prompt engineering cannot reliably recover information that was never retrieved.

---

## Generation Failure

The correct evidence is retrieved:

```text
Question
   ↓
Retriever
   ↓
Correct Chunks
   ↓
LLM
   ↓
Incorrect Answer
```

Now the issue may involve:

```text
Prompt design
Context formatting
LLM behavior
Grounding
```

Being able to distinguish these two failure types is essential when debugging RAG.

---

# 33. Hybrid Retrieval Strength

Hybrid retrieval combines complementary signals.

```text
BM25
=
Lexical relevance


Vector Search
=
Semantic relevance
```

Together:

```text
Hybrid Retrieval
=
Lexical
+
Semantic
```

This is useful when a corpus contains both:

```text
Exact technical terminology
```

and:

```text
Different descriptions of similar concepts
```

---

# 34. Example: Technical Skill Retrieval

Suppose a document contains:

```text
Apache Spark
```

A query containing:

```text
Spark
```

can benefit from lexical matching.

But another document may contain:

```text
distributed large-scale data processing
```

without explicitly mentioning Spark.

Semantic retrieval may identify conceptual similarity.

Therefore:

```text
Exact Skill
     ↓
BM25


Related Experience
     ↓
Vector Search


Both
     ↓
Hybrid Retrieval
```

---

# 35. Module 7 vs Module 8

Module 7 focused on:

```text
Conversation
```

Module 8 focuses on:

```text
Retrieval Quality
```

Module 7:

```text
History
   +
Current Question
   ↓
Standalone Question
   ↓
Retriever
```

Module 8:

```text
Question
   ↓
BM25 + Vector
   ↓
RRF
   ↓
Reranking
```

They solve different problems.

---

# 36. Future Combined Architecture

Eventually the two approaches can be combined.

```text
                   Chat History
                        +
                 Current Question
                        │
                        ▼
                Question Rewriter
                        │
                        ▼
                Standalone Question
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
            BM25               Vector
              │                   │
              └─────────┬─────────┘
                        ▼
                       RRF
                        │
                        ▼
                    Reranker
                        │
                        ▼
                 Relevant Evidence
                        │
                        ▼
                    Answer LLM
                        │
                        ▼
                      Answer
                        │
                        ▼
                Update Chat History
```

This combines:

```text
Conversational Understanding
+
Lexical Retrieval
+
Semantic Retrieval
+
Rank Fusion
+
Reranking
+
Grounded Generation
```

---

# 37. Important Parameters

The module introduces several parameters.

## `CHUNK_SIZE`

```python
CHUNK_SIZE = 1000
```

Controls approximate chunk size.

---

## `CHUNK_OVERLAP`

```python
CHUNK_OVERLAP = 200
```

Preserves context across chunk boundaries.

---

## `RETRIEVAL_K`

```python
RETRIEVAL_K = 5
```

Controls how many candidates each initial retriever returns.

For example:

```text
BM25   → 5

Vector → 5
```

---

## `RRF_CONSTANT`

```python
RRF_CONSTANT = 60
```

Controls the reciprocal-rank contribution.

---

## `FINAL_K`

```python
FINAL_K = 3
```

Controls how many reranked chunks are finally supplied to the answer LLM.

Therefore:

```text
RETRIEVAL_K
=
Candidate generation


FINAL_K
=
Final context selection
```

These are not the same thing.

---

# 38. Module 8 Dependencies

The main new dependency is:

```text
rank_bm25
```

Install with:

```powershell
pip install rank_bm25
```

The project also continues using:

```text
langchain
langchain-core
langchain-community
langchain-ollama
langchain-text-splitters
langchain-chroma
chromadb
pymupdf4llm
```

---

# 39. Models Used

## Embedding Model

```text
nomic-embed-text
```

Purpose:

```text
Query
↓
Vector
```

Used for semantic retrieval.

---

## LLM

```text
gemma3:4b
```

Used in this module for:

```text
Reranking
+
Final answer generation
```

The embedding model and LLM perform different jobs.

---

# 40. Module 8 Learning Progression

The module deliberately progresses step-by-step.

```text
01
BM25 Basics
   ↓
02
BM25 on Real Document
   ↓
03
BM25 vs Vector
   ↓
04
Hybrid Retrieval
   ↓
05
Reciprocal Rank Fusion
   ↓
06
Reranking
   ↓
07
Complete Advanced Retrieval
```

Each file adds one new idea.

---

# 41. Complete Project Progress

After Module 8:

```text
Module 1
LangChain Fundamentals
        ↓
Module 2
First LLM Chain
        ↓
Module 3
Document Loading & Chunking
        ↓
Module 4
Embeddings & Vector Stores
        ↓
Module 5
Retrievers
        ↓
Module 6
Complete RAG Pipeline
        ↓
Module 7
Conversational RAG & Memory
        ↓
Module 8
Advanced Retrieval
```

---

# Final Module 8 Architecture

```text
                         QUERY
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       BM25 RETRIEVER              VECTOR RETRIEVER
             │                           │
       lexical matching              embeddings
             │                           │
             │                      nomic-embed-text
             │                           │
             ▼                           ▼
       ranked chunks                ranked chunks
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
               RECIPROCAL RANK FUSION
                           │
                           ▼
                  candidate documents
                           │
                           ▼
                       RERANKER
                           │
                           ▼
                     best chunks
                           │
                           ▼
                     RAG CONTEXT
                           │
                           ▼
                         GEMMA
                           │
                           ▼
                        ANSWER
```

---

# Key Takeaways

## 1. Vector retrieval is not the only retrieval strategy

Semantic search is powerful, but exact lexical information can also matter.

---

## 2. BM25 provides lexical retrieval

```text
BM25
=
Term-based relevance
```

It does not require embeddings or a vector database.

---

## 3. Vector search provides semantic retrieval

```text
Vector Search
=
Meaning-based relevance
```

It can retrieve useful chunks even when the exact query words are absent.

---

## 4. Hybrid retrieval combines complementary signals

```text
BM25
+
Vector
=
Hybrid Retrieval
```

This allows exact and semantic evidence to participate.

---

## 5. Fusion is different from retrieval

Retrievers produce ranked lists.

Fusion combines those lists.

```text
BM25 Ranking
+
Vector Ranking
      ↓
     RRF
```

---

## 6. RRF avoids blindly combining incompatible raw scores

Instead of directly adding BM25 and vector scores:

```text
RRF uses rank positions.
```

---

## 7. Reranking is different from initial retrieval

```text
Retrieval
=
Find candidates


Reranking
=
Improve candidate ordering
```

---

## 8. Retrieve broadly, then narrow

A useful architecture is:

```text
Large Corpus
    ↓
Fast Retrieval
    ↓
Candidate Set
    ↓
Reranking
    ↓
Small High-Quality Context
```

---

## 9. More retrieved context is not automatically better

Sending too many chunks can introduce:

```text
Noise
Redundancy
Longer prompts
Irrelevant information
```

The goal is not maximum context.

The goal is:

```text
Relevant Context
```

---

## 10. Retrieval quality strongly affects RAG quality

The complete relationship is:

```text
Extraction Quality
       ↓
Chunk Quality
       ↓
Embedding / Index Quality
       ↓
Retrieval Quality
       ↓
Fusion Quality
       ↓
Reranking Quality
       ↓
Context Quality
       ↓
Answer Quality
```

---

# Module 8 Mental Model

Remember:

```text
BM25
=
WORDS


Vector Search
=
MEANING


Hybrid Retrieval
=
WORDS + MEANING


RRF
=
COMBINE RANKINGS


Reranking
=
RECHECK RELEVANCE


Final RAG
=
BEST EVIDENCE → LLM
```

That is the core idea behind Module 8.