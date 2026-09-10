# Module 1 — LangChain Fundamentals

## Objective

Understand what LangChain is, why it is useful, and the major components involved in building LLM-powered applications.

This module focuses mainly on concepts before building larger pipelines.

---

## What is LangChain?

LangChain is a framework for building applications that use Large Language Models.

Instead of directly calling an LLM for every task, LangChain provides abstractions for connecting:

```text
User Input
    ↓
Prompt
    ↓
LLM
    ↓
Output
```

and later more advanced systems such as:

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retriever
    ↓
LLM
    ↓
Answer
```

---

## Core LangChain Components

### Models

Models are responsible for interacting with language models.

Examples used in this project include:

```text
Gemma
Qwen
```

running locally through Ollama.

---

### Prompts

Prompts define how information is sent to the model.

Instead of manually constructing strings, LangChain provides prompt templates.

Conceptually:

```text
Instructions
     +
User Input
     ↓
Prompt
     ↓
LLM
```

---

### Output Parsers

LLMs return message objects or generated responses.

Output parsers transform model output into a format that is easier for the application to use.

Example:

```text
LLM Response
    ↓
StrOutputParser
    ↓
Python String
```

---

### Chains

Chains connect multiple components together.

Example:

```text
Prompt
  ↓
LLM
  ↓
Output Parser
```

LangChain Expression Language (LCEL) allows components to be connected using:

```python
|
```

Example:

```python
chain = prompt | llm | parser
```

---

### Documents

LangChain represents document content using `Document` objects.

A `Document` mainly contains:

```text
page_content
+
metadata
```

Documents become important when building RAG applications.

---

### Embeddings

Embeddings convert text into numerical vectors.

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

They allow semantic similarity between pieces of text to be calculated.

---

### Vector Stores

Vector stores store document embeddings and allow similarity searches.

Examples include:

```text
Chroma
FAISS
Qdrant
Pinecone
```

Chroma is used later in this project.

---

### Retrievers

Retrievers accept a query and return relevant documents.

```text
Question
   ↓
Retriever
   ↓
Relevant Documents
```

---

## LangChain Application Mental Model

A basic LLM application:

```text
User
 ↓
Prompt
 ↓
LLM
 ↓
Response
```

A RAG application:

```text
Documents
 ↓
Process + Store Knowledge
 ↓

User Question
 ↓
Retrieve Knowledge
 ↓
Prompt
 ↓
LLM
 ↓
Grounded Answer
```

---

## Key Takeaway

LangChain does not replace the LLM.

It provides abstractions for connecting LLMs with other components such as:

```text
Prompts
Documents
Embeddings
Vector Stores
Retrievers
Tools
Memory
Agents
```

This module establishes the foundation for the rest of the learning project.