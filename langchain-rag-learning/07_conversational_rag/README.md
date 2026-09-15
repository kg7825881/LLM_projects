# Module 7 — Conversational RAG & Memory

## Objective

Extend the basic RAG pipeline from Module 6 so that the system can understand follow-up questions using conversation history.

A normal RAG pipeline treats every question independently.

Conversational RAG adds:

- Chat history
- Follow-up question understanding
- Standalone question rewriting
- History-aware retrieval
- Session-based conversation memory
- Grounded answer generation
- Source tracking

---

# Why Conversational RAG?

Consider this conversation:

```text
User:
Which machine learning model performed best?

AI:
Random Forest performed best.

User:
What was its accuracy?
```

The second question:

```text
What was its accuracy?
```

is not a good standalone retrieval query.

The word:

```text
"its"
```

depends on the previous conversation.

The intended meaning is closer to:

```text
What was the accuracy of Random Forest?
```

A normal retriever does not automatically know this relationship.

---

# Normal RAG vs Conversational RAG

## Normal RAG

```text
Question
   ↓
Retriever
   ↓
Relevant Documents
   ↓
Context
   ↓
LLM
   ↓
Answer
```

Every question is processed independently.

---

## Conversational RAG

```text
Chat History
     +
Current Question
     ↓
Question Rewriter
     ↓
Standalone Question
     ↓
Retriever
     ↓
Relevant Documents
     ↓
Context
     +
Chat History
     +
Current Question
     ↓
LLM
     ↓
Grounded Answer
```

Conversation history now helps the system understand follow-up questions.

---

# Module Architecture

```text
User Question
      │
      ▼
Conversation History
      │
      ▼
Question Rewriting LLM
      │
      ▼
Standalone Question
      │
      ▼
Embedding Model
      │
      ▼
Retriever
      │
      ▼
Relevant Chunks
      │
      ▼
Context Formatting
      │
      ▼
Context + Chat History + Question
      │
      ▼
Answer Prompt
      │
      ▼
LLM
      │
      ▼
Grounded Answer
      │
      ▼
Update Conversation History
```

---

# Module Files

```text
07_conversational_rag/
│
├── 01_chat_history_basics.py
├── 02_history_aware_prompt.py
├── 03_question_rewriting.py
├── 04_history_aware_retriever.py
├── 05_conversational_rag.py
├── 06_session_history.py
├── 07_complete_conversational_rag.py
└── README.md
```

---

# 1. Chat History Basics

File:

```text
01_chat_history_basics.py
```

LangChain represents conversation using message objects.

The main message types used are:

```python
HumanMessage
AIMessage
```

Example:

```python
chat_history = [
    HumanMessage(
        content="Which model performed best?"
    ),

    AIMessage(
        content="Random Forest performed best."
    )
]
```

Conceptually:

```text
Chat History

HumanMessage
    ↓
Which model performed best?

AIMessage
    ↓
Random Forest performed best.
```

Using message objects preserves the role of each message.

---

# 2. MessagesPlaceholder

`MessagesPlaceholder` allows previous conversation messages to be inserted into a prompt.

Example:

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a research assistant."
        ),

        MessagesPlaceholder(
            variable_name="chat_history"
        ),

        (
            "human",
            "{question}"
        )
    ]
)
```

At runtime this may become:

```text
SYSTEM:
You are a research assistant.

HUMAN:
Which model performed best?

AI:
Random Forest performed best.

HUMAN:
What was its accuracy?
```

The model can now use previous conversation context to interpret the current question.

---

# 3. History-Aware Prompt

File:

```text
02_history_aware_prompt.py
```

The first step is to allow the LLM to understand conversation history.

```text
Chat History
     +
Current Question
     ↓
Prompt
     ↓
LLM
```

However, this alone does not completely solve conversational retrieval.

The retriever still needs a good search query.

---

# 4. The Retrieval Problem

Suppose the current question is:

```text
What was its accuracy?
```

If this exact question is sent to Chroma:

```text
"What was its accuracy?"
        ↓
Embedding Model
        ↓
Retriever
```

important information is missing.

The query does not explicitly contain:

```text
Random Forest
```

Therefore, conversation history should be used before retrieval.

---

# 5. Question Rewriting

File:

```text
03_question_rewriting.py
```

Question rewriting converts a context-dependent question into a standalone question.

Example:

```text
Chat History:

User:
Which machine learning model performed best?

AI:
Random Forest performed best.


Current Question:

What was its accuracy?
```

The question rewriting LLM produces something similar to:

```text
What accuracy did Random Forest achieve?
```

The exact wording may differ.

The important requirement is that the meaning is preserved.

---

# Why Rewrite the Question?

Compare these two retrieval queries:

```text
What was its accuracy?
```

and:

```text
What accuracy did Random Forest achieve?
```

The second query contains stronger semantic information:

```text
Random Forest
accuracy
```

Therefore:

```text
Conversation History
       ↓
Better Search Query
       ↓
Better Retrieval
       ↓
Better Context
       ↓
Better Answer
```

---

# Question Rewriting Prompt

The rewriting prompt has one responsibility:

```text
Conversation History
        +
Latest Question
        ↓
Standalone Question
```

It should not answer the user's question.

Example instructions:

```text
Rewrite the latest user question into a
standalone question using conversation history.

Do not answer the question.

If the question is already standalone,
return it unchanged.
```

---

# 6. History-Aware Retriever

File:

```text
04_history_aware_retriever.py
```

The rewritten question is passed to the retriever.

```text
"What was its accuracy?"
          ↓
Question Rewriter
          ↓
"What accuracy did Random Forest achieve?"
          ↓
Retriever
          ↓
Relevant Documents
```

This is different from simply passing conversation history to the final answer LLM.

The conversation is now improving the retrieval process itself.

---

# 7. Two LLM Responsibilities

Conversational RAG uses the LLM for two separate tasks.

## Job 1 — Question Rewriting

```text
History + Follow-up Question
           ↓
          LLM
           ↓
Standalone Question
```

Example:

```text
What was its accuracy?

↓

What accuracy did Random Forest achieve?
```

The model should not answer the question during this step.

---

## Job 2 — Answer Generation

```text
Question
    +
Retrieved Context
    +
Conversation History
    ↓
   LLM
    ↓
Answer
```

The answer should be grounded in the retrieved document context.

Although the same Gemma model can perform both jobs, the prompts and responsibilities are different.

---

# 8. Conversational RAG

File:

```text
05_conversational_rag.py
```

The components are combined:

```text
Chat History
      +
Question
      ↓
Rewrite Question
      ↓
Standalone Question
      ↓
Retriever
      ↓
Documents
      ↓
Format Documents
      ↓
Context
      ↓
Answer Prompt
      ↓
LLM
      ↓
Answer
```

---

# 9. Document Context Formatting

Retrieved LangChain `Document` objects are converted into text before being supplied to the LLM.

Example:

```python
def format_documents(documents):

    formatted_documents = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "unknown"
        )

        formatted_documents.append(
            f"""
Source: {source}
Chunk ID: {chunk_id}

{document.page_content}
"""
        )

    return "\n\n".join(
        formatted_documents
    )
```

The LLM receives context similar to:

```text
Source: Cardiac Arrest.pdf
Chunk ID: 66

[Retrieved document content]


Source: Cardiac Arrest.pdf
Chunk ID: 33

[Retrieved document content]
```

This provides traceability between the generated answer and retrieved chunks.

---

# 10. Grounded Answer Generation

The answer prompt instructs the model to use only retrieved document context.

Important rules include:

```text
Use only provided document context

Do not invent information

Do not rely on outside knowledge

Use chat history only to understand the conversation

Do not treat previous AI answers as document evidence
```

If sufficient evidence is unavailable:

```text
I could not find this information in the provided document.
```

---

# Why Previous AI Answers Are Not Evidence

Suppose the model previously generated an incorrect answer.

If that AI response is later treated as authoritative knowledge:

```text
Incorrect AI Answer
       ↓
Stored in History
       ↓
Used as Evidence
       ↓
Another Incorrect Answer
```

the error can propagate through the conversation.

Therefore:

```text
Chat History
     ↓
Conversation Understanding
```

while:

```text
Retrieved Documents
     ↓
Factual Evidence
```

These responsibilities should remain separate.

---

# 11. Session History

File:

```text
06_session_history.py
```

Conversation history needs to be associated with a session.

A simple in-memory dictionary is used:

```python
sessions = {}
```

Example structure:

```text
sessions
│
├── user_001
│   ├── HumanMessage
│   ├── AIMessage
│   ├── HumanMessage
│   └── AIMessage
│
└── user_002
    ├── HumanMessage
    └── AIMessage
```

Each session maintains its own conversation.

---

# Session ID vs Memory

A session ID is only an identifier.

Example:

```text
user_001
```

It tells the application which conversation history belongs to the current user/session.

The actual conversation memory consists of:

```text
HumanMessage
AIMessage
HumanMessage
AIMessage
...
```

---

# Updating Conversation History

After generating an answer:

```python
chat_history.append(
    HumanMessage(
        content=question
    )
)

chat_history.append(
    AIMessage(
        content=answer
    )
)
```

The next question can then use the previous interaction.

---

# 12. Complete Conversational RAG

File:

```text
07_complete_conversational_rag.py
```

The complete runtime pipeline is:

```text
User Question
      ↓
Get Session History
      ↓
Rewrite Question
      ↓
Standalone Question
      ↓
Retrieve Top-K Documents
      ↓
Format Documents
      ↓
Generate Grounded Answer
      ↓
Store User Question
      ↓
Store AI Answer
      ↓
Wait for Next Question
```

---

# Complete Runtime Example

```text
User:

Which machine learning model performed best?

        ↓

Standalone Question:

Which machine learning model performed best?

        ↓

Retriever

        ↓

Relevant Cardiac Arrest paper chunks

        ↓

AI:

Random Forest performed best.

        ↓

Store conversation history
```

Then:

```text
User:

What was its accuracy?

        ↓

Conversation History
+
Current Question

        ↓

Question Rewriter

        ↓

Standalone Question:

What was the accuracy of Random Forest?

        ↓

Retriever

        ↓

Relevant accuracy chunks

        ↓

Answer LLM
```

---

# 13. Testing Conversational RAG

Do not test conversational RAG using only independent questions.

Test references across multiple turns.

Example:

```text
Which machine learning model performed best?

What was its accuracy?

What other models were compared with it?

Which one performed worst?
```

This tests whether conversation history is correctly influencing question rewriting and retrieval.

---

# Debugging With Standalone Questions

The complete application prints:

```text
Standalone Question:
```

before generating an answer.

This is useful for debugging.

For example:

```text
User:
What was its accuracy?

Standalone Question:
What was the accuracy of Random Forest?
```

If the standalone question is wrong, the problem occurs before retrieval.

The debugging flow becomes:

```text
Wrong Answer
    ↓
Check Standalone Question
    ↓
Check Retrieved Chunks
    ↓
Check Context
    ↓
Check Final LLM Answer
```

This makes the pipeline much easier to diagnose.

---

# 14. Retrieved Sources

The application also displays retrieved sources:

```text
Retrieved Sources:

1. Cardiac Arrest.pdf (Chunk 66)
2. Cardiac Arrest.pdf (Chunk 33)
3. Cardiac Arrest.pdf (Chunk 76)
```

This helps inspect whether retrieval found the correct evidence.

Therefore, when an answer is incorrect:

```text
Check retrieval first
        ↓
Were the correct chunks retrieved?
```

If not, changing the final LLM prompt alone may not solve the problem.

---

# 15. Chat History vs RAG Knowledge

These should not be confused.

## Chat History

Contains the conversation:

```text
User questions
AI answers
```

Purpose:

```text
Understand conversational context
```

---

## RAG Knowledge

Contains document information stored in the vector database.

In this project:

```text
Cardiac Arrest.pdf
        ↓
Chunks
        ↓
Embeddings
        ↓
Chroma
```

Purpose:

```text
Provide factual document evidence
```

Therefore:

```text
Chat History ≠ Vector Database Knowledge
```

---

# 16. Short-Term Memory

The in-memory session history acts as basic short-term conversational memory.

```text
Current Session
      ↓
Recent Conversation
      ↓
Context for Future Questions
```

However, this implementation is not persistent.

If the Python application stops:

```text
sessions = {}
```

is lost.

Therefore:

```text
Current Implementation
=
In-Memory Session History
```

not permanent memory.

---

# 17. Long Conversations

Keeping unlimited conversation history is not scalable.

For example:

```text
5 messages
50 messages
500 messages
5000 messages
```

Passing everything into the prompt can cause:

```text
Larger prompts
Higher token usage
Slower inference
Context-window pressure
Irrelevant information
Potentially worse answers
```

Future memory strategies can include:

```text
Recent-message windows
Conversation summarization
Relevant-memory retrieval
Episodic memory
Long-term profile memory
Persistent storage
```

These are beyond the basic Module 7 implementation.

---

# 18. Models and Components

The final conversational RAG application uses:

```text
nomic-embed-text
        ↓
Query Embeddings


Chroma
        ↓
Stored Document Knowledge


Retriever
        ↓
Relevant Document Chunks


Gemma 3 4B
        ↓
Question Rewriting
+
Answer Generation


ChatPromptTemplate
        ↓
Structured Prompts


MessagesPlaceholder
        ↓
Conversation History


HumanMessage / AIMessage
        ↓
Structured Chat History
```

---

# 19. Reusing the Existing Vector Store

Module 7 does not recreate document embeddings.

The vector database created in Module 4 is reused:

```text
04_embeddings_vectorstore/
└── chroma_db/
```

The conversational application loads it using:

```python
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)
```

Therefore:

```text
Document Ingestion
      ↓
Done Once


User Questions
      ↓
Embed Query
      ↓
Search Existing Database
```

---

# 20. Complete Learning Architecture

After Module 7, the project has progressed through:

```text
Module 1
LangChain Fundamentals
        ↓
Module 2
LLM Chains
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
Basic RAG
        ↓
Module 7
Conversational RAG
```

---

# Final Architecture

```text
                    OFFLINE INGESTION

Cardiac Arrest.pdf
        │
        ▼
   PyMuPDF4LLM
        │
        ▼
      Markdown
        │
        ▼
LangChain Document
        │
        ▼
      Chunking
        │
        ▼
      Chunks
        │
        ▼
 nomic-embed-text
        │
        ▼
      Chroma


                    ONLINE CONVERSATIONAL RAG

                     User Question
                           │
                           ▼
                     Chat History
                           │
                           ▼
                   Question Rewriter
                           │
                           ▼
                  Standalone Question
                           │
                           ▼
                    nomic-embed-text
                           │
                           ▼
                       Retriever
                           │
                           ▼
                   Relevant Chunks
                           │
                           ▼
                    Format Context
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Document Evidence           Chat History
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                      Answer Prompt
                           │
                           ▼
                         Gemma
                           │
                           ▼
                   Grounded Answer
                           │
                           ▼
                  Update Chat History
```

---

# Key Takeaways

### 1. Conversational RAG is more than chat history

It is not simply:

```text
RAG + Previous Messages
```

Conversation history should also help improve retrieval.

---

### 2. Rewrite context-dependent questions

```text
"What was its accuracy?"
        ↓
"What was the accuracy of Random Forest?"
```

This creates a better retrieval query.

---

### 3. Separate rewriting from answering

```text
Question Rewriter
      ↓
Standalone Question
```

and:

```text
Answer Generator
      ↓
Grounded Answer
```

are different responsibilities.

---

### 4. Chat history is context, not factual evidence

```text
Chat History
    ↓
Understand conversation

Retrieved Documents
    ↓
Provide factual evidence
```

---

### 5. Session history provides basic memory

```text
session_id
    ↓
Conversation History
```

The current implementation stores this only in Python memory.

---

### 6. Retrieval should be observable

Inspect:

```text
Standalone Question
Retrieved Chunks
Source
Chunk ID
Final Answer
```

This makes RAG failures much easier to debug.

---

## Module 7 Mental Model

Remember:

```text
History + Question
        ↓
Rewrite
        ↓
Standalone Question
        ↓
Retrieve
        ↓
Evidence
        ↓
History + Evidence + Question
        ↓
Generate
        ↓
Answer
        ↓
Remember Conversation
```

That is the core idea behind the conversational RAG pipeline built in this module.