from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings,
)

# Configuration
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

VECTOR_DB_DIR = (
    PROJECT_DIR
    / "04_embeddings_vectorstore"
    / "chroma_db"
)

# 1. Loading Embedding Model
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

# 2. Load Existing vectore store
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

# 3. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs = {
        "k": 3
    }
)

# 4. User question
query = (
    "What all machine learning models are used for testing?"
)

# 5. Retrieve Relevant documents
documents = retriever.invoke(
    query
)

# 6. Build Context
context = "\n\n".join(
    document.page_content
    for document in documents
)

# 7. Create Prompt
prompt = f"""
Answer the query using only the context below.

Context:
{context}

Query:
{query}

Answer:
"""

# 8. Load LLM

llm = ChatOllama(
    model = "gemma3:4b",
    temperature=0
)

# 9. Generate Answer
response = llm.invoke(
    prompt
)

print("\nAnswer:\n")

print(
    response.content
)

