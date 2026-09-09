from pathlib import Path 

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuration
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

VECTOR_DB_DIR = (
    PROJECT_DIR
    / "04_embeddings_vectorstore"
    / "chroma_db"
)

# Load Embedding Model
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

# Load existing vector store
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

# Retrieve
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)

# LOAD LLM Model
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

question = (
    "Which machine learning model "
    "performed best?"
)


documents = retriever.invoke(
    question
)


context_parts = []


for document in documents:

    chunk_id = document.metadata.get(
        "chunk_id",
        "unknown"
    )

    source = document.metadata.get(
        "source",
        "unknown"
    )

    context_parts.append(
        f"""
SOURCE: {source}
CHUNK ID: {chunk_id}

{document.page_content}
"""
    )


context = "\n\n".join(
    context_parts
)