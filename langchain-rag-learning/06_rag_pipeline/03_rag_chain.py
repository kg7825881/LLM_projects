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

# Create a formatting function
def format_document(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )

prompt = ChatPromptTemplate.from_messages(
    """
You are a research assistant.

Answer the question using only the provided context.

If the context does not contain enough information to answer the question,
say: "I could not find this information in the provided document."

Do not use outside knowledge.
Do not invent information.

Context:
{context}

Question:
{query}

Answer:
"""
)

rag_chain = (
    {
        "context": retriever | format_document,
        "query": RunnablePassthrough()
    }
    | prompt

    | llm

    | StrOutputParser()
)

answer = rag_chain.invoke(
    "Who is the President of the United States?"
)

print(answer)