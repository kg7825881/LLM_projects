from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

# Configuration 
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

VECTOR_DB_DIR = (
    PROJECT_DIR
    / "04_embeddings_vectorstore"
    / "chroma_db"
)

# Embedding Model 
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

# Vector Store 
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

# Retriever
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs={
        "k": 3
    }
)

# LLM 
llm = ChatOllama(
    model= "gemma3:4b",
    temperature=0
)

# Document Formatter
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
SOURCE: {source}
CHUNK ID: {chunk_id}

{document.page_content}
"""
        )
    return "\n\n".join(
        formatted_documents
    )

# Prompts
prompt = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Answer the question using only the provided
context.

If the answer cannot be found in the context,
say:

"I could not find this information in the provided document."

Do not use outside knowledge.
Do not invent information.


Context:

{context}


Question:

{question}


Answer:
"""
)

# Rag Chain
rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }

    | prompt 
    | llm 
    | StrOutputParser()
)

# Run
question = input(
    "\n Ask a question about the document: "
)

answer = rag_chain.invoke(
    question
)

print("\nANSWER:\n")
print(answer)