from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).parent

PROJECT_ROOT = BASE_DIR.parent

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "04_embeddings_vectorstore"
    / "chroma_db"
)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

# Similarity Retriever

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)

# MMR Retriever

mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# Query

query = (
    "Which machine learning models "
    "were used and how did they perform?"
)


# Similarity Results

similarity_results = (
    similarity_retriever.invoke(query)
)


print("\n")
print("=" * 70)
print("SIMILARITY SEARCH")
print("=" * 70)


for index, document in enumerate(
    similarity_results
):

    print(
        f"\nRESULT {index + 1}"
    )

    print(
        document.page_content[:500]
    )

    print(
        "\nChunk ID:",
        document.metadata.get("chunk_id")
    )


# MMR Results

mmr_results = (
    mmr_retriever.invoke(query)
)


print("\n")
print("=" * 70)
print("MMR SEARCH")
print("=" * 70)


for index, document in enumerate(
    mmr_results
):

    print(
        f"\nRESULT {index + 1}"
    )

    print(
        document.page_content[:500]
    )

    print(
        "\nChunk ID:",
        document.metadata.get("chunk_id")
    )