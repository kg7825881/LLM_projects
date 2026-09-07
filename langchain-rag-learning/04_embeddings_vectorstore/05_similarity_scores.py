from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent

VECTOR_DB_DIR = (
    BASE_DIR
    / "chroma_db"
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma (
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

query = (
    "What is the capital of Japan?"
)

results = (
    vector_store.similarity_search_with_score(
        query,
        k=5
    )
)

for document, score in results:

    print("\n" + "=" * 70)

    print("Score: ",score)

    print("-" * 70)

    print(document.page_content[:500])

    print(document.metadata)