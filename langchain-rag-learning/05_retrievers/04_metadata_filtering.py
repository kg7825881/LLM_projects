from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Configuration
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "04_embeddings_vectorstore"
    / "chroma_db"
)

# 2. Embedding Model
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

# 3. Load Existing vector store
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

# 4. Convert vectore store to retriever
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {
            "domain": "healthcare"
        }
    }
)

# 5. Ask question 
query = (
    "Which model performed best? "
)

# 6. Retrieve Relevant documents
documents = retriever.invoke(
    query
)

# 6. Inspect result


for index, document in enumerate(documents):

    print("\n" + "=" * 70)

    print(
        f"RESULT {index + 1}"
    )

    print("-" * 70)

    print(
        document.page_content
    )

    print("\nMETADATA:")
    
    print(
        document.metadata
    )