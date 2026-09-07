from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. Configuration
BASE_DIR = Path(__file__).parent

MODULE_DIR = BASE_DIR.parent

MARKDOWN_PATH = (
    MODULE_DIR
    / "03_document_loading_chunking"
    / "outputs"
    / "Cardiac_Arrest.md"
)

VECTOR_DB_DIR = (
    BASE_DIR
    / "chroma_db"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 2. LOAD MARKDOWN

markdown_text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

document = Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "extraction_method": "pymupdf4llm",
        "format": "markdown",
        "domain": "healthcare",
        "document_type": "research_paper",
        "topic": "cardiac_arrest_prediction"
    }
)

# 3. Chunk

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(
    [document]
)

for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = index

print(
    f"Created {len(chunks)} chunks"
)

# 4. Embedding Model
embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

# 5. Store Vectors
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(VECTOR_DB_DIR)
)

print(
    "Vector Store created successfully."
)

# 6. Search

query = (
    "Which machine learning model performed best?"
)

results = (
    vector_store.similarity_search_with_score(
        query,
        k=3
    )
)

# 7. Inspect
for index, (result, score) in enumerate(results):

    print("\n" + "=" * 70)

    print(
        f"RESULT {index + 1}"
    )

    print("-" * 70)

    print(
        f"Distance score: {score}"
    )

    print("-" * 70)

    print(
        result.page_content
    )

    print("\nMETADATA:")

    print(
        result.metadata
    )