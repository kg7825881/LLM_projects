# We are loading and chunking the text again because the Chroma database is vector index and we need BM25 lexical index

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Configuration
#==============================
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

MARKDOWN_PATH = (
    PROJECT_DIR
    / "03_document_loading_chunking"
    / "outputs"
    / "Cardiac_Arrest.md"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Load Document
#================================
markdown_text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

document = Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "domain": "healthcare",
        "document_type": "research_paper"
    }
)

# Chunk Document
#===================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP
)

chunks = splitter.split_documents(
    [document]
)

for index, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = index


print(
    f"Created {len(chunks)} chunks"
)

# BM25 Retriever
retriever = BM25Retriever.from_documents(
    chunks
)
retriever.k = 3

# Query
query = "Random Forest accuracy"

results = retriever.invoke(
    query
)

# DISPLAY RESULTS
# ============================================================

for index, document in enumerate(
    results,
    start=1
):

    print("\n" + "=" * 70)

    print(
        f"RESULT {index}"
    )

    print("-" * 70)

    print(
        document.page_content
    )

    print(
        "\nMETADATA:"
    )

    print(
        document.metadata
    )