# We are loading and chunking the text again because the Chroma database is vector index and we need BM25 lexical index

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

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

VECTOR_DB_DIR = (
    PROJECT_DIR
    /"04_embeddings_vectorstore"
    / "chroma_db"
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
        "extraction_method": "pymupdf4llm",
        "format": "markdown",
        "domain": "healthcare",
        "document_type": "research_paper",
        "topic": "cardiac_arrest_prediction",
    }
)

# Chunk Document
#===================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP,
     length_function=len,
)

chunks = splitter.split_documents(
    [document]
)

for index, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = index


print("\n" + "=" * 70)
print("DOCUMENT PREPARATION")
print("=" * 70)

print(
    f"Created {len(chunks)} chunks for BM25"
)

# Create BM25 Retriever
# ============================================================
bm25_retriever = BM25Retriever.from_documents(
    chunks
)
bm25_retriever.k = 3

# Load Embedding Model
# ============================================================
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Load the existing Chroma Vector store
# ============================================================
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
)

# Create Vector Retriever
# ============================================================
vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)

# Query
# ============================================================
query =  (
    "Random Forest accuracy"
)


print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(query)

# BM25 RETRIEVAL
# ============================================================

bm25_results = bm25_retriever.invoke(
    query
)

# VECTOR RETRIEVAL
# ============================================================

vector_results = vector_retriever.invoke(
    query
)

# DISPLAY BM25 RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BM25 RESULTS — LEXICAL SEARCH")
print("=" * 70)


for index, result in enumerate(
    bm25_results,
    start=1,
):

    chunk_id = result.metadata.get(
        "chunk_id",
        "unknown",
    )

    print(
        f"\nBM25 RESULT {index}"
    )

    print(
        f"Chunk ID: {chunk_id}"
    )

    print("-" * 70)

    print(
        result.page_content[:1000]
    )



#  DISPLAY VECTOR RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VECTOR RESULTS — SEMANTIC SEARCH")
print("=" * 70)


for index, result in enumerate(
    vector_results,
    start=1,
):

    chunk_id = result.metadata.get(
        "chunk_id",
        "unknown",
    )

    print(
        f"\nVECTOR RESULT {index}"
    )

    print(
        f"Chunk ID: {chunk_id}"
    )

    print("-" * 70)

    print(
        result.page_content[:1000]
    )


# COMPARE CHUNK IDS
# ============================================================

bm25_chunk_ids = [
    document.metadata.get(
        "chunk_id",
        "unknown",
    )
    for document in bm25_results
]


vector_chunk_ids = [
    document.metadata.get(
        "chunk_id",
        "unknown",
    )
    for document in vector_results
]


print("\n" + "=" * 70)
print("RETRIEVAL COMPARISON")
print("=" * 70)


print(
    "\nBM25 Chunk IDs:"
)

print(
    bm25_chunk_ids
)


print(
    "\nVector Chunk IDs:"
)

print(
    vector_chunk_ids
)



# FIND OVERLAPPING RESULTS
# ============================================================

bm25_valid_ids = {
    chunk_id
    for chunk_id in bm25_chunk_ids
    if chunk_id != "unknown"
}

vector_valid_ids = {
    chunk_id
    for chunk_id in vector_chunk_ids
    if chunk_id != "unknown"
}

common_chunk_ids = (
    bm25_valid_ids
    & vector_valid_ids
)


print(
    "\nCommon Chunk IDs:"
)

if common_chunk_ids:

    print(
        sorted(common_chunk_ids)
    )

else:

    print(
        "No common chunks were retrieved."
    )


#  SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)


print(
    f"""
Query:
{query}

BM25:
Lexical / keyword-based retrieval.

Vector Search:
Semantic / meaning-based retrieval.

BM25 returned:
{bm25_chunk_ids}

Vector search returned:
{vector_chunk_ids}

Common chunks:
{sorted(common_chunk_ids)}
"""
)