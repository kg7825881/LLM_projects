from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = MODULE_DIR.parent

MARKDOWN_PATH = (
    PROJECT_ROOT
    / "03_document_loading_chunking"
    / "outputs"
    / "Cardiac_Arrest.md"
)

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "04_embeddings_vectorstore"
    / "chroma_db"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 5


# ============================================================
# 1. LOAD MARKDOWN
# ============================================================

markdown_text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

document = Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "domain": "healthcare",
        "document_type": "research_paper",
        "topic": "cardiac_arrest_prediction",
    },
)


# ============================================================
# 2. CHUNK DOCUMENT
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

chunks = splitter.split_documents(
    [document]
)

for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = index

print(
    f"\nCreated {len(chunks)} chunks"
)


# ============================================================
# 3. BM25 RETRIEVER
# ============================================================

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = TOP_K


# ============================================================
# 4. VECTOR RETRIEVER
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
)

vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": TOP_K
    },
)


# ============================================================
# 5. QUERY
# ============================================================

query = (
    "Which classifier showed the strongest "
    "predictive performance?"
)

print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(query)


# ============================================================
# 6. RUN BOTH RETRIEVERS
# ============================================================

bm25_results = bm25_retriever.invoke(
    query
)

vector_results = vector_retriever.invoke(
    query
)


# ============================================================
# 7. DISPLAY INDIVIDUAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BM25 RESULTS")
print("=" * 70)

for rank, document in enumerate(
    bm25_results,
    start=1,
):

    chunk_id = document.metadata.get(
        "chunk_id",
        "unknown",
    )

    print(
        f"{rank}. Chunk {chunk_id}"
    )


print("\n" + "=" * 70)
print("VECTOR RESULTS")
print("=" * 70)

for rank, document in enumerate(
    vector_results,
    start=1,
):

    chunk_id = document.metadata.get(
        "chunk_id",
        "unknown",
    )

    print(
        f"{rank}. Chunk {chunk_id}"
    )


# ============================================================
# 8. SIMPLE HYBRID MERGE
# ============================================================

combined_documents = {}

for document in bm25_results + vector_results:

    chunk_id = document.metadata.get(
        "chunk_id"
    )

    if chunk_id is not None:

        if chunk_id not in combined_documents:

            combined_documents[chunk_id] = document


hybrid_results = list(
    combined_documents.values()
)


# ============================================================
# 9. DISPLAY HYBRID RESULTS
# ============================================================

print("\n" + "=" * 70)
print("HYBRID RESULTS")
print("=" * 70)

print(
    f"\nUnique candidate chunks: "
    f"{len(hybrid_results)}"
)


for index, document in enumerate(
    hybrid_results,
    start=1,
):

    chunk_id = document.metadata.get(
        "chunk_id",
        "unknown",
    )

    print("\n" + "-" * 70)

    print(
        f"Candidate {index}"
    )

    print(
        f"Chunk ID: {chunk_id}"
    )

    print(
        document.page_content[:500]
    )