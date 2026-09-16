from pathlib import Path
from collections import defaultdict

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

RRF_CONSTANT = 60


# ============================================================
# 1. LOAD DOCUMENT
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


# ============================================================
# 3. BM25
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


# ============================================================
# 6. RETRIEVE
# ============================================================

bm25_results = bm25_retriever.invoke(
    query
)

vector_results = vector_retriever.invoke(
    query
)


# ============================================================
# 7. RRF FUNCTION
# ============================================================

def reciprocal_rank_fusion(
    result_lists,
    constant=60,
):

    scores = defaultdict(float)

    document_map = {}


    for result_list in result_lists:

        for rank, document in enumerate(
            result_list,
            start=1,
        ):

            chunk_id = document.metadata.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue


            # Store the document once.

            document_map[chunk_id] = document


            # Reciprocal Rank Fusion score.

            scores[chunk_id] += (
                1 / (constant + rank)
            )


    ranked_chunk_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


    ranked_results = []

    for chunk_id in ranked_chunk_ids:

        ranked_results.append(
            (
                document_map[chunk_id],
                scores[chunk_id],
            )
        )


    return ranked_results


# ============================================================
# 8. APPLY RRF
# ============================================================

fused_results = reciprocal_rank_fusion(
    [
        bm25_results,
        vector_results,
    ],
    constant=RRF_CONSTANT,
)


# ============================================================
# 9. DISPLAY ORIGINAL RANKINGS
# ============================================================

print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(query)


print("\n" + "=" * 70)
print("BM25 RANKING")
print("=" * 70)

for rank, document in enumerate(
    bm25_results,
    start=1,
):

    print(
        f"{rank}. Chunk "
        f"{document.metadata.get('chunk_id')}"
    )


print("\n" + "=" * 70)
print("VECTOR RANKING")
print("=" * 70)

for rank, document in enumerate(
    vector_results,
    start=1,
):

    print(
        f"{rank}. Chunk "
        f"{document.metadata.get('chunk_id')}"
    )


# ============================================================
# 10. DISPLAY FUSED RANKING
# ============================================================

print("\n" + "=" * 70)
print("RECIPROCAL RANK FUSION")
print("=" * 70)


for rank, (
    document,
    score,
) in enumerate(
    fused_results,
    start=1,
):

    chunk_id = document.metadata.get(
        "chunk_id"
    )

    print(
        f"\nRank {rank}"
    )

    print(
        f"Chunk ID: {chunk_id}"
    )

    print(
        f"RRF Score: {score:.6f}"
    )

    print("-" * 70)

    print(
        document.page_content[:400]
    )