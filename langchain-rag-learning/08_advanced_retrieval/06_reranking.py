from pathlib import Path
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
)

from langchain_core.output_parsers import (
    StrOutputParser,
)


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

RETRIEVAL_K = 5
FINAL_K = 3

RRF_CONSTANT = 60


# ============================================================
# 1. LOAD AND CHUNK
# ============================================================

markdown_text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

document = Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "domain": "healthcare",
    },
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunks = splitter.split_documents(
    [document]
)

for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = index


# ============================================================
# 2. BM25
# ============================================================

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = RETRIEVAL_K


# ============================================================
# 3. VECTOR SEARCH
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings,
)

vector_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_K
    }
)


# ============================================================
# 4. LLM
# ============================================================

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
)


# ============================================================
# 5. RRF
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

            document_map[chunk_id] = document

            scores[chunk_id] += (
                1 / (constant + rank)
            )


    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


    return [
        document_map[chunk_id]
        for chunk_id in ranked_ids
    ]


# ============================================================
# 6. RERANKING PROMPT
# ============================================================

rerank_prompt = ChatPromptTemplate.from_template(
    """
You are evaluating whether a document chunk
is relevant to answering a user's question.

Question:

{question}


Document Chunk:

{document}


Classify the relevance as exactly one of:

HIGH
MEDIUM
LOW

Definitions:

HIGH:
The chunk contains information directly useful
for answering the question.

MEDIUM:
The chunk contains related information but does
not directly answer the question.

LOW:
The chunk is mostly unrelated to the question.

Return only one word:

HIGH
MEDIUM
LOW
"""
)


rerank_chain = (
    rerank_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 7. RERANK FUNCTION
# ============================================================

def rerank_documents(
    question,
    documents,
):

    label_scores = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    reranked = []


    for document in documents:

        label = rerank_chain.invoke(
            {
                "question": question,
                "document": document.page_content,
            }
        )

        label = label.strip().upper()

        score = label_scores.get(
            label,
            0,
        )


        reranked.append(
            {
                "document": document,
                "label": label,
                "score": score,
            }
        )


    reranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )


    return reranked


# ============================================================
# 8. QUERY
# ============================================================

query = (
    "Random Forest accuracy?"
)


# ============================================================
# 9. INITIAL RETRIEVAL
# ============================================================

bm25_results = bm25_retriever.invoke(
    query
)

vector_results = vector_retriever.invoke(
    query
)


# ============================================================
# 10. FUSION
# ============================================================

candidate_documents = reciprocal_rank_fusion(
    [
        bm25_results,
        vector_results,
    ],
    constant=RRF_CONSTANT,
)


print(
    f"\nCandidates after fusion: "
    f"{len(candidate_documents)}"
)


# ============================================================
# 11. RERANK
# ============================================================

reranked_results = rerank_documents(
    query,
    candidate_documents,
)


# ============================================================
# 12. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RERANKED RESULTS")
print("=" * 70)


for rank, item in enumerate(
    reranked_results,
    start=1,
):

    document = item["document"]

    chunk_id = document.metadata.get(
        "chunk_id",
        "unknown",
    )

    print(
        f"\nRank {rank}"
    )

    print(
        f"Chunk ID: {chunk_id}"
    )

    print(
        f"Relevance: {item['label']}"
    )

    print("-" * 70)

    print(
        document.page_content[:500]
    )


# ============================================================
# 13. FINAL CONTEXT
# ============================================================

final_documents = [
    item["document"]
    for item in reranked_results[:FINAL_K]
]


print("\n" + "=" * 70)
print("FINAL DOCUMENTS")
print("=" * 70)


for rank, document in enumerate(
    final_documents,
    start=1,
):

    print(
        f"{rank}. Chunk "
        f"{document.metadata.get('chunk_id')}"
    )