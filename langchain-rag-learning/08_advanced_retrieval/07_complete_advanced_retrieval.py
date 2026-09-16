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


print(
    f"Loaded {len(chunks)} chunks"
)


# ============================================================
# 3. BM25 RETRIEVER
# ============================================================

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = RETRIEVAL_K


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
        "k": RETRIEVAL_K
    },
)


# ============================================================
# 5. LLM
# ============================================================

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
)


# ============================================================
# 6. RECIPROCAL RANK FUSION
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
        {
            "document": document_map[chunk_id],
            "rrf_score": scores[chunk_id],
        }

        for chunk_id in ranked_ids
    ]


# ============================================================
# 7. RERANK PROMPT
# ============================================================

rerank_prompt = ChatPromptTemplate.from_template(
    """
Evaluate how relevant the document chunk is
for answering the question.

Question:

{question}


Document:

{document}


Return exactly one label:

HIGH
MEDIUM
LOW

HIGH:
Directly useful for answering the question.

MEDIUM:
Related but only partially useful.

LOW:
Mostly unrelated.

Return only the label.
"""
)


rerank_chain = (
    rerank_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 8. RERANK FUNCTION
# ============================================================

def rerank_documents(
    question,
    fused_results,
):

    label_scores = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }


    reranked = []


    for item in fused_results:

        document = item["document"]


        label = rerank_chain.invoke(
            {
                "question": question,
                "document": document.page_content,
            }
        )


        label = label.strip().upper()


        reranked.append(
            {
                "document": document,

                "label": label,

                "label_score":
                    label_scores.get(
                        label,
                        0,
                    ),

                "rrf_score":
                    item["rrf_score"],
            }
        )


    # First:
    # LLM relevance label
    #
    # Second:
    # RRF score as tie-breaker.

    reranked.sort(
        key=lambda item: (
            item["label_score"],
            item["rrf_score"],
        ),
        reverse=True,
    )


    return reranked


# ============================================================
# 9. FORMAT FINAL CONTEXT
# ============================================================

def format_documents(
    documents
):

    formatted = []


    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown",
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "unknown",
        )


        formatted.append(
            f"""
Source: {source}
Chunk ID: {chunk_id}

{document.page_content}
"""
        )


    return "\n\n".join(
        formatted
    )


# ============================================================
# 10. ANSWER PROMPT
# ============================================================

answer_prompt = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Answer the question using only the provided
document context.

Do not use outside knowledge.
Do not invent information.

If the answer cannot be found in the context,
say exactly:

"I could not find this information in the provided document."


Context:

{context}


Question:

{question}


Answer:
"""
)


answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 11. ADVANCED RETRIEVAL FUNCTION
# ============================================================

def advanced_retrieve(
    question
):

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_results = bm25_retriever.invoke(
        question
    )


    # --------------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------------

    vector_results = vector_retriever.invoke(
        question
    )


    # --------------------------------------------------------
    # RRF
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        [
            bm25_results,
            vector_results,
        ],
        constant=RRF_CONSTANT,
    )


    # --------------------------------------------------------
    # Rerank
    # --------------------------------------------------------

    reranked_results = rerank_documents(
        question,
        fused_results,
    )


    return (
        bm25_results,
        vector_results,
        fused_results,
        reranked_results,
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "ADVANCED HYBRID RAG"
    )

    print(
        "=" * 70
    )

    print(
        "\nBM25 + Vector Search + RRF + Reranking"
    )

    print(
        "\nType 'exit' to stop."
    )


    while True:

        question = input(
            "\nYou: "
        ).strip()


        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "\nApplication stopped."
            )

            break


        if not question:

            continue


        # ----------------------------------------------------
        # Advanced Retrieval
        # ----------------------------------------------------

        (
            bm25_results,
            vector_results,
            fused_results,
            reranked_results,

        ) = advanced_retrieve(
            question
        )


        # ----------------------------------------------------
        # Display BM25
        # ----------------------------------------------------

        print(
            "\nBM25:"
        )

        for rank, document in enumerate(
            bm25_results,
            start=1,
        ):

            print(
                f"{rank}. Chunk "
                f"{document.metadata.get('chunk_id')}"
            )


        # ----------------------------------------------------
        # Display Vector
        # ----------------------------------------------------

        print(
            "\nVector:"
        )

        for rank, document in enumerate(
            vector_results,
            start=1,
        ):

            print(
                f"{rank}. Chunk "
                f"{document.metadata.get('chunk_id')}"
            )


        # ----------------------------------------------------
        # Display RRF
        # ----------------------------------------------------

        print(
            "\nRRF:"
        )

        for rank, item in enumerate(
            fused_results,
            start=1,
        ):

            document = item["document"]

            print(
                f"{rank}. Chunk "
                f"{document.metadata.get('chunk_id')} "
                f"| Score: "
                f"{item['rrf_score']:.6f}"
            )


        # ----------------------------------------------------
        # Display Reranking
        # ----------------------------------------------------

        print(
            "\nReranked:"
        )

        for rank, item in enumerate(
            reranked_results,
            start=1,
        ):

            document = item["document"]

            print(
                f"{rank}. Chunk "
                f"{document.metadata.get('chunk_id')} "
                f"| {item['label']} "
                f"| RRF "
                f"{item['rrf_score']:.6f}"
            )


        # ----------------------------------------------------
        # Select Final Documents
        # ----------------------------------------------------

        final_documents = [
            item["document"]

            for item in reranked_results[
                :FINAL_K
            ]
        ]


        # ----------------------------------------------------
        # Format Context
        # ----------------------------------------------------

        context = format_documents(
            final_documents
        )


        # ----------------------------------------------------
        # Generate Answer
        # ----------------------------------------------------

        answer = answer_chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )


        # ----------------------------------------------------
        # Answer
        # ----------------------------------------------------

        print(
            "\nAI:"
        )

        print(
            answer
        )


        # ----------------------------------------------------
        # Final Sources
        # ----------------------------------------------------

        print(
            "\nFinal Sources:"
        )


        for rank, document in enumerate(
            final_documents,
            start=1,
        ):

            source = document.metadata.get(
                "source",
                "unknown",
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                "unknown",
            )


            print(
                f"{rank}. "
                f"{source} "
                f"(Chunk {chunk_id})"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()