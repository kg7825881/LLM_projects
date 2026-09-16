from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# SAMPLE DOCUMENTS
# ============================================================
documents = [

    Document(
        page_content=(
            "Random Forest achieved the best "
            "classification performance."
        )
    ),

    Document(
        page_content=(
            "Logistic Regression was used as "
            "a baseline classification model."
        )
    ),

    Document(
        page_content=(
            "Support Vector Machine was evaluated "
            "for cardiac arrest prediction."
        )
    ),

    Document(
        page_content=(
            "Deep neural networks are commonly "
            "used for image classification."
        )
    ),
]


# CREATE BM25 RETRIEVER
# ============================================================
retriever = BM25Retriever.from_documents(
    documents
)

retriever.k = 2

# SEARCH
# ============================================================
query = "Random Forest classification"

results = retriever.invoke(
    query
)

# DISPLAY RESULTS
# ============================================================

for index, document in enumerate(
    results,
    start=1
):

    print(
        f"\nRESULT {index}"
    )

    print(
        document.page_content
    )