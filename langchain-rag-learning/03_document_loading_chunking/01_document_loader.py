from langchain_core.documents import Document

doc = Document(
    page_content="""
    New Jersey property damage claims may have
    a limitation period depending on the applicable statute.
    """,
    metadata={
        "source": "legal_rules.pdf",
        "jurisdiction": "New Jersey"
    }
)

print(doc)
print(doc.page_content)
print(doc.metadata)
print(doc.metadata["jurisdiction"])
