from pathlib import Path
from langchain_core.documents import Document

MARKDOWN_PATH = Path("03_document_loading_chunking/outputs/Cardiac_Arrest.md") 

#Read Extraction Markdown
markdown_text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

#Create LangChain Document

document = Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "extraction_method": "pymupdf4llm",
        "format": "markdown",
        "domain": "healthcare",
        "topic": "cardiac_arrest"
    }
)

# 3. INSPECT

print(
    "Object type:",
    type(document)
)


print(
    "\nTotal characters:",
    len(document.page_content)
)


print(
    "\nMetadata:"
)

print(
    document.metadata
)


print(
    "\nContent preview:\n"
)

print(
    document.page_content[:3000]
)


