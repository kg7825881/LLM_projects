from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

MARKDOWN_PATH = Path("03_document_loading_chunking/outputs/Cardiac_Arrest.md")

#Load the Markdown 
markdown_text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

#Create Langchain Document
document = Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "extraction_type":  "pymupdf4llm",
        "format": "markdown",
        "domain":"healthcare",
        "topic": "cardiac_arrest"
    }
)

documents=[document]

print(
    f"Loaded {len(documents)} LangChain Document"
)

print(
    f"Total Characters {len(document.page_content)}"
)


# Chunk

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = splitter.split_documents(documents)


# Add chunk IDs
for index, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = index

# Inspect

for chunk in chunks[:5]:

    print("\n" + "=" * 60)

    print(chunk.page_content)

    print("\nMETADATA:")

    print(chunk.metadata)