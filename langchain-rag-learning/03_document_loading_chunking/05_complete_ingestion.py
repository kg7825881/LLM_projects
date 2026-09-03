from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Configuration
# ==========================================
MARKDOWN_PATH = Path("03_document_loading_chunking/outputs/Cardiac_Arrest.md") 

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


# ==========================================
# 1. Load the Markdown
# ==========================================

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
        "topic": "cardiac_arrest_prediction"
    }
)

documents = [document]


print(
    f"Loaded {len(documents)} LangChain document"
)


print(
    f"Total characters: {len(document.page_content)}"
)

# 2. Create splitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    length_function=len
)


# 3. Split documents

chunks = splitter.split_documents(
    documents
)


print(
    f"Created {len(chunks)} chunks"
)

# 4. Add chunk identifiers

for index, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = index


# 5. Inspect result

for index, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 70)

    print(f"CHUNK {index}")

    print("-" * 70)

    print(chunk.page_content)

    print("\nMETADATA:")

    print(chunk.metadata)