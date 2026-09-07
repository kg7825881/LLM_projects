from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MARKDOWN_PATH = (
    PROJECT_ROOT
    / "03_document_loading_chunking"
    / "outputs"
    / "Cardiac Arrest.md"
)

markdown_text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

document = Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.md"
    }
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(
    [document]
)



embeddings = OllamaEmbeddings(
    model = "nomic-embed-text"
)

vector = embeddings.embed_documents(
    [
        chunk.page_content
        for chunk in chunks[:5]
    ]
)

print("Chunks embedded:", len(vector))

print(
    "Vector dimension:",
    len(vector[0])
)