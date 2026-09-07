from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).parent.parent

MARKDOWN_PATH = (
    BASE_DIR
    / "03_document_loading_chunking"
    / "outputs" 
    / "Cardiac Arrest.md"
)

VECTOR_DB_DIR = (
    Path(__file__).parent
    / "chroma_db"
)

text = MARKDOWN_PATH.read_text(
    encoding="utf-8"
)

document = Document(
    page_content=text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "domain": "healthcare",
        "topic": "cardiac_arrest_prediction" 
    }
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(
    [document]
)

for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = index

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(VECTOR_DB_DIR)
)

print(
    f"Stored {len(chunks)} chunks "
    "in Chroma"
)