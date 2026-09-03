from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load
loader = PyPDFLoader(
    "langchain-rag-learning/03_document_loading_chunking/documents/Cardiac Arrest.pdf"
)

documents = loader.load()

# Add custom metadata

for document in documents:

    document.metadata["domain"] = "healthcare"
    document.metadata["document_type"] = "research_paper"
    document.metadata["topic"] = "cardiac_arrest_prediction"
    document.metadata["subtopic"] = "machine_learning"
    document.metadata["module"] = "langchain_learning"

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