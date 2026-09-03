from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Configuration
# ==========================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


# ==========================================
# 1. Load document
# ==========================================

loader = PyPDFLoader(
    "langchain-rag-learning\\03_document_loading_chunking\\documents\\Cardiac Arrest.pdf"
)

documents = loader.load()


print(
    f"Loaded {len(documents)} documents"
)


# 2. Enrich metadata

for document in documents:

    document.metadata.update({
        "domain": "healthcare",
        "document_type": "research_paper",
        "topic": "cardiac_arrest_prediction"
    })

# 3. Create splitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    length_function=len
)


# 4. Split documents

chunks = splitter.split_documents(
    documents
)


print(
    f"Created {len(chunks)} chunks"
)

# 5. Add chunk identifiers

for index, chunk in enumerate(chunks):

    chunk.metadata["chunk_id"] = index


# 6. Inspect result

for index, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 70)

    print(f"CHUNK {index}")

    print("-" * 70)

    print(chunk.page_content)

    print("\nMETADATA:")

    print(chunk.metadata)