from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load a PDF document 
loader = PyPDFLoader(
    "langchain-rag-learning\\03_document_loading_chunking\\documents\\Cardiac Arrest.pdf"
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

chunks = text_splitter.split_documents(
    documents
)

print("Total Chunks:", len(chunks))

#Inspect chunks 
for i, chunks in enumerate(chunks[:5]):
    print("="*70)
    print("Chunk:", i)
    print(chunks.page_content)
    print("METADATA:")
    print(chunks.metadata)