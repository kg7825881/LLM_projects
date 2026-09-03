from langchain_community.document_loaders import PyPDFLoader

# Load a PDF document using PyPDFLoader
loader = PyPDFLoader(
    "langchain-rag-learning\\03_document_loading_chunking\\documents\\Cardiac Arrest.pdf"
)

documents = loader.load()

# Inspect result
print("Total Documents:", len(documents))

print("\n--- FIRST DOCUMENT ---")
print(documents[0].page_content)



print("\n--- METADATA ---")
print(documents[0].metadata)
