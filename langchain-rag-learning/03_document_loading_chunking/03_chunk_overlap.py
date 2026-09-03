from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
Retrieval-Augmented Generation allows an LLM to retrieve
external information before generating its answer.

The retrieval system searches a knowledge base and returns
relevant documents to the language model.

Chunk overlap helps preserve information that might otherwise
be separated across chunk boundaries.
"""


splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=60
)


chunks = splitter.split_text(text)


for index, chunk in enumerate(chunks):

    print("\n" + "=" * 50)

    print(f"CHUNK {index}")

    print("=" * 50)

    print(chunk)