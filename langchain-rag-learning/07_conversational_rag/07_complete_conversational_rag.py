from pathlib import Path
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_chroma import Chroma
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings,
)
from langchain_core.output_parsers import StrOutputParser

# Configuration
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

VECTOR_DB_DIR = (
    PROJECT_DIR
    /"04_embeddings_vectorstore"
    / "chroma_db"
)

# Models
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Vectore Store
vector_store = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

# Retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)

# SESSION MEMORY
sessions = {}

def get_history(session_id):
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


# QUESTION REWRITING PROMPT
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You rewrite conversational questions into
standalone questions for document retrieval.

Use the conversation history only when it is
necessary to understand the latest question.

Resolve references such as:

- it
- its
- they
- them
- that model
- that method
- the previous result

Do not answer the question.

Only rewrite the question.

If the latest question is already standalone,
return it unchanged.
"""
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),
        (
            "human",
            "{question}"
        )
    ]
)

# QUESTION REWRITING CHAIN
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# DOCUMENT FORMATTER
def format_documents(documents):

    formatted_documents = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "unknown"
        )

        formatted_document = f"""
Source: {source}
Chunk ID: {chunk_id}

{document.page_content}
"""

        formatted_documents.append(
            formatted_document
        )

    return "\n\n".join(
        formatted_documents
    )

# ANSWER PROMPT
answer_prompt = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a document question-answering assistant.

Answer the user's question using only the
provided document context.

Rules:

1. Use only information from the provided context.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Use conversation history only to understand
   the conversation. Do not treat previous AI
   responses as authoritative document evidence.

5. If the provided context does not contain
   enough information to answer the question,
   respond exactly with:

   "I could not find this information in the provided document."

6. Keep the answer clear, concise, and factual.


DOCUMENT CONTEXT:

{context}
"""
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            (
                "human",
                "{question}"
            )
        ]
    )
)


# ANSWER CHAIN
answer_chain = answer_prompt | llm | StrOutputParser()

# CONVERSATIONAL RAG FUNCTION
def ask_question(session_id, question):
    # --------------------------------------------------------
    # 1. Get conversation history
    # --------------------------------------------------------
    
    chat_history = get_history(session_id)

     # --------------------------------------------------------
    # 2. Rewrite question
    # --------------------------------------------------------

    standalone_question = rewrite_chain.invoke(
        {
            "chat_history": chat_history,
            "question": question
        }
    )

    print(f"Standalone Question: {standalone_question}")

    # --------------------------------------------------------
    # 3. Retrieve relevant documents
    # --------------------------------------------------------

    documents = retriever.invoke(
        standalone_question
    ) 

     # --------------------------------------------------------
    # 4. Format retrieved documents
    # --------------------------------------------------------
    context = format_documents(
        documents
    )

    # --------------------------------------------------------
    # 5. Generate grounded answer
    # --------------------------------------------------------
    answer = answer_chain.invoke(
        {
            "context": context,
            "chat_history": chat_history,
            "question": question
        }
    )

    # --------------------------------------------------------
    # 6. Update conversation history
    # --------------------------------------------------------

    chat_history.append(
            HumanMessage(
                content=question
            )
        )


    chat_history.append(
        AIMessage(
            content=answer
        )
    )

    # --------------------------------------------------------
    # 7. Return results
    # --------------------------------------------------------

    return answer, documents

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print("CONVERSATIONAL RAG")

    print("=" * 70)

    print("Ask questions about the Cardiac Arrest research paper.")

    print("Type 'exit' to stop.")


    # For this learning application we use
    # one fixed session.

    session_id = "user_001"

    while True:

        # User Input
        question = input("\nYou: ").strip()

        # Exit
        if question.lower() in {"exit","quit"}:

            print("Conversation ended.")
            break

        if not question:
            continue

        # Conversational RAG
        answer, documents = ask_question(session_id=session_id, question=question)

        # Print Answer
        print("AI:")

        print(answer)

        # Print Retrieved Sources

        print("Retrieved Sources:")

        for index, document in enumerate(documents, start=1):

            source = document.metadata.get(
                "source",
                "unknown"
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                "unknown"
            )

            print(
                f"{index}. "
                f"{source} "
                f"(Chunk {chunk_id})"
            )

# ENTRY POINTs

if __name__ == "__main__":

    main()