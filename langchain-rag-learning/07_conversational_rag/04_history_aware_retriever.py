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

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Rewrite the latest user question into a fully self-contained
standalone question using the conversation history.

Resolve all pronouns and ambiguous references such as:
"it", "its", "that", "this", "they", "them", "the model",
and replace them with the specific entity they refer to.

The rewritten question must make sense to someone who has
not seen the conversation history.

Do not answer the question.
Return only the rewritten question.

If no information from the conversation history is needed,
return the question unchanged.

Example:

Conversation:
User: Which machine learning model performed best?
Assistant: Random Forest performed best.

Latest question:
What was its accuracy?

Rewritten question:
What was the accuracy of the Random Forest model?
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

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# History 
chat_history = [
    HumanMessage(
        content="Which machine learning model performed best?"
    ),

    AIMessage(
        content="Random Forest performed best."
    )
]

question = "What was its accuracy?"

# Rewrite
standalone_question = rewrite_chain.invoke(
    {
        "chat_history": chat_history,
        "question": question
    }
)

print(f"Standalone Question: {standalone_question}")

# Retrieve 
documents = retriever.invoke(
    standalone_question
)

print(
    f"\nRetrieved {len(documents)} documents"
)


for index, document in enumerate(documents):

    print("\n" + "=" * 70)

    print(
        f"RESULT {index + 1}"
    )

    print("-" * 70)

    print(
        document.page_content[:700]
    )

    print(
        "\nMETADATA:"
    )

    print(
        document.metadata
    )