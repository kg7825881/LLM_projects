from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,   
)

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

llm = ChatOllama(
    model="gemma3:4b",
    temperature = 0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Use the conversation history to understand
the user's current question.
"""
        ),
        MessagesPlaceholder(
            variable_name="chat_history"
        )

        (
            "human",
            "{question}"
        )
    ]
)

chain = prompt | llm | StrOutputParser()

chat_history = [
    HumanMessage(
        content="Which model performed best?"
    ),

    AIMessage(
        content="Random Forest performed best."
    )
]

response = chain.invoke(
    {
        "chat_history": chat_history,
        "question":"What was its accuracy?"
    }
)

print(response)