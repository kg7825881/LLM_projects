from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

model = ChatOllama(
    model = "gemma3:4b"
)

messages = [
    SystemMessage(
        content="You are an AI engineering Teacher."
    ),
    HumanMessage(
        content="What is Langchain?"
    )
]

response = model.invoke(messages)

print(response.content)