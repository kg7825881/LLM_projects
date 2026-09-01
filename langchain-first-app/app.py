from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

model = ChatOllama(
    model = "gemma3:4b"
)

prompts = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI Engineering Assistant."
    ),
    ("human",
     "{question}"
    )
])

chain = prompts | model

response = chain.invoke({
    "question": "What is RAG?"
})

print(response.content)