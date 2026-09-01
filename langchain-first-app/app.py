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

formatted_prompt = prompts.invoke({
    "question": "What is RAG?"
})

print(formatted_prompt)