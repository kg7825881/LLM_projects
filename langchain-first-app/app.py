from langchain_ollama import ChatOllama

model = ChatOllama(
    model = "gemma3:4b"
)

response = model.invoke(
    "Explain machine learning in simple terms."
)

print(response)