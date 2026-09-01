from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()

chain = prompts | model | parser

response = chain.invoke({
    "question": "What's the difference between RAG and fine-tuning?"
})

print(response)