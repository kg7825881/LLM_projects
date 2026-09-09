from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research assistant.

Answer the user's questions using only the provided context.
    
If the answer cannot be found in the context,
say:

"I could not find this information in the provided document"

Do not invent information
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{query}
"""
        )
    ]
)