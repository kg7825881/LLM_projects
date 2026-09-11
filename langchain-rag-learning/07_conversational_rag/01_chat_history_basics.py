from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

chat_history = []

chat_history.append(
    HumanMessage(
        content="Which machine learning model performed best?"
    )
)

chat_history.append(
    AIMessage(
        content="Random Forest performed best."
    )
)

chat_history.append(
    HumanMessage(
        content="What was its accuracy?"
    )
)

for message in chat_history:
    print(
        type(message).__name__,
        ":",
        message.content
    )