from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)


sessions = {}


def get_history(session_id):

    if session_id not in sessions:

        sessions[session_id] = []

    return sessions[session_id]


session_id = "user_001"


history = get_history(
    session_id
)


history.append(
    HumanMessage(
        content="Which model performed best?"
    )
)


history.append(
    AIMessage(
        content="Random Forest performed best."
    )
)