#Import
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)
from langchain_core.messages import (HumanMessage, AIMessage)
from langchain_core.output_parsers import StrOutputParser

# Configure Streamlit
st.set_page_config(
    page_title="AI Engineer Assistant",
    page_icon="🤖"
)
st.title("🤖 AI Engineer Assistant")

#Intialize the Model
model = ChatOllama(
    model = "gemma3:4b",
    temperature = 0.3,
    top_p = 0.9,
    top_k = 40,
    num_ctx = 8192
)

# Create a chat Prompt template
prompts = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an AI Engineer Assistant.
        
        Explain the concepts clearly and precisely.
        Use architectural diagrams, code snippets and examples when useful.
        """
    ),
    MessagesPlaceholder(
        variable_name = "chat_history"
    ),
    (
        "human",
        "{question}"
    )
])

#Build LCEL chain
parser = StrOutputParser()

chain = prompts | model | parser

#Store the conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display the previous messages
for messages in st.session_state.messages:
    with st.chat_message(messages["role"]):
        st.markdown(messages["content"])

#Accept user input
question = st.chat_input(
    "Ask something about AI...."
)
if question:
    st.session_state.messages.append({
        "role": "user", 
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

#Convert Streamlit history into LangChain messages
    
    chat_history = []
    for message in st.session_state.messages[:-1]:

        if message["role"] == "user":
            chat_history.append(
                HumanMessage(
                    content = message["content"]
                )
            )
        elif message["role"] == "assistant":
            
            chat_history.append(
                AIMessage(
                    content  = message["content"]
                )
            )

    #Run Chain
    response = chain.invoke({
        "chat_history": chat_history,
        "question": question
    })

    #Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    #Store assistant response 
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })