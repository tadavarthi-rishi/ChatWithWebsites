import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
def get_response(user_input):
    return "I dont Know"

#app config
st.set_page_config(page_title="Chat with websites", page_icon=":shark:")
st.title("Chat with websites")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
            AIMessage(content = "Hello, I am a chatbot. I can help you with your queries."),
    ]

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website Url")

#user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":

    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))


with st.sidebar:
    st.write(st.session_state.chat_history)
