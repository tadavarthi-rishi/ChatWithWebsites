import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

def get_response(user_input):
    return "I dont Know"

def get_vectorstore_from_url(url):
    # To Load/extract the documents from the website
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    ## create a vector stores from chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


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

if website_url is None or website_url == "":
    st.info("Please enter a website URL in the sidebar.")
else:
    document_chunks = get_vectorstore_from_url(website_url)

    with st.sidebar:
        st.write(document_chunks)
    #user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":

        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
