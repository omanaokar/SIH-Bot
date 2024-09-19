import os
import streamlit as st
from streamlit_chat import message
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests

load_dotenv()

def ask_flask_api(question, language):
    # Send question and language to Flask API
    api_url = "http://localhost:5001/ask_question"
    response = requests.post(api_url, json={"question": question, "language": language})
    if response.status_code == 200:
        return response.json().get("answer")
    else:
        return "Error communicating with backend"

def main():
    st.set_page_config(page_title="Judiciary Assistance System", page_icon="🚜", layout="wide")
    
    st.title("🚜 Judiciary Assistance System")
    st.subheader("Your AI assistant for judiciary information and details")

    # Language selection dropdown
    languages = {
        "English": "English",
        "Hindi": "Hindi",
        "Hinglish": "Hinglish",
        "Gujarati": "Gujarati",
        "Urdu": "Urdu",
        "Punjabi": "Punjabi"
    }
    selected_language = st.selectbox("Select Language", list(languages.keys()), index=0)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for i, chat in enumerate(st.session_state.messages):
        message(chat["content"], is_user=chat["is_user"], key=str(i))

    # React to user input
    if prompt := st.chat_input(f"Ask about judiciary system in {selected_language}:"):
        # Display user message in chat message container
        st.chat_message("user").write(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"content": prompt, "is_user": True})

        with st.spinner("Thinking..."):
            response = ask_flask_api(prompt, languages[selected_language])
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"content": response, "is_user": False})

    # Sidebar with additional information
    with st.sidebar:
        st.subheader("About")
        st.write("This AI assistant is designed to help with judiciary system. Feel free to ask any questions about pending cases, judiciary procedures")
        
        st.subheader("Need More Help?")
        st.info("If you need further assistance, consider contacting your local judicial office or the judiciary support line.")

if __name__ == "__main__":
    main()
