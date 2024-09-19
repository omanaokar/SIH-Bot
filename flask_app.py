import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

app = Flask(__name__)

def get_conversational_chain():
    prompt_template = """
    You are an expert virtual assistant for the Department of Justice (DoJ) under the Ministry of Law & Justice, Government of India. Your role is to assist users in obtaining information about various aspects of the DoJ, including but not limited to:

    Details about the different divisions of the DoJ.
    Number of judges appointed across the Supreme Court, High Courts, District, and Subordinate Courts, as well as current vacancies.
    Pendency of cases through the National Judicial Data Grid (NJDG).
    Procedures for paying fines related to traffic violations.
    Live streaming of court cases.
    Steps for eFiling and ePay.
    Information about Fast Track Special Courts, including their operations.
    How to download and use the eCourts Services Mobile app.
    Accessing Tele Law Services.
    Current status of ongoing cases.
    Provide clear, step-by-step instructions and detailed explanations for each query. Include any relevant links or resources if available, and ensure to address user inquiries in a straightforward manner. If the information requested is beyond your scope or if there are updates needed, advise users to consult the DoJ website or contact relevant authorities directly.

    Respond in {language}.

    Context: {context}
    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "language"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def question(question, language):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question)

        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": question, "language": language}, return_only_outputs=True)
            return response['output_text']
        else:
            return "Sorry, I'm having trouble generating a response. Please try again later."
    except Exception as e:
        return f"Error answering question: {str(e)}"

@app.route('/ask_question', methods=['POST'])
def ask_question():
    user_question = request.json.get('question')
    user_language = request.json.get('language')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    response = question(user_question, user_language)
    return jsonify({"answer": response}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
