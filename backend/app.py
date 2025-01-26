from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import PyPDF2 as pdf
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize global variables
qa_chain = None
chat_model = None
API_KEY = 'AIzaSyDw8Hip8rTDFG-Dyd2C88wuMD1WgUl3Y3c'

# Configure Gemini
genai.configure(api_key=API_KEY)

def extract_text_from_pdf(file):
    reader = pdf.PdfReader(file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def load_documents(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

def initialize_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)

def create_vector_store(documents, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return Chroma.from_documents(texts, embeddings)

def initialize_chat_model():
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=API_KEY,
        temperature=0.3,
        safety_settings=safety_settings
    )

def initialize_qa_chain():
    global qa_chain, chat_model
    try:
        # Initialize embeddings
        embeddings = initialize_embeddings()
        
        # Load and process documents
        documents = load_documents("backend/dataset/placement_details_complete.csv")
        
        # Create vector store
        vector_store = create_vector_store(documents, embeddings)
        
        # Initialize chat model if not already initialized
        if chat_model is None:
            chat_model = initialize_chat_model()
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        raise

@app.route('/match', methods=['POST', 'OPTIONS'])
def get_match():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('jobDescription')
        
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
            
        resume_text = extract_text_from_pdf(resume_file)
        matching_prompt = f"""
        Hey Act Like a skilled or very experienced ATS. Your task is to evaluate the resume based on the given job description. 
        Only provide the JD Match percentage as a response.

        Resume: {resume_text}
        Job Description: {job_description}

        Provide response in the format:
        {{"JD Match": "%"}}
        """
        
        response = get_gemini_response(matching_prompt)
        return jsonify({'similarity': response})
        
    except Exception as e:
        print(f"Error processing match request: {str(e)}")
        return jsonify({'error': 'Failed to process the matching request'}), 500

@app.route('/detailed-match', methods=['POST', 'OPTIONS'])
def get_detailed_match():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('jobDescription')
        
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
            
        resume_text = extract_text_from_pdf(resume_file)
        feedback_prompt = f"""
        Hey Act Like a skilled or very experienced ATS. Your task is to evaluate the resume based on the given job description. 
        Provide detailed feedback including:
        1. JD Match (%): Assign the percentage matching based on the Job Description (JD) and the resume provided.
        2. Missing Keywords: Identify missing keywords with high accuracy and relevance to the JD.
        3. Profile Summary: Summarize the profile's strengths and alignment with the JD.
        4. Strengths: Highlight the key strengths of the candidate based on the resume.
        5. Weaknesses: Point out weaknesses or areas that need improvement based on the JD.
        6. Recommend Courses & Resources: Suggest relevant courses or resources to improve the profile and match the JD better.

        Resume: {resume_text}
        Job Description: {job_description}

        Provide the response in this format:
        {{
          "JD Match": "%",
          "Missing Keywords": [],
          "Profile Summary": "",
          "Strengths": "",
          "Weaknesses": "",
          "Recommend Courses & Resources": ""
        }}
        """
        
        response = get_gemini_response(feedback_prompt)
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing detailed match request: {str(e)}")
        return jsonify({'error': 'Failed to process the detailed matching request'}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Initialize QA chain if not already initialized
        global qa_chain
        if qa_chain is None:
            initialize_qa_chain()

        # Get response from QA chain
        response = qa_chain.invoke({"query": question})
        answer = response.get('result', 'I could not find an answer to your question.')

        return jsonify({'answer': answer})

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    chat_model = initialize_chat_model()
    app.run(debug=True, port=5000)