from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import PyPDF2 as pdf
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import json

app = Flask(__name__)
CORS(app)

# Initialize global variables
qa_chain = None
chat_model = None

# Hardcoded Google API Key
GOOGLE_API_KEY = "AIzaSyC9NvHQGcU34fXbs6fcgIYtXEeK14E4InM"  # Replace with your actual API key

def initialize_chat_model():
    try:
        safety_settings = {
            HarmCategory.DANGEROUS_CONTENT: HarmBlockThreshold.MEDIUM_AND_ABOVE,
            HarmCategory.HATE_SPEECH: HarmBlockThreshold.MEDIUM_AND_ABOVE,
            HarmCategory.HARASSMENT: HarmBlockThreshold.MEDIUM_AND_ABOVE,
            HarmCategory.SEXUALLY_EXPLICIT: HarmBlockThreshold.MEDIUM_AND_ABOVE,
        }
        
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=GOOGLE_API_KEY,
            safety_settings=safety_settings,
            convert_system_message_to_human=True
        )
        print("Chat model initialized successfully.")
        return model
    except Exception as e:
        print(f"Error initializing chat model: {str(e)}")
        return None

def initialize_embeddings():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            task_type="retrieval_query",
            google_api_key=GOOGLE_API_KEY,
            request_timeout=120
        )
        print("Embeddings initialized successfully.")
        return embeddings
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    try:
        reader = pdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return None

# ... keep existing code (load_documents, create_vector_store, initialize_qa_chain functions)

@app.route('/match', methods=['POST'])
def get_matching_percentage():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('jobDescription')
        
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
        
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text:
            return jsonify({'error': 'Failed to extract text from resume'}), 400

        matching_prompt = f"""
        Act like a skilled ATS (Applicant Tracking System). Evaluate the resume based on the given job description. 
        Only provide the match percentage as a number between 0 and 100.

        Resume: {resume_text}
        Job Description: {job_description}

        Return only the number, no other text.
        """

        response = chat_model.invoke(matching_prompt)
        try:
            similarity = int(response.content.strip())
            return jsonify({'similarity': similarity})
        except ValueError:
            return jsonify({'error': 'Invalid response format'}), 500

    except Exception as e:
        print(f"Error processing match request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/detailed-match', methods=['POST'])
def get_detailed_feedback():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No resume file provided'}), 400
        
        resume_file = request.files['resume']
        job_description = request.form.get('jobDescription')
        
        if not job_description:
            return jsonify({'error': 'No job description provided'}), 400
        
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text:
            return jsonify({'error': 'Failed to extract text from resume'}), 400

        feedback_prompt = f"""
        Act like a skilled ATS (Applicant Tracking System). Evaluate the resume based on the given job description.
        Provide detailed feedback in the following JSON format:
        {{
          "JD Match": "percentage as string with % symbol",
          "Missing Keywords": ["array of missing important keywords"],
          "Profile Summary": "brief summary of profile alignment",
          "Strengths": "key strengths based on resume",
          "Weaknesses": "areas needing improvement",
          "Recommend Courses & Resources": "relevant course suggestions"
        }}

        Resume: {resume_text}
        Job Description: {job_description}

        Ensure the response is valid JSON.
        """

        response = chat_model.invoke(feedback_prompt)
        try:
            feedback_dict = json.loads(response.content.strip())
            return jsonify(feedback_dict)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid response format'}), 500

    except Exception as e:
        print(f"Error processing detailed match request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        if qa_chain is None:
            success = initialize_qa_chain()
            if not success:
                return jsonify({'error': 'Failed to initialize QA chain'}), 500

        response = qa_chain.invoke({"query": question})
        answer = response.get('result', 'I could not find an answer to your question.')

        return jsonify({'answer': answer})

    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize chat model
    chat_model = initialize_chat_model()
    if not chat_model:
        print("Failed to initialize chat model. Exiting...")
        exit(1)
    
    # Initialize QA chain
    success = initialize_qa_chain()
    if not success:
        print("Failed to initialize QA chain. Exiting...")
        exit(1)
    
    print("Server initialized successfully!")
    app.run(debug=True, port=5000)