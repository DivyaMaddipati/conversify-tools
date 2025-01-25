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
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import json

app = Flask(__name__)
CORS(app)

# Initialize global variables
qa_chain = None
chat_model = None

# ... keep existing code (load_documents, initialize_embeddings, initialize_chat_model, create_vector_store, initialize_qa_chain functions)

def extract_text_from_pdf(pdf_file):
    reader = pdf.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

@app.route('/match', methods=['POST'])
def match_resume():
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

        response = chat_model.invoke(matching_prompt)
        try:
            match_data = json.loads(response.text)
            match_percentage = int(match_data["JD Match"].strip("%"))
            return jsonify({"similarity": match_percentage})
        except:
            return jsonify({"error": "Failed to parse AI response"}), 500

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/detailed-match', methods=['POST'])
def detailed_match():
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

        response = chat_model.invoke(feedback_prompt)
        try:
            feedback_data = json.loads(response.text)
            return jsonify(feedback_data)
        except:
            return jsonify({"error": "Failed to parse AI response"}), 500

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    chat_model = initialize_chat_model()
    app.run(debug=True, port=5000)