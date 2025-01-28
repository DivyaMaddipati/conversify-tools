from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import google.generativeai as genai
from dotenv import load_dotenv
import os

from modules.resume_handler import handle_resume_match, handle_detailed_match
from modules.chat_handler import load_documents, create_qa_chain, get_qa_response
from modules.image_handler import ImageHandler
from audio_handler import transcribe_audio

app = Flask(__name__)
CORS(app)

load_dotenv()

# Initialize components
image_handler = ImageHandler()
image_handler.populate_vector_database()

def initialize_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key='AIzaSyDp_2EUaFTtBtW5ikHA-5WJMoHEwszSRVA',
        task_type="retrieval_query"
    )

def initialize_chat_model():
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key='AIzaSyDp_2EUaFTtBtW5ikHA-5WJMoHEwszSRVA',
        temperature=0.3,
        safety_settings=safety_settings
    )

# Initialize QA chain
chat_model = initialize_chat_model()
embeddings = initialize_embeddings()
documents = load_documents()
qa_chain = create_qa_chain(chat_model, documents, embeddings)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        try:
            answer = get_qa_response(qa_chain, question)
            return jsonify({'answer': answer})
        except Exception as e:
            return jsonify({
                'error': 'Service temporarily unavailable. Please try again in a few moments.',
                'details': str(e)
            }), 503

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/match', methods=['POST'])
def match():
    if 'resume' not in request.files or 'jobDescription' not in request.form:
        return jsonify({"error": "Missing resume or job description"}), 400
    return handle_resume_match(request.files['resume'], request.form['jobDescription'])

@app.route('/detailed-match', methods=['POST'])
def detailed_match():
    if 'resume' not in request.files or 'jobDescription' not in request.form:
        return jsonify({"error": "Missing resume or job description"}), 400
    return handle_detailed_match(request.files['resume'], request.form['jobDescription'])

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    return image_handler.generate_image(query)

@app.route('/transcribe-audio', methods=['POST'])
def handle_transcription():
    try:
        data = request.json
        if not data or 'audio' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
            
        transcript = transcribe_audio(data['audio'])
        return jsonify({'transcript': transcript})
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)