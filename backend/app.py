from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2 as pdf
import json
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
import time
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
CORS(app)

load_dotenv()

# Initialize ChromaDB client and collection
client = chromadb.Client()
collection = client.get_or_create_collection("image_labels")

# Initialize SentenceTransformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load labels from JSON file
try:
    with open(os.path.join(os.path.dirname(__file__), 'labels.json'), 'r') as f:
        labels = json.load(f)
except FileNotFoundError:
    labels = {}

def populate_vector_database():
    for label, image_path in labels.items():
        existing_records = collection.get(where={"label": label})
        if len(existing_records["ids"]) == 0:
            embedding = embedder.encode(label).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[label],
                metadatas=[{"image_path": image_path}],
                ids=[label]
            )

# Initialize vector database on startup
populate_vector_database()

def load_documents():
    # Load both PDF and CSV documents
    pdf_path = os.path.join(os.path.dirname(__file__), 'quicklinks.pdf')
    csv_path = os.path.join(os.path.dirname(__file__), 'placement_details_complete.csv')

    pdf_loader = PyPDFLoader(pdf_path)
    csv_loader = CSVLoader(file_path=csv_path)

    pdf_documents = pdf_loader.load()
    csv_documents = csv_loader.load()

    return pdf_documents + csv_documents


def extract_pdf_text(file):
    """Extracts text from an uploaded PDF file."""
    reader = pdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_gemini_response(prompt):
    """Gets a response from the Gemini API."""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text


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


def create_vector_store(documents, embeddings):
    persist_directory = os.path.join(os.path.dirname(__file__), 'chroma_db_combined')
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


def initialize_qa_chain():
    global qa_chain, chat_model

    # Load combined documents
    documents = load_documents()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings and vector store
    embeddings = initialize_embeddings()
    vectordb = create_vector_store(documents=texts, embeddings=embeddings)

    # Create prompt template
    prompt_template = """
    ## Safety and Respect Come First!

    You are programmed to be a helpful and harmless AI. You will not answer requests that promote:

    * **Harassment or Bullying:** Targeting individuals or groups with hateful or hurtful language.
    * **Hate Speech:**  Content that attacks or demeans others based on race, ethnicity, religion, gender, sexual orientation, disability, or other protected characteristics.
    * **Violence or Harm:**  Promoting or glorifying violence, illegal activities, or dangerous behavior.
    * **Misinformation and Falsehoods:**  Spreading demonstrably false or misleading information.

    **How to Use You:**

    1. **Provide Context:** Give me background information on a topic.
    2. **Ask Your Question:** Clearly state your question related to the provided context.

    **Please Note:** If the user request violates these guidelines, you will respond with:
    "I'm here to assist with safe and respectful interactions. Your query goes against my guidelines. Let's try something different that promotes a positive and inclusive environment."

    ##  Answering User Question:

    Answer the question as precisely as possible using the provided context. The context can be from different topics. Please make sure the context is highly related to the question. If the answer is not in the context, you only say "answer is not in the context".

    Context: \n {context}
    Question: \n {question}
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Create QA chain
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def get_qa_response(question):
    """Gets a response from the QA chain with retry logic."""
    response = qa_chain.invoke({"query": question})
    return response.get('result', 'I could not find an answer to your question.')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Initialize QA chain if not already initialized
        global qa_chain
        if qa_chain is None:
            initialize_qa_chain()

        try:
            # Get response from QA chain with retry logic
            answer = get_qa_response(question)
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
    """Handles quick resume match requests."""
    try:
        print("trying")
        if 'resume' not in request.files or 'jobDescription' not in request.form:
            return jsonify({"error": "Missing resume or job description"}), 400

        resume_file = request.files['resume']
        job_description = request.form['jobDescription']

        # Extract resume text
        resume_text = extract_pdf_text(resume_file)
        

        # Create prompt for matching percentage
        matching_prompt = f"""
        Hey Act Like a skilled or very experienced ATS. Your task is to evaluate the resume based on the given job description. 
        Only provide the JD Match percentage as a response.

        Resume: {resume_text}
        Job Description: {job_description}

        Provide response in the format:
        {{"JD Match": "%"}}
        """

        response = get_gemini_response(matching_prompt)
        print(type(response))
        response = response.split(':')[1].strip('}% ')
        response = response[1:-2]
        print(response)
        return jsonify({"similarity": int(response)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/detailed-match', methods=['POST'])
def detailed_match():
    """Handles detailed resume analysis requests."""
    try:
        if 'resume' not in request.files or 'jobDescription' not in request.form:
            return jsonify({"error": "Missing resume or job description"}), 400

        resume_file = request.files['resume']
        job_description = request.form['jobDescription']

        # Extract resume text
        resume_text = extract_pdf_text(resume_file)

        # Create prompt for detailed feedback
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
        feedback = json.loads(response)
        return jsonify(feedback)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Compute embedding for the user query
        query_embedding = embedder.encode(query).tolist()
        
        # Search for the most similar label
        results = collection.query(query_embeddings=[query_embedding], n_results=1)
        
        if results["documents"]:
            matched_label = results["documents"][0][0]
            metadata = results["metadatas"][0][0]
            image_path = metadata["image_path"]
            
            if os.path.exists(image_path):
                return send_file(image_path, mimetype='image/jpeg')
            else:
                return jsonify({'error': 'Image not found'}), 404
        else:
            return jsonify({'error': 'No matching image found'}), 404

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    chat_model = initialize_chat_model()
    app.run(debug=True, port=5000)
