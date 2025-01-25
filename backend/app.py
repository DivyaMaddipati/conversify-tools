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

def initialize_chat_model():
    try:
        # Updated safety settings to be less restrictive
        safety_settings = {
            HarmCategory.DANGEROUS_CONTENT: HarmBlockThreshold.LOW_AND_ABOVE,
            HarmCategory.HATE_SPEECH: HarmBlockThreshold.LOW_AND_ABOVE,
            HarmCategory.HARASSMENT: HarmBlockThreshold.LOW_AND_ABOVE,
            HarmCategory.SEXUALLY_EXPLICIT: HarmBlockThreshold.LOW_AND_ABOVE,
        }
        
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            safety_settings=safety_settings,
            convert_system_message_to_human=True
        )
        return model
    except Exception as e:
        print(f"Error initializing chat model: {str(e)}")
        return None

def initialize_embeddings():
    try:
        return GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            task_type="retrieval_query",
            request_timeout=120
        )
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    try:
        reader = pdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += str(page.extract_text())
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def load_documents():
    try:
        pdf_path = os.path.join(os.path.dirname(__file__), 'quicklinks.pdf')
        csv_path = os.path.join(os.path.dirname(__file__), 'placement_details_complete.csv')

        if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
            print("Required files not found!")
            return []

        pdf_loader = PyPDFLoader(pdf_path)
        csv_loader = CSVLoader(file_path=csv_path)

        pdf_documents = pdf_loader.load()
        csv_documents = csv_loader.load()

        return pdf_documents + csv_documents
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []

def create_vector_store(documents, embeddings):
    try:
        persist_directory = os.path.join(os.path.dirname(__file__), 'chroma_db_combined')
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()
        return vectordb
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None

def initialize_qa_chain():
    global qa_chain, chat_model

    try:
        documents = load_documents()
        if not documents:
            raise Exception("No documents loaded")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        embeddings = initialize_embeddings()
        if not embeddings:
            raise Exception("Failed to initialize embeddings")

        vectordb = create_vector_store(documents=texts, embeddings=embeddings)
        if not vectordb:
            raise Exception("Failed to create vector store")

        prompt_template = """
        Answer the question as precisely as possible using the provided context. 
        If the answer is not in the context, say "I don't have enough information to answer that question."

        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

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
        
        return True
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        return False

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
    # Initialize chat model before running the app
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