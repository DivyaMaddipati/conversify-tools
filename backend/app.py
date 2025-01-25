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

def extract_text_from_pdf(pdf_file):
    reader = pdf.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

def load_documents():
    try:
        # Load both PDF and CSV documents
        pdf_path = os.path.join(os.path.dirname(__file__), 'quicklinks.pdf')
        csv_path = os.path.join(os.path.dirname(__file__), 'placement_details_complete.csv')

        pdf_loader = PyPDFLoader(pdf_path)
        csv_loader = CSVLoader(file_path=csv_path)

        pdf_documents = pdf_loader.load()
        csv_documents = csv_loader.load()

        return pdf_documents + csv_documents
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []

def initialize_embeddings():
    try:
        return GoogleGenerativeAIEmbeddings(
            model='models/embedding-001',
            task_type="retrieval_query"
        )
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        return None

def initialize_chat_model():
    try:
        safety_settings = {
            HarmCategory.DANGEROUS_CONTENT: HarmBlockThreshold.MEDIUM,
            HarmCategory.HATE_SPEECH: HarmBlockThreshold.MEDIUM,
            HarmCategory.HARASSMENT: HarmBlockThreshold.MEDIUM,
            HarmCategory.SEXUALLY_EXPLICIT: HarmBlockThreshold.MEDIUM,
        }
        
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        print(f"Error initializing chat model: {str(e)}")
        return None

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
        # Load combined documents
        documents = load_documents()
        if not documents:
            raise Exception("No documents loaded")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Initialize embeddings and vector store
        embeddings = initialize_embeddings()
        if not embeddings:
            raise Exception("Failed to initialize embeddings")

        vectordb = create_vector_store(documents=texts, embeddings=embeddings)
        if not vectordb:
            raise Exception("Failed to create vector store")

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
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        return None

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

        # Get response from QA chain
        response = qa_chain.invoke({"query": question})
        answer = response.get('result', 'I could not find an answer to your question.')

        return jsonify({'answer': answer})

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize chat model before running the app
    chat_model = initialize_chat_model()
    if not chat_model:
        print("Failed to initialize chat model. Exiting...")
        exit(1)
    
    app.run(debug=True, port=5000)