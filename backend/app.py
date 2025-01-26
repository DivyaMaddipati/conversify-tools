from flask import Flask, request, jsonify
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

app = Flask(__name__)
CORS(app)

# Initialize global variables
qa_chain = None
chat_model = None

def load_documents():
    # Load both PDF and CSV documents
    pdf_path = os.path.join(os.path.dirname(__file__), 'quicklinks.pdf')
    csv_path = os.path.join(os.path.dirname(__file__), 'placement_details_complete.csv')

    pdf_loader = PyPDFLoader(pdf_path)
    csv_loader = CSVLoader(file_path=csv_path)

    pdf_documents = pdf_loader.load()
    csv_documents = csv_loader.load()

    return pdf_documents + csv_documents


def initialize_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key='AIzaSyC9NvHQGcU34fXbs6fcgIYtXEeK14E4InM',
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
        google_api_key='AIzaSyC9NvHQGcU34fXbs6fcgIYtXEeK14E4InM',
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
    chat_model = initialize_chat_model()
    app.run(debug=True, port=5000)