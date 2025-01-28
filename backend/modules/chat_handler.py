from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader
import os
from tenacity import retry, stop_after_attempt, wait_exponential

def load_documents():
    # Load both PDF and CSV documents
    pdf_path = os.path.join(os.path.dirname(__file__), '../quicklinks.pdf')
    csv_path = os.path.join(os.path.dirname(__file__), '../placement_details_complete.csv')

    pdf_loader = PyPDFLoader(pdf_path)
    csv_loader = CSVLoader(file_path=csv_path)

    pdf_documents = pdf_loader.load()
    csv_documents = csv_loader.load()

    return pdf_documents + csv_documents

def create_vector_store(documents, embeddings):
    persist_directory = os.path.join(os.path.dirname(__file__), '../chroma_db_combined')
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def create_qa_chain(chat_model, documents, embeddings):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create vector store
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

    return RetrievalQA.from_chain_type(
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
def get_qa_response(qa_chain, question):
    """Gets a response from the QA chain with retry logic."""
    response = qa_chain.invoke({"query": question})
    return response.get('result', 'I could not find an answer to your question.')