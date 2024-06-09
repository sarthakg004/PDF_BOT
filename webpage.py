import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import cv2
import pytesseract
import tempfile
import numpy as np
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_core.messages import AIMessage, HumanMessage
from htmlTemplates import css, bot_template, user_template

# Load environment variables from .env file
load_dotenv()

# Set Tesseract OCR path for extracting text from images
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_pdf_text(pdf_docs, zoom_x=2.0, zoom_y=2.0):
    """
    Extract text from a list of PDF documents using OCR.

    Args:
        pdf_docs (list): List of uploaded PDF files.
        zoom_x (float): Zoom factor for x-axis.
        zoom_y (float): Zoom factor for y-axis.

    Returns:
        str: Extracted text from all PDF documents.
    """
    text = ''
    for pdf_data in pdf_docs:
        try:
            # Save PDF data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf_data.getvalue())
                tmp_file_path = tmp_file.name

            # Open the PDF document
            pdf_document = fitz.open(tmp_file_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # Zoom to enhance OCR accuracy
                matrix = fitz.Matrix(zoom_x, zoom_y)
                pix = page.get_pixmap(matrix=matrix)

                # Convert page to image
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Threshold to binary image for better OCR
                _, thresh = cv2.threshold(gray, 210, 230, cv2.THRESH_BINARY)
                # Extract text using Tesseract OCR
                ocr_text = pytesseract.image_to_string(thresh)

                text += ocr_text
            pdf_document.close()
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
    return text

def get_text_chunks(text):
    """
    Split text into chunks for efficient processing.

    Args:
        text (str): The raw text to be split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """
    Create a vector store from text chunks using embeddings.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        FAISS: A FAISS vector store.
    """
    # model_name = 'Alibaba-NLP/gte-large-en-v1.5'
    model_name = 'intfloat/multilingual-e5-large-instruct'
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    """
    Create a conversational retrieval chain for Q&A.

    Args:
        vectorstore (FAISS): The vector store for retrieval.

    Returns:
        ConversationalRetrievalChain: A conversation chain for Q&A.
    """
    llm = HuggingFaceHub(repo_id='meta-llama/Meta-Llama-3-8B-Instruct', model_kwargs={"temperature": 0.5, 
                                                                                        "max_length": 512
                                                                                        })
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_userinput(user_question):
    """
    Handle user input and generate responses using the conversation chain.

    Args:
        user_question (str): The user's question.

    Returns:
        None
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Extract the final answer from the conversation chain response
    final_answer = response['chat_history'][-1].content if response['chat_history'] else "No response"

    # Remove instructions and retain only the final answer
    if "Helpful Answer:" in final_answer:
        final_answer = final_answer.split("Helpful Answer:")[-1].strip()

    # Display the final answer
    st.write(bot_template.replace("{{MSG}}", final_answer), unsafe_allow_html=True)

def main():
    """
    Main function to run the Streamlit app.
    """
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your PDF :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract text from PDFs
                raw_text = get_pdf_text(pdf_docs)

                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store from text chunks
                vectorstore = get_vectorstore(text_chunks)

                # Create a conversation chain for Q&A
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()


