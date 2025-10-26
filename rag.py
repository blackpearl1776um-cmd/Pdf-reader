import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from htmlTemplates import css, bot_template, user_template
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split text into chunks for processing"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """Create vector store from text chunks using embeddings"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Create conversational retrieval chain with memory - using LOCAL model"""
    
    with st.spinner("🔄 Loading AI model locally... (first time may take 1-2 minutes)"):
        # Load model locally - no API needed!
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create text generation pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )
        
        # Create LLM from pipeline
        llm = HuggingFacePipeline(pipeline=pipe)

    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    # Create conversational retrieval chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    """Handle user input and display conversation"""
    if st.session_state.conversation is None:
        st.warning("⚠️ Please upload and process PDFs first!")
        return
    
    try:
        with st.spinner("🤔 Thinking..."):
            # Get response from conversation chain
            response = st.session_state.conversation.invoke({'question': user_question})
            st.session_state.chat_history = response['chat_history']

        # Display conversation history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def main():
    """Main application function"""
    load_dotenv()
    
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main header
    st.header("Chat with multiple PDFs :books:")
    st.info("💡 This app runs AI completely locally on your computer - no API needed!")
    
    # User input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.error("❌ Please upload at least one PDF file!")
            else:
                with st.spinner("Processing your documents... ⏳"):
                    try:
                        # Extract text from PDFs
                        raw_text = get_pdf_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text found in the uploaded PDFs!")
                            return

                        # Split text into chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.info(f"📄 Created {len(text_chunks)} text chunks")

                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # Create conversation chain (downloads model first time)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        
                        st.success("✅ PDFs processed successfully! You can now ask questions.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())


if __name__ == '__main__':
    main()