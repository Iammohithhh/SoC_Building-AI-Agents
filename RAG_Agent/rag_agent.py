import streamlit as st
import os
import tempfile
import logging
from typing import Optional
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Chat with Your PDF Document")
st.markdown("Upload a PDF and ask questions about its content!")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key check
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        st.success("✅ GROQ API Key found")
    else:
        st.error("❌ GROQ API Key not found")
        st.markdown("Please add your GROQ API key to a `.env` file:")
        st.code("GROQ_API_KEY=your_api_key_here")
        st.stop()
    
    # Model selection
    model_choice = st.selectbox(
        "Choose Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index=0
    )
    
    # Advanced settings
    st.subheader("Advanced Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200)
    k_results = st.slider("Retrieval Results", 3, 10, 5)

def initialize_embeddings():
    """Initialize embeddings model with error handling"""
    try:
        # Try different embedding models in order of preference
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        
        for model_name in embedding_models:
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                # Test the embedding
                test_text = "This is a test sentence."
                embeddings.embed_query(test_text)
                st.success(f"✅ Using embedding model: {model_name}")
                return embeddings
            except Exception as model_error:
                st.warning(f"⚠️ Failed to load {model_name}: {str(model_error)}")
                continue
        
        # If all sentence-transformers fail, use OpenAI-compatible embeddings
        st.warning("⚠️ Falling back to simple TF-IDF embeddings")
        return initialize_simple_embeddings()
        
    except Exception as e:
        st.error(f"❌ Error initializing embeddings: {str(e)}")
        return None

def initialize_simple_embeddings():
    """Fallback to simple TF-IDF embeddings if transformers fail"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        class SimpleTFIDFEmbeddings:
            def __init__(self):
                self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
                self.is_fitted = False
                
            def embed_documents(self, texts):
                if not self.is_fitted:
                    self.vectorizer.fit(texts)
                    self.is_fitted = True
                vectors = self.vectorizer.transform(texts).toarray()
                return vectors.tolist()
                
            def embed_query(self, text):
                if not self.is_fitted:
                    # If not fitted, return zero vector
                    return [0.0] * 384
                vector = self.vectorizer.transform([text]).toarray()[0]
                return vector.tolist()
        
        return SimpleTFIDFEmbeddings()
        
    except Exception as e:
        st.error(f"❌ Error with fallback embeddings: {str(e)}")
        return None

def process_pdf(uploaded_file, chunk_size, chunk_overlap):
    """Process uploaded PDF with comprehensive error handling"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name
        
        # Load PDF
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            if not docs:
                st.error("❌ No content found in PDF")
                return None, None
            
            # Check if PDF has readable content
            total_content = "".join([doc.page_content for doc in docs])
            if len(total_content.strip()) < 50:
                st.error("❌ PDF appears to be empty or contains only images")
                return None, None
                
        except Exception as e:
            st.error(f"❌ Error loading PDF: {str(e)}")
            return None, None
        
        # Split documents
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            split_docs = text_splitter.split_documents(docs)
            
            if not split_docs:
                st.error("❌ No text chunks created from PDF")
                return None, None
                
        except Exception as e:
            st.error(f"❌ Error splitting documents: {str(e)}")
            return None, None
        
        # Clean up temporary file
        try:
            os.unlink(pdf_path)
        except:
            pass
        
        return docs, split_docs
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return None, None

def create_vectorstore(split_docs, embeddings):
    """Create vectorstore with error handling"""
    try:
        # Create temporary directory for Chroma
        persist_directory = tempfile.mkdtemp()
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_results}
        )
        
        return vectorstore, retriever, persist_directory
        
    except Exception as e:
        st.error(f"❌ Error creating vectorstore: {str(e)}")
        return None, None, None

def query_llm(question, context, model_choice):
    """Query Groq LLM with context"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context from a PDF document.

Context from PDF:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the answer is not in the context, say "I cannot find this information in the provided document"
- Be specific and cite relevant parts of the document
- Keep your answer concise but comprehensive

Answer:"""
        
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"❌ Error querying LLM: {str(e)}")
        return f"Error generating response: {str(e)}"

def main():
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.processed_file = None
        st.session_state.persist_directory = None
        st.session_state.chat_history = []
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your PDF document",
        type=["pdf"],
        help="Upload a PDF file to start asking questions about its content"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        if st.session_state.processed_file != uploaded_file.name:
            # Clear previous data
            if st.session_state.persist_directory and os.path.exists(st.session_state.persist_directory):
                try:
                    shutil.rmtree(st.session_state.persist_directory)
                except:
                    pass
            
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.chat_history = []
            
            # Process new file
            with st.spinner("🔄 Processing PDF... This may take a moment"):
                # Initialize embeddings
                embeddings = initialize_embeddings()
                if embeddings is None:
                    st.stop()
                
                # Process PDF
                docs, split_docs = process_pdf(uploaded_file, chunk_size, chunk_overlap)
                if docs is None or split_docs is None:
                    st.stop()
                
                # Create vectorstore
                vectorstore, retriever, persist_directory = create_vectorstore(split_docs, embeddings)
                if vectorstore is None:
                    st.stop()
                
                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.processed_file = uploaded_file.name
                st.session_state.persist_directory = persist_directory
                
                # Display success message with document stats
                st.success(f"✅ PDF processed successfully!")
                
                # Display document statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", len(docs))
                with col2:
                    st.metric("Text Chunks", len(split_docs))
                with col3:
                    total_chars = sum(len(doc.page_content) for doc in docs)
                    st.metric("Characters", f"{total_chars:,}")
        
        # Chat interface
        if st.session_state.retriever is not None:
            st.subheader("💬 Ask Questions About Your Document")
            
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {q[:50]}..."):
                        st.write(f"**Question:** {q}")
                        st.write(f"**Answer:** {a}")
            
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What is this document about?",
                key="question_input"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🔍 Ask", type="primary")
            with col2:
                clear_button = st.button("🗑️ Clear History")
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
            
            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Retrieve relevant documents
                        relevant_docs = st.session_state.retriever.invoke(question)
                        
                        if not relevant_docs:
                            st.warning("⚠️ No relevant information found in the document.")
                        else:
                            # Combine context
                            context = "\n\n".join([doc.page_content for doc in relevant_docs])
                            
                            # Query LLM
                            answer = query_llm(question, context, model_choice)
                            
                            # Display answer
                            st.subheader("Answer:")
                            st.write(answer)
                            
                            # Add to chat history
                            st.session_state.chat_history.append((question, answer))
                            
                            # Show source chunks
                            with st.expander("📄 Source Chunks"):
                                for i, doc in enumerate(relevant_docs):
                                    st.write(f"**Chunk {i+1}:**")
                                    st.write(doc.page_content[:300] + "...")
                                    st.write("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
                        logger.error(f"Error processing question: {str(e)}")
    
    else:
        st.info("👆 Please upload a PDF document to get started")
        
        # Show example questions
        st.subheader("💡 Example Questions You Can Ask:")
        examples = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key findings?",
            "Who are the main people mentioned?",
            "What dates are mentioned in the document?",
            "What are the conclusions or recommendations?"
        ]
        
        for example in examples:
            st.write(f"• {example}")

if __name__ == "__main__":
    main()