import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, CombinedMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

# --- API Key Setup ---
try:
    # Try Streamlit secrets first (for deployment)
    API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
except (AttributeError, FileNotFoundError):
    # Fall back to environment variables (for local development)
    API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Don't stop import if API key is missing - let the individual functions handle it
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

# --- RAG Helper Functions ---
@st.cache_resource
def get_vector_store(file_paths):
    """Creates a vector store from uploaded files."""
    all_docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        loader = PyPDFLoader(path) if ext == '.pdf' else TextLoader(path)
        all_docs.extend(loader.load())
    
    if not all_docs: return None
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Increased from 1000
        chunk_overlap=400,  # Increased from 200
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embeddings)

def get_qa_prompt():
    """Creates a prompt template for document QA."""
    template = """Based on the following context and conversation history, provide a detailed and accurate response to the question. If the context doesn't contain enough information to fully answer the question, say so.

Context: {context}

Chat History: {chat_history}
Current Question: {question}

Response:"""
    
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

@st.cache_resource
def get_conversational_chain(_vector_store):
    """Initializes the conversational chain with advanced memory."""
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        input_key="question"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 8}
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": get_qa_prompt()
        },
        verbose=True
    )

def generate_summary(docs):
    """Generate a focused summary of the uploaded documents using the LLM."""
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.5)
    # Use first 3 chunks instead of 4 for a more focused summary
    text = "\n".join([doc.page_content for doc in docs[:3]])
    prompt = (
        "Provide a clear and focused summary of the following document content. "
        "Include:\n"
        "1. Main topic and key themes\n"
        "2. Important concepts\n"
        "3. Key findings or conclusions\n"
        "4. Practical implications\n\n"
        "Keep the summary concise while highlighting the most important points. "
        "Use bullet points where appropriate.\n\n"
        "CONTENT:\n" + text
    )
    return llm.invoke(prompt).content

def run_app():
    """Runs the Streamlit UI for the RAG Chatbot."""
    st.title("📄 RAG Chatbot: Chat with your Documents")
    st.markdown("Upload your PDF or TXT documents, ask questions, and get answers based on their content.")

    if not API_KEY:
        st.error("🔑 API Key is not configured. Please set your Gemini API key in Streamlit secrets (GEMINI_API_KEY or GOOGLE_API_KEY) for deployment, or in environment variables for local development.")
        st.info("💡 **For Streamlit Cloud:** Add your API key in the app settings under 'Secrets management'")
        return

    # --- Session State for RAG ---
    if "rag_conversation" not in st.session_state:
        st.session_state.rag_conversation = None
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []
    if "rag_summary" not in st.session_state:
        st.session_state.rag_summary = None
    if "rag_doc_context" not in st.session_state:
        st.session_state.rag_doc_context = None

    # --- Sidebar for Document Upload ---
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )

        if st.button("Process Documents", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    temp_dir = "temp_docs"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_paths = []
                    for f in uploaded_files:
                        path = os.path.join(temp_dir, f.name)
                        with open(path, "wb") as file:
                            file.write(f.getbuffer())
                        temp_paths.append(path)
                    vector_store = get_vector_store(temp_paths)
                    if vector_store:
                        # Generate and store summary
                        docs = []
                        for path in temp_paths:
                            ext = os.path.splitext(path)[1].lower()
                            loader = PyPDFLoader(path) if ext == '.pdf' else TextLoader(path)
                            docs.extend(loader.load())
                        summary = generate_summary(docs)
                        st.session_state.rag_summary = summary
                        st.session_state.rag_conversation = get_conversational_chain(vector_store)
                        st.success("Documents processed!")
                    else:
                        st.error("Failed to process documents.")
            else:
                st.warning("Please upload at least one file.")

    # --- Main Chat Interface ---
    if st.session_state.rag_conversation:
        # Show summary if available
        if st.session_state.rag_summary:
            st.subheader(":bookmark_tabs: Document Summary")
            st.markdown(st.session_state.rag_summary)
        for message in st.session_state.rag_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if user_question := st.chat_input("Ask a question about your documents..."):
            st.session_state.rag_chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.spinner("Thinking..."):
                # Process the user question
                response = st.session_state.rag_conversation({"question": user_question})
                ai_response = response["answer"]
                
                # Add source information if available
                if response.get("source_documents"):
                    sources = set()
                    for doc in response["source_documents"]:
                        if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                            sources.add(os.path.basename(doc.metadata['source']))
                    
                    if sources:
                        ai_response += "\n\n---\n**Sources:** " + ", ".join(sources)
                
                st.session_state.rag_chat_history.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                    
                    # Show confidence and relevance metrics
                    with st.expander("Response Details"):
                        st.info("This response was generated using multiple relevant passages from your documents. "
                              "The most relevant sections were selected using semantic search and maximum marginal relevance "
                              "to ensure comprehensive and accurate information.")
    else:
        st.info("Please upload and process documents in the sidebar to start chatting.")