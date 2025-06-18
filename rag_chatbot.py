import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, CombinedMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --- API Key Setup ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource
def get_conversational_chain(_vector_store, doc_context=None):
    """Initializes the conversational chain with advanced memory."""
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.7)
    chat_memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # Store document context in memory if provided
    if doc_context:
        doc_memory = ConversationBufferMemory(memory_key='doc_context', return_messages=True)
        doc_memory.save_context({'input': ''}, {'output': doc_context})
        memory = CombinedMemory(memories=[chat_memory, doc_memory])
    else:
        memory = chat_memory
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=_vector_store.as_retriever(), memory=memory
    )

def generate_summary(docs):
    """Generate a concise summary of the uploaded documents using the LLM."""
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.5)
    # Use only the first 2 chunks for a shorter summary
    text = "\n".join([doc.page_content for doc in docs[:2]])
    prompt = (
        "You are an expert assistant. Read the following document content and provide a concise summary "
        "highlighting only the main topics and most important points. Limit the summary to 5-7 sentences.\n\nCONTENT:\n" + text
    )
    return llm.invoke(prompt).content

def run_app():
    """Runs the Streamlit UI for the RAG Chatbot."""
    st.title("📄 RAG Chatbot: Chat with your Documents")
    st.markdown("Upload your PDF or TXT documents, ask questions, and get answers based on their content.")

    if not API_KEY:
        st.error("API Key is not configured. Please set it in Streamlit secrets.")
        st.stop()

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
                        st.session_state.rag_doc_context = summary  # Use summary as doc context
                        st.session_state.rag_conversation = get_conversational_chain(vector_store, doc_context=summary)
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
                # Pass document context as part of the question for richer memory
                question_with_context = user_question
                if st.session_state.rag_doc_context:
                    question_with_context += f"\n\n[Document Context]\n{st.session_state.rag_doc_context}"
                response = st.session_state.rag_conversation({'question': question_with_context})
                ai_response = response['chat_history'][-1].content
                st.session_state.rag_chat_history.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
    else:
        st.info("Please upload and process documents in the sidebar to start chatting.")
