import streamlit as st
import json
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate

# --- Configuration ---
st.set_page_config(
    page_title="Smart Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

# --- Session State Initialization ---
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}
if "current_course" not in st.session_state:
    st.session_state.current_course = None
if "courses" not in st.session_state:
    st.session_state.courses = []
if "selected_course_index" not in st.session_state:
    st.session_state.selected_course_index = None

# --- LangChain Components ---
@st.cache_resource
def get_llm():
    """Initialize Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.7
    )

@st.cache_resource
def get_embeddings():
    """Initialize Gemini Embeddings."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

def process_documents(files, course_name):
    """Process documents using LangChain orchestration."""
    temp_dir = "temp_docs"
    os.makedirs(temp_dir, exist_ok=True)
    all_documents = []
    document_names = []

    try:
        # Save and process files
        for uploaded_file in files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load documents using appropriate loader
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                continue

            all_documents.extend(loader.load())
            document_names.append(uploaded_file.name)
            os.remove(file_path)  # Clean up

        if all_documents:
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(all_documents)

            # Create vector store
            vector_store = FAISS.from_documents(
                chunks,
                get_embeddings()
            )

            # Store in session state
            st.session_state.vector_stores[course_name] = vector_store
            
            # Update course documents list
            for idx, course in enumerate(st.session_state.courses):
                if course['title'] == course_name:
                    course['documents'].extend(document_names)
                    st.session_state.courses[idx] = course
                    break
            
            return True

    except Exception as e:
        st.error(f"Error processing documents: {e}")
    finally:
        # Cleanup
        try:
            os.rmdir(temp_dir)
        except:
            pass

    return False

def setup_conversation_chain(vector_store):
    """Setup LangChain conversation chain."""
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # Custom prompt for better context handling
    qa_prompt = PromptTemplate(
        template="""You are a knowledgeable learning assistant. Use the following context to answer the question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: Let me help you with that.""",
        input_variables=["context", "question"]
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return conversation_chain

def handle_chat(user_input, course_name):
    """Handle chat interactions using LangChain."""
    if course_name not in st.session_state.vector_stores:
        st.error("Please process some documents first!")
        return

    if not st.session_state.conversation:
        st.session_state.conversation = setup_conversation_chain(
            st.session_state.vector_stores[course_name]
        )

    try:
        response = st.session_state.conversation({
            "question": user_input
        })
        st.session_state.chat_history = response['chat_history']

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg.type == 'human':
                st.chat_message("user").write(msg.content)
            else:
                st.chat_message("assistant").write(msg.content)

    except Exception as e:
        st.error(f"Error generating response: {e}")

def main():
    st.title("🎓 Smart Learning Assistant")

    # Sidebar for course management
    with st.sidebar:
        st.header("📚 Course Management")
        new_course_name = st.text_input(
            "Create New Course",
            placeholder="e.g., Python Programming, Machine Learning, etc."
        )
        if st.button("Add Course"):
            if new_course_name:
                new_course = {
                    'title': new_course_name,
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'documents': []
                }
                st.session_state.courses.append(new_course)
                st.success(f"Course '{new_course_name}' created!")
            else:
                st.warning("Please enter a course name.")

        st.markdown("---")
        st.subheader("📖 My Courses")
        if not st.session_state.courses:
            st.info("No courses yet. Create one above!")
        else:
            for idx, course in enumerate(st.session_state.courses):
                if st.button(f"📘 {course['title']}", key=f"course_{idx}"):
                    st.session_state.current_course = course['title']
                    st.session_state.selected_course_index = idx

    # Main content area
    if st.session_state.current_course:
        st.subheader(f"📚 Course: {st.session_state.current_course}")
        
        # Show uploaded documents
        current_course = st.session_state.courses[st.session_state.selected_course_index]
        if current_course['documents']:
            with st.expander("📑 Uploaded Documents"):
                for doc in current_course['documents']:
                    st.write(f"- {doc}")
        
        # Document upload section
        uploaded_files = st.file_uploader(
            "Upload Additional Study Materials",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    if process_documents(uploaded_files, st.session_state.current_course):
                        st.success("Documents processed successfully!")
                    else:
                        st.error("Failed to process documents. Please try again.")

        # Chat interface
        st.subheader("💬 Learning Assistant")
        user_input = st.chat_input("Ask anything about your materials...")
        if user_input:
            handle_chat(user_input, st.session_state.current_course)

        # Clear chat option
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.experimental_rerun()
    else:
        st.info("👈 Please create or select a course from the sidebar to begin!")

if __name__ == "__main__":
    main()
