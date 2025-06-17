import streamlit as st
import course_generator
import rag_chatbot
import document_course_creator # Import the new module

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Learning Suite",
    page_icon="🚀",
    layout="wide"
)

# --- Session State Initialization (Centralized) ---
# Initialize session state for the entire application,
# ensuring 'courses' and 'selected_course_index' are available globally.
def init_global_session_state():
    """Initialize session state variables for all components."""
    # Course Generator state
    if "courses" not in st.session_state:
        st.session_state.courses = []
    if "selected_course_index" not in st.session_state:
        st.session_state.selected_course_index = None
    if "chapter_contents" not in st.session_state:
        st.session_state.chapter_contents = {}
    if "quiz_progress" not in st.session_state:
        st.session_state.quiz_progress = {}
    
    # Document Course Creator state
    if "doc_courses" not in st.session_state:
        st.session_state.doc_courses = []
    if "selected_doc_course" not in st.session_state:
        st.session_state.selected_doc_course = None
    if "doc_chat_history" not in st.session_state:
        st.session_state.doc_chat_history = []
    if "document_content" not in st.session_state: # Ensure document_content is initialized here
        st.session_state.document_content = None
    
    # RAG chatbot state
    if "rag_conversation" not in st.session_state:
        st.session_state.rag_conversation = None
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "course_generator"


init_global_session_state()

# --- Main App Navigation ---
st.sidebar.title("Navigation")

# Choose Application Mode
st.sidebar.subheader("Choose Application Mode")

# Create a container for application mode selection
mode_container = st.sidebar.container()

# Better visual separation between modes
if mode_container.button(
    "Course Generator & Quizzes",
    type="primary" if st.session_state.current_mode == "course_generator" else "secondary",
    use_container_width=True,
    key="btn_course_gen"
):
    st.session_state.current_mode = "course_generator"
    st.rerun()

if mode_container.button(
    "Chat with Your Documents (RAG)",
    type="primary" if st.session_state.current_mode == "rag_chat" else "secondary",
    use_container_width=True,
    key="btn_rag"
):
    st.session_state.current_mode = "rag_chat"
    st.rerun()

if mode_container.button(
    "Course Creation from Documents",
    type="primary" if st.session_state.current_mode == "doc_creator" else "secondary",
    use_container_width=True,
    key="btn_doc_create"
):
    st.session_state.current_mode = "doc_creator"
    st.rerun()

st.sidebar.divider()

# Show the relevant courses list based on mode
if st.session_state.current_mode == "course_generator":
    if st.session_state.courses:
        st.sidebar.subheader("My Generated Courses")
        for idx, course in enumerate(st.session_state.courses):
            # Use 'courseTitle' from document_course_creator schema, or 'title' from course_generator
            course_title = course.get('courseTitle', course.get('title', f'Course {idx + 1}')).strip()
            if not course_title:  # Skip empty courses
                continue
            if st.sidebar.button(
                f"📚 {course_title}",
                key=f"gen_course_select_{idx}", # Unique key for generator courses
                use_container_width=True,
                type="primary" if st.session_state.selected_course_index == idx else "secondary"
            ):
                if st.session_state.selected_course_index != idx:
                    st.session_state.selected_course_index = idx
                    st.rerun()
    else:
        st.sidebar.info("No AI-generated courses yet.")
elif st.session_state.current_mode == "doc_creator":
    if st.session_state.doc_courses: # Use doc_courses for document creator
        st.sidebar.subheader("My Document Courses")
        for idx, course in enumerate(st.session_state.doc_courses): # Iterate doc_courses
            doc_course_title = course.get('courseTitle', f'Document Course {idx + 1}').strip()
            if not doc_course_title:
                continue
            if st.sidebar.button(
                f"📄 {doc_course_title}",
                key=f"doc_course_select_{idx}", # Unique key for document courses
                use_container_width=True,
                type="primary" if st.session_state.selected_doc_course == idx else "secondary" # Use selected_doc_course
            ):
                if st.session_state.selected_doc_course != idx:
                    st.session_state.selected_doc_course = idx
                    st.rerun()
    else:
        st.sidebar.info("No document-generated courses yet. Upload documents to create one!")


# Visual feedback for current mode
mode_messages = {
    "course_generator": "You are in the Course Generator mode.",
    "rag_chat": "You are in the RAG Chatbot mode.",
    "doc_creator": "You are in the Document Course Creator mode."
}
st.sidebar.info(mode_messages.get(st.session_state.current_mode, ""))

# --- App Routing ---
if st.session_state.current_mode == "course_generator":
    course_generator.run_app()
elif st.session_state.current_mode == "rag_chat":
    rag_chatbot.run_app()
elif st.session_state.current_mode == "doc_creator":
    document_course_creator.run_app()

