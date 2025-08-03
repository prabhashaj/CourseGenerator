import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Learning Suite",
    page_icon="🚀",
    layout="wide"
)

import course_generator
import rag_chatbot
import document_course_creator
import play_zone  # Import the new PlayZone module
import lang_news_enhanced # Import the enhanced langchain news module

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
    if "current_card_idx" not in st.session_state:
        st.session_state.current_card_idx = 0
    
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

if mode_container.button(
    "🎮 PlayZone",
    type="primary" if st.session_state.current_mode == "play_zone" else "secondary",
    use_container_width=True,
    key="btn_play_zone"
):
    st.session_state.current_mode = "play_zone"
    st.rerun()

if mode_container.button(
    "🤖 Knowledge Hub (Enhanced)",
    type="primary" if st.session_state.current_mode == "lang_news" else "secondary",
    use_container_width=True,
    key="btn_lang_news"
):
    st.session_state.current_mode = "lang_news"
    st.rerun()

st.sidebar.divider()

# --- App Routing ---
try:
    # Run the selected application mode
    mode_handlers = {
        "course_generator": course_generator.run_app,
        "rag_chat": rag_chatbot.run_app,
        "doc_creator": document_course_creator.run_app,
        "play_zone": play_zone.show_play_zone,
        "lang_news": lang_news_enhanced.main, # Add enhanced langchain news handler
    }
    
    # Get and execute the appropriate handler
    handler = mode_handlers.get(st.session_state.current_mode)
    if handler:
        handler()
    else:
        st.error("Invalid application mode selected.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please refresh the page and try again.")

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
elif st.session_state.current_mode == "lang_news":
    st.sidebar.subheader("🤖 Knowledge Hub (Enhanced)")
    st.sidebar.info("Dynamic AI agent with step-by-step reasoning! Ask questions about tech, news, or current events!")

# Visual feedback for current mode
mode_messages = {
    "course_generator": "You are in the Course Generator mode.",
    "rag_chat": "You are in the RAG Chatbot mode.",
    "doc_creator": "You are in the Document Course Creator mode.",
    "play_zone": "You are in the PlayZone mode.",
    "lang_news": "You are in the Enhanced Knowledge Hub mode. Watch the AI agent think and reason step-by-step!"
}
st.sidebar.info(mode_messages.get(st.session_state.current_mode, ""))

