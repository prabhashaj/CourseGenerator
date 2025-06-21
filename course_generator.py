from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import streamlit as st
import httpx
import json
import os
import importlib
import asyncio  # Import asyncio for event loop management
from datetime import datetime  # Import datetime for timestamping courses

# --- Utility Import ---
try:
    quiz_utils = importlib.import_module("quiz_utils")
except ImportError:
    st.error("The 'quiz_utils.py' file was not found. Please make sure it's in the same directory.")
    st.stop()

# --- API Key Setup ---
# Use Streamlit secrets for deployment and fall back to environment variable
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not API_KEY:
    st.error("API Key is not configured. Please set it in Streamlit secrets or environment variables (GEMINI_API_KEY or GOOGLE_API_KEY).")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY # Ensure it's in os.environ for other modules that might use it

# --- Session State Initialization ---
def init_session_state():
    """Initializes session state for the course generator."""
    if "courses" not in st.session_state:
        st.session_state.courses = []
    if "selected_course_index" not in st.session_state:
        st.session_state.selected_course_index = None
    if "chapter_contents" not in st.session_state:
        st.session_state.chapter_contents = {}
    if "quiz_progress" not in st.session_state:
        st.session_state.quiz_progress = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation" not in st.session_state:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        memory = ConversationBufferMemory()
        st.session_state.conversation = ConversationChain(llm=llm, memory=memory)
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "course_generator"
    if "previous_course_index" not in st.session_state:
        st.session_state.previous_course_index = None
    # Initialize AI settings if not present
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1024
    if "top_k" not in st.session_state: # Ensure top_k and top_p are initialized
        st.session_state.top_k = 32
    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0
    # Add a flag to ensure sidebar content is rendered only once
    if "sidebar_rendered" not in st.session_state:
        st.session_state.sidebar_rendered = False


def clear_chat_history():
    """Clear chat history when switching courses or generating a new one"""
    if "chat_history" in st.session_state:
        st.session_state.chat_history = []
    if "conversation" in st.session_state:
        st.session_state.conversation.memory.clear()
    # Reset sidebar_rendered flag when clearing chat history (e.g., switching modes)
    # This allows the sidebar to redraw its dynamic content if necessary
    st.session_state.sidebar_rendered = False


def show_navigation():
    """Display the navigation menu and AI settings in the sidebar."""
    with st.sidebar:
    #     # Check if sidebar content has already been rendered in this session
    #     # This prevents duplicate rendering on reruns unless explicitly reset (e.g., by clear_chat_history)
    #     if st.session_state.sidebar_rendered:
    #         return 

    #     st.title("Navigation")
    #     st.subheader("Choose Application Mode")
    #     # Re-added all intended modes as per user clarification
    #     modes = {
    #         "course_generator": "Course Generator & Quizzes",
    #         "rag_chat": "Chat with Your Documents (RAG)",
    #         "doc_creator": "Course Creation from Documents"
    #     }
        
    #     for mode, label in modes.items():
    #         if st.button(
    #             label,
    #             use_container_width=True,
    #             type="primary" if st.session_state.current_mode == mode else "secondary"
    #         ):
    #             if st.session_state.current_mode != mode:
    #                 clear_chat_history() # This will now also reset sidebar_rendered
    #                 st.session_state.current_mode = mode
    #                 st.rerun()

    #     st.markdown("---") # Separator between navigation and AI settings

        # AI Generation Settings in sidebar (only temperature and max_tokens)
        st.header("⚙️ AI Generation Settings")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature, # Use session state value
            step=0.01,
            help="Controls the randomness of the output. Higher values mean more creative."
        )
        st.session_state.max_tokens = st.slider(
            "Max Output Tokens",
            min_value=50,
            max_value=2048,
            value=st.session_state.max_tokens, # Use session state value
            step=50,
            help="Maximum number of tokens to generate in the response."
        )
        # top_k and top_p are not shown but still used in the API call, so keep their defaults in session_state.

        st.markdown("---") # Separator between AI settings and saved courses

        # Show saved courses section (at the very end of sidebar)
        if st.session_state.courses:
            st.subheader("My Saved Courses")
            for idx, course in enumerate(st.session_state.courses):
                # Use a consistent key for course selection buttons
                if st.button(
                    f"📚 {course.get('courseTitle', 'Untitled Course')} ({course.get('created_at', 'N/A').split(' ')[0]})",
                    key=f"sidebar_course_{idx}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_course_index == idx else "secondary"
                ):
                    if st.session_state.selected_course_index != idx:
                        clear_chat_history() # Clear chat history when switching courses
                        st.session_state.selected_course_index = idx
                        st.rerun()
        else:
            st.info("No courses generated yet. Generate one above!")
        
        # Set the flag to True after rendering the sidebar content
        st.session_state.sidebar_rendered = True

# --- Helper function to get or create an asyncio loop ---
def get_or_create_eventloop():
    """Gets the running event loop or creates a new one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def safe_json_parse(text_response):
    """Safely parse JSON, attempting to repair common issues."""
    import re
    # Remove trailing commas before } or ]
    text_response = re.sub(r',([\s\n]*[}}\]])', r'\1', text_response)
    # Remove any markdown code block markers
    text_response = re.sub(r'^```json|```$', '', text_response, flags=re.MULTILINE)
    # Truncate to last closing curly brace if needed (for incomplete JSON)
    last_brace = text_response.rfind('}')
    if last_brace != -1:
        text_response = text_response[:last_brace+1]
    
    try:
        return json.loads(text_response)
    except json.JSONDecodeError as e:
        st.warning(f"JSONDecodeError: {e}. Attempting to fix...")
        # A more robust repair could be implemented here if needed.
        # For now, just return with an error flag if it fails after basic cleaning.
        return {"error": f"Invalid JSON after cleaning: {str(e)}", "raw": text_response}

# --- Gemini API Call Function ---
async def generate_content_with_gemini(prompt, temperature, max_tokens, top_k, top_p, response_schema=None):
    """
    Calls the Gemini API to generate content with specified parameters and an optional schema.
    """
    if not API_KEY:
        st.error("API Key is not configured. Cannot call Gemini API.")
        return None

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topK": int(top_k), # Ensure integer
        "topP": top_p
    }
    payload = {"contents": chat_history, "generationConfig": generation_config}

    if response_schema:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = response_schema

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=120 # Increased timeout for potentially longer content generation
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("candidates"):
                st.error("No response generated by LLM. Please try again.")
                if result.get("promptFeedback") and result["promptFeedback"].get("blockReason"):
                    st.error(f"LLM blocked the prompt due to: {result['promptFeedback']['blockReason']}")
                return None

            text_response = result["candidates"][0]["content"]["parts"][0].get("text", "")

            if response_schema:
                parsed_response = safe_json_parse(text_response)
                if isinstance(parsed_response, dict) and "error" in parsed_response:
                    st.error(f"Failed to parse response as JSON: {parsed_response['error']}")
                    st.json(parsed_response["raw"]) # Show raw response for debugging
                    return None
                # Validate required fields if a schema was used and parsing was successful
                # (This can be more rigorous if needed, comparing parsed_response keys to schema's required)
                return parsed_response
            return text_response
    except httpx.RequestError as e:
        st.error(f"API request error: {str(e)}. Check your internet connection or API key.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {str(e)}")
        return None

# --- Course Assistant Chat Function ---
def process_chat_message(user_message, course_content):
    """Process chat messages with context about the course"""
    if not user_message:
        return ""

    try:
        # Prepare context about the course
        context = f"""You are a helpful course assistant. Use the following course content to answer questions:
        {course_content}
        
        Only answer questions directly related to this course content. If the question is not related,
        politely redirect the user to ask course-related questions.
        If you encounter any errors or cannot answer, provide a clear explanation.
        
        User question: {user_message}
        """
        
        # Use the pre-initialized ConversationChain for context-aware chat
        response = st.session_state.conversation.predict(input=context)
        st.session_state.chat_history.append({"user": user_message, "assistant": response})
        return response
    except Exception as e:
        error_message = f"I apologize, but I encountered an error while processing your chat: {str(e)}"
        st.error(error_message)
        st.session_state.chat_history.append({"user": user_message, "assistant": error_message})
        return error_message

def run_app():
    """Runs the course generator application."""
    init_session_state()
    show_navigation() # Display navigation in sidebar

    st.title("🎓 Course Generator & Tracker")
    # Add friendly message about sidebar features
    st.info("👈 Open the sidebar to explore exciting features")
    st.markdown("Generate custom course outlines, track your progress, and get detailed chapter content!")
    
    # The AI Generation Settings are now in the sidebar, so remove them from the main area.
    # The parameters are accessed from st.session_state because they are updated in the sidebar.
    temperature = st.session_state.temperature
    max_tokens = st.session_state.max_tokens
    top_k = st.session_state.top_k 
    top_p = st.session_state.top_p 
    
    st.markdown("---") # Separator

    # Course generation form
    with st.form(key="new_course_form"):
        st.subheader("Generate New Course Outline")
        
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            course_topic = st.text_input(
                "Course Topic", 
                key="new_course_topic",
                help="Enter the main topic for your course",
                placeholder="e.g., Machine Learning"
            )
            difficulty = st.selectbox(
                "Difficulty Level",
                ["Beginner", "Intermediate", "Advanced"],
                key="new_course_difficulty",
                help="Select the target audience level"
            )
        with col_input2:
            num_modules = st.number_input(
                "Number of Modules", 
                min_value=1, 
                max_value=10,
                value=5, 
                step=1,
                key="new_course_modules",
                help="Choose how many modules you want"
            )
            read_time_per_module = st.radio(
                "Approx. Read Time per Module",
                ["2 minutes", "5 minutes", "10 minutes"],
                index=1,
                horizontal=True,
                help="Estimate the reading time for each module.",
                key="new_course_read_time_input"
            )
        
        generate_button = st.form_submit_button(
            "Generate New Course Outline",
            type="primary",
            use_container_width=True
        )
        
    # --- Course Generation Logic (Unified) ---
    if generate_button:
        if not course_topic.strip():
            st.warning("⚠️ Please enter a course topic!")
            return
        
        with st.spinner(f"🔄 Generating {difficulty} level course on {course_topic}..."):
            try:
                # Define the JSON schema for the expected course outline (Aligned with app (2).py structure)
                course_schema = {
                    "type": "OBJECT",
                    "properties": {
                        "courseTitle": {"type": "STRING", "description": "The title of the course."},
                        "introduction": {"type": "STRING", "description": "A brief introduction to the course."},
                        "modules": {
                            "type": "ARRAY",
                            "description": "A list of course modules.",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "moduleNumber": {"type": "INTEGER", "description": "The sequential number of the module."},
                                    "moduleTitle": {"type": "STRING", "description": "The title of the module."},
                                    "chapters": {
                                        "type": "ARRAY",
                                        "description": "A list of chapters within the module.",
                                        "items": {
                                            "type": "OBJECT",
                                            "properties": {
                                                "chapterTitle": {"type": "STRING", "description": "The title of the chapter."},
                                                "description": {"type": "STRING", "description": "A brief description of the chapter content."}
                                            },
                                            "required": ["chapterTitle", "description"]
                                        }
                                    }
                                },
                                "required": ["moduleNumber", "moduleTitle", "chapters"]
                            }
                        },
                        "conclusion": {"type": "STRING", "description": "A brief conclusion or next steps for the course."}
                    },
                    "required": ["courseTitle", "introduction", "modules", "conclusion"]
                }

                # Construct the prompt for the LLM
                course_prompt = f"""
                Generate a detailed course outline in JSON format for a '{course_topic}' course.
                The course should be designed for a '{difficulty}' level audience.
                It must have exactly {num_modules} modules.
                Each module should have chapters, and the content for each module should be designed to take approximately {read_time_per_module} to read.

                The JSON output should strictly follow this schema. Do not include any additional text outside the JSON.
                """

                loop = get_or_create_eventloop()
                course_data = loop.run_until_complete(
                    generate_content_with_gemini(
                        course_prompt,
                        temperature, # Use values from session state (updated by sidebar sliders)
                        max_tokens,  # Use values from session state (updated by sidebar sliders)
                        top_k,       # Use values from session state (updated by sidebar sliders)
                        top_p,       # Use values from session state (updated by sidebar sliders)
                        response_schema=course_schema
                    )
                )
                loop.close() # Close the loop after use

                if course_data and isinstance(course_data, dict) and course_data.get("courseTitle"):
                    # Initialize completion status for chapters
                    completion_status = {}
                    total_chapters = 0
                    
                    for m_idx, module in enumerate(course_data.get("modules", [])):
                        for c_idx, chapter in enumerate(module.get("chapters", [])):
                            # Use the same chapter_id logic as app (2).py for consistency
                            chapter_id = f"course_{len(st.session_state.courses)}_module_{module['moduleNumber']}_chapter_{chapter['chapterTitle'].replace(' ', '_').replace('.', '').replace(',', '')}"
                            completion_status[chapter_id] = False
                            total_chapters += 1

                    # Create new course with tracking data
                    new_course = {
                        **course_data,
                        "completion_status": completion_status,
                        "total_chapters": total_chapters,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.courses.append(new_course)
                    st.session_state.selected_course_index = len(st.session_state.courses) - 1 # Select the newly created course
                    clear_chat_history() # Clear chat history for the new course
                    st.success(f"✨ Successfully generated course: {new_course['courseTitle']}")
                    st.rerun()                
                else:
                    st.warning("Unable to generate the course with the current number of modules. Try reducing the number of modules and try again.")
            except Exception as e:
                st.warning("We encountered some difficulty generating the course with this many modules. Consider reducing the number of modules and try again.")
                # Log the actual error for debugging (not visible to users)
                print(f"Course generation error: {str(e)}")

    # --- Display selected course and chat interface ---
    if st.session_state.selected_course_index is not None and st.session_state.selected_course_index < len(st.session_state.courses):
        course = st.session_state.courses[st.session_state.selected_course_index]
        
        st.subheader(f"📚 Course: {course.get('courseTitle', 'Course Content')}")
        
        # Course completion tracking
        completed_chapters = sum(1 for status in course.get("completion_status", {}).values() if status)
        total_chapters = course.get("total_chapters", 0)
        completion_percentage = (completed_chapters / total_chapters) * 100 if total_chapters > 0 else 0
        st.markdown("**Course Progress:**")
        st.progress(completion_percentage / 100, text=f"{completion_percentage:.1f}% Completed ({completed_chapters}/{total_chapters} chapters)")
        
        # Course introduction
        st.markdown(f"**Introduction:** {course.get('introduction', '')}")
        
        # Modules and Chapters
        for m_idx, module in enumerate(course.get("modules", [])):
            st.markdown(f"### Module {module.get('moduleNumber', m_idx + 1)}: {module.get('moduleTitle', 'N/A')}")
            # The new schema does not have moduleDescription or learningObjectives in the same way.
            # If the original schema's fields are desired, they need to be added to the course_schema dictionary.
            # For now, I'm removing these lines to align with the new schema's structure.
            # st.markdown(f"*{module.get('moduleDescription', '')}*")
            # if module.get("learningObjectives"):
            #     st.markdown("**Learning Objectives:**")
            #     for obj in module.get("learningObjectives", []):
            #         st.markdown(f"- {obj}")

            for c_idx, chapter in enumerate(module.get("chapters", [])):
                # Use the same chapter_id logic as app (2).py for consistency
                chapter_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_chapter_{chapter['chapterTitle'].replace(' ', '_').replace('.', '').replace(',', '')}"

                st.markdown(f"**Chapter: {chapter['chapterTitle']}**")
                st.markdown(f"*{chapter.get('description', '')}*")

                # Checkbox for completion
                is_completed = st.checkbox(
                    f"Mark as complete: **{chapter['chapterTitle']}**",
                    value=course['completion_status'].get(chapter_id, False),
                    key=f"checkbox_{chapter_id}" # Unique key for each checkbox
                )
                # Update completion status in session state if changed
                if is_completed != course['completion_status'].get(chapter_id, False):
                    course['completion_status'][chapter_id] = is_completed
                    st.session_state.courses[st.session_state.selected_course_index] = course # Update the course in session state
                    st.rerun() # Rerun to update the progress bar immediately

                # Generate detailed content button
                if st.button(f"Generate Detailed Content for '{chapter['chapterTitle']}'", key=f"gen_content_{chapter_id}"):
                    with st.spinner("Generating detailed chapter content..."):
                        content_prompt = f"""
                        Generate detailed content for chapter '{chapter['chapterTitle']}' in the '{course['courseTitle']}' course.
                        This course is for a {difficulty} level audience.
                        Chapter description: {chapter['description']}.
                        Provide comprehensive, readable content with examples if relevant, aiming for a few paragraphs.
                        """
                        loop = get_or_create_eventloop()
                        detailed_content = loop.run_until_complete(
                            generate_content_with_gemini(content_prompt, temperature, max_tokens, top_k, top_p)
                        )
                        loop.close() # Close the loop after use
                        if detailed_content:
                            st.session_state.chapter_contents[chapter_id] = detailed_content
                            st.success("Detailed content generated!")                        
                        else:
                            st.warning("We're having trouble generating detailed content at the moment. Try generating content for a different chapter first, or wait a moment before trying again.")
                
                # Display chapter content
                if chapter_id in st.session_state.chapter_contents:
                    st.markdown("---")
                    st.info(st.session_state.chapter_contents[chapter_id])
                    st.markdown("---")
                else:
                    st.info("Click 'Generate Detailed Content' to get more information for this chapter.")

            # --- Quiz for this module (after chapters) ---
            module_quiz_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_quiz"
            if module_quiz_id not in st.session_state.quiz_progress:
                st.session_state.quiz_progress[module_quiz_id] = {"completed": False, "score": 0, "answers": []}

            # Button to take quiz for the module
            if st.button(f"Take Quiz for Module {module['moduleNumber']} ({module['moduleTitle']})", key=f"quiz_btn_{module_quiz_id}"):
                with st.spinner("Generating quiz... This may take a moment."):
                    # Collect content from all chapters in the module for quiz generation context
                    module_content = "\n".join([chapter['description'] for chapter in module.get('chapters', [])])
                    if not module_content.strip():
                        st.warning("Cannot generate quiz: No content available for this module's chapters.")
                    else:
                        loop = get_or_create_eventloop()
                        quiz_data = loop.run_until_complete(quiz_utils.generate_quiz_with_gemini(
                            module_content, API_KEY, temperature, max_tokens, top_k, top_p, num_questions=5
                        ))
                        loop.close() # Close the loop after use
                        if quiz_data and "questions" in quiz_data:
                            st.session_state.quiz_progress[module_quiz_id] = {
                                "questions": quiz_data["questions"],
                                "completed": False,
                                "answers": [None] * len(quiz_data["questions"]),
                                "score": 0
                            }
                            st.success("Quiz generated! Scroll down to attempt it.")
                        else:
                            st.warning("We're having trouble generating the quiz right now. Try generating detailed content for more chapters in this module first, then attempt the quiz generation again.")
            
            # Display quiz if available
            quiz_obj = st.session_state.quiz_progress.get(module_quiz_id, {})
            if quiz_obj.get("questions"):
                st.markdown(f"#### Quiz for Module {module['moduleNumber']} ({module['moduleTitle']})")
                
                if not quiz_obj.get("completed"):
                    with st.form(f"quiz_form_{module_quiz_id}"):
                        current_answers = []
                        for i, q in enumerate(quiz_obj["questions"]):
                            st.markdown(f"**Q{i+1}: {q['question']}**")
                            options = q["options"]
                            selected_option = st.radio(
                                f"Select answer for Q{i+1}", # Use unique key for each radio button
                                options,
                                index=options.index(quiz_obj['answers'][i]) if quiz_obj['answers'][i] in options else None, # No default selection
                                key=f"quiz_q_{module_quiz_id}_{i}"
                            )
                            current_answers.append(selected_option)
                        
                        submit_quiz_button = st.form_submit_button("Submit Quiz", help="Submit your answers for this quiz.")
                        if submit_quiz_button:
                            correct_answers = [q["answer"] for q in quiz_obj["questions"]]
                            quiz_utils.update_quiz_progress(st.session_state, module_quiz_id, current_answers, correct_answers)
                            st.rerun() # Rerun to display quiz results (score, explanations)
                else:
                    st.success(f"Quiz completed! Score: {quiz_obj['score']}/{len(quiz_obj['questions'])}")
                    if st.button("Retake Quiz", key=f"retake_{module_quiz_id}"):
                        del st.session_state.quiz_progress[module_quiz_id]
                        st.rerun()
                    for idx, q in enumerate(quiz_obj["questions"]):
                        st.markdown(f"**Q{idx+1}: {q['question']}**")
                        user_ans = quiz_obj['answers'][idx]
                        correct_ans = q['answer']
                        
                        # Display user's answer and correctness
                        if user_ans == correct_ans:
                            st.markdown(f"> ✅ Your answer: **{user_ans}** (Correct)")
                        else:
                            st.markdown(f"> ❌ Your answer: **{user_ans}** (Incorrect, Correct was: **{correct_ans}**)")
                        st.markdown(f"> *Explanation: {q['explanation']}*")
                        st.markdown("---") # Separator between questions
        
        st.markdown(f"**Conclusion:** {course.get('conclusion', 'N/A')}")
    elif st.session_state.selected_course_index is None and not st.session_state.courses:
        st.info("Please generate a new course using the form above to get started!")
    elif st.session_state.selected_course_index is None and st.session_state.courses:
        st.info("Select an existing course from the sidebar to view its content.")


    # --- Chatbot Section ---
    if st.session_state.selected_course_index is not None:
        st.divider()
        st.subheader("💬 Course Assistant")
        st.write("Ask questions about the currently displayed course content!")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message["user"])
            with st.chat_message("assistant"):
                st.write(message["assistant"])
        
        # Chat input
        if user_message := st.chat_input("Ask a question about the course..."):
            # Get current course content as JSON string for context
            course = st.session_state.courses[st.session_state.selected_course_index]
            course_content = json.dumps(course)  
            
            with st.chat_message("user"):
                st.write(user_message)
            
            with st.chat_message("assistant"):
                response = process_chat_message(user_message, course_content)
                st.write(response)

# To run the app directly if this file is executed
if __name__ == "__main__":
    run_app()
