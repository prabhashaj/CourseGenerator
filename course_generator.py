from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import streamlit as st
import httpx
import json
import os
import importlib
import asyncio  # Import asyncio for event loop management
import time  # Import time for delays
from datetime import datetime  # Import datetime for timestamping courses

# --- Utility Import ---
try:
    quiz_utils = importlib.import_module("quiz_utils")
except ImportError:
    st.error("The 'quiz_utils.py' file was not found. Please make sure it's in the same directory.")
    st.stop()

# --- API Key Setup ---
# Use Streamlit secrets for deployment and fall back to environment variable
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") 
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
        st.session_state.max_tokens = 6144  # Increased from 4096 for better chapter details
    if "top_k" not in st.session_state: # Ensure top_k and top_p are initialized
        st.session_state.top_k = 32
    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0
    # Add a flag to ensure sidebar content is rendered only once
    if "sidebar_rendered" not in st.session_state:
        st.session_state.sidebar_rendered = False
    # Track successful course generation
    if "last_generation_successful" not in st.session_state:
        st.session_state.last_generation_successful = False


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
            min_value=1024,
            max_value=8192,
            value=st.session_state.max_tokens, # Use session state value
            step=512,
            help="Maximum number of tokens to generate. Increase for larger courses (5-10 modules need 4096+ tokens)."
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

def is_response_complete(content):
    """Check if the generated content appears complete with emphasis on summary completion"""
    if not content or len(content.strip()) < 100:
        return False
    
    content_lower = content.lower().strip()
    
    # Check for obvious incomplete endings
    incomplete_indicators = [
        content.strip().endswith('...'),
        content.strip().endswith('etc.'),
        'continues...' in content_lower,
        'to be continued' in content_lower,
        len(content.split()) < 150,
        # Mid-sentence endings
        content.strip().endswith(':') and not any(word in content_lower[-50:] for word in ['example:', 'note:', 'summary:', 'conclusion:']),
        content.strip().endswith('='),
        content.strip().endswith('+'),
        content.strip().endswith('('),
        # Check for incomplete calculations
        (content.strip().endswith('.') and 
         any(pattern in content.strip()[-30:] for pattern in ['= 0.', '+ 0.', '- 0.', '* 0.', '/ 0.']))
    ]
    
    # Special check for incomplete summary section
    has_summary_section = 'summary' in content_lower or 'conclusion' in content_lower
    if has_summary_section:
        # Find the summary section
        summary_start = -1
        for keyword in ['## summary', '# summary', 'summary:', 'in summary', 'to summarize']:
            idx = content_lower.find(keyword)
            if idx != -1:
                summary_start = idx
                break
        
        if summary_start != -1:
            summary_section = content[summary_start:].strip()
            # Check if summary section is too short or ends abruptly
            summary_words = summary_section.split()
            if len(summary_words) < 20:  # Summary should have at least 20 words
                return False
            
            # Check for incomplete sentence endings in summary
            last_sentence = summary_section.strip()
            if (last_sentence.endswith(',') or 
                last_sentence.endswith('and') or 
                last_sentence.endswith('the') or
                last_sentence.endswith('of') or
                last_sentence.endswith('in') or
                last_sentence.endswith('to') or
                last_sentence.endswith('for')):
                return False
    
    # If content has reasonable length and structure, consider it complete
    if len(content.split()) > 200 and has_summary_section:
        return not any(incomplete_indicators)
    
    return not any(incomplete_indicators)

def safe_json_parse(text_response):
    """Safely parse JSON, attempting to repair common issues."""
    import re
    
    # Clean the response
    text_response = text_response.strip()
    
    # Remove trailing commas before } or ]
    text_response = re.sub(r',([\s\n]*[}}\]])', r'\1', text_response)
    
    # Remove any markdown code block markers
    text_response = re.sub(r'^```json|```$', '', text_response, flags=re.MULTILINE)
    text_response = text_response.strip()
    
    # Handle incomplete JSON by finding the last complete object
    if not text_response.endswith('}'):
        # Find the last complete closing brace for the main object
        brace_count = 0
        last_complete_pos = -1
        for i, char in enumerate(text_response):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_complete_pos = i
        
        if last_complete_pos != -1:
            text_response = text_response[:last_complete_pos + 1]
    
    try:
        return json.loads(text_response)
    except json.JSONDecodeError as e:
        st.warning(f"JSONDecodeError: {e}. Attempting advanced repair...")
        
        # Try to fix common JSON issues
        try:
            # Fix unescaped quotes in strings
            text_response = re.sub(r'(?<!\\)"(?=.*")', '\\"', text_response)
            
            # Try again
            return json.loads(text_response)
        except json.JSONDecodeError:
            # Last resort: try to extract partial valid JSON
            try:
                # Find the modules array and try to parse it separately
                modules_match = re.search(r'"modules"\s*:\s*\[(.*?)\]', text_response, re.DOTALL)
                if modules_match:
                    # Create a minimal valid structure
                    return {
                        "courseTitle": "Generated Course",
                        "introduction": "Course introduction",
                        "modules": json.loads('[' + modules_match.group(1) + ']'),
                        "conclusion": "Course conclusion"
                    }
            except:
                pass
                
        return {"error": f"Invalid JSON after all repair attempts: {str(e)}", "raw": text_response}

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
            # Dynamic timeout based on token count and complexity
            if max_tokens <= 1024:
                timeout_duration = 60
            elif max_tokens <= 2048:
                timeout_duration = 120
            elif max_tokens <= 4096:
                timeout_duration = 180
            else:
                timeout_duration = 240
                
            response = await client.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=timeout_duration
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
        
        # Reset generation success flag
        st.session_state.last_generation_successful = False
        
        with st.spinner(f"🔄 Generating {difficulty} level course on {course_topic}... This may take a while for larger courses."):
            # Show progress for larger courses
            if num_modules >= 7:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Preparing course generation...")
            
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

                # Construct the prompt for the LLM with improved instructions for larger courses
                chapters_per_module = 3  # Fixed to exactly 3 chapters per module
                
                # Dynamic token allocation based on number of modules
                base_tokens = 1024
                tokens_per_module = 400  # Approximate tokens needed per module (with 3 chapters each)
                optimal_tokens = base_tokens + (num_modules * tokens_per_module)
                
                # Ensure we don't exceed the max_tokens setting but use efficient allocation
                if optimal_tokens > max_tokens:
                    # Use available max_tokens
                    course_tokens = max_tokens
                    st.info(f"🔧 Using {course_tokens} tokens for {num_modules} modules. Consider increasing max tokens in sidebar for better quality.")
                else:
                    # Use optimal allocation
                    course_tokens = optimal_tokens
                    st.info(f"🎯 Optimized token usage: {course_tokens} tokens for {num_modules} modules")
                
                course_prompt = f"""
                Generate a comprehensive course outline in JSON format for a '{course_topic}' course.
                
                REQUIREMENTS:
                - Course level: {difficulty}
                - Number of modules: EXACTLY {num_modules}
                - Each module should have EXACTLY {chapters_per_module} chapters
                - Target reading time per module: {read_time_per_module}
                - Ensure all content is relevant and well-structured
                
                CHAPTER DESCRIPTION REQUIREMENTS:
                - Each chapter description should be {3 if num_modules <= 5 else 2}-{4 if num_modules <= 3 else 3} sentences long
                - Include specific topics, concepts, and learning outcomes covered in that chapter
                - Mention key skills or knowledge the student will gain
                - Be descriptive about the actual content, not just what the chapter is about
                - Use actionable language (e.g., "Learn how to...", "Discover...", "Master...")
                
                IMPORTANT JSON FORMATTING RULES:
                1. Return ONLY valid JSON, no additional text
                2. Use proper JSON escaping for quotes and special characters
                3. Ensure all arrays and objects are properly closed
                4. Module numbers should be sequential starting from 1
                
                The JSON must strictly follow this structure:
                {{
                    "courseTitle": "Clear, descriptive title",
                    "introduction": "Brief course overview and objectives",
                    "modules": [
                        {{
                            "moduleNumber": 1,
                            "moduleTitle": "Module title",
                            "chapters": [
                                {{
                                    "chapterTitle": "Chapter title",
                                    "description": "Comprehensive {3 if num_modules <= 5 else 2}-{4 if num_modules <= 3 else 3} sentence description explaining what topics are covered, what skills will be learned, and what specific concepts will be mastered in this chapter."
                                }}
                            ]
                        }}
                    ],
                    "conclusion": "Course wrap-up and next steps"
                }}
                
                {f'Example of good chapter descriptions for {num_modules} modules:' if num_modules <= 5 else 'Keep descriptions concise but informative:'}
                - {"Learn the fundamental concepts of neural networks including perceptrons, activation functions, and forward propagation. Discover how neurons process information and understand the mathematical foundations behind network architecture. Master the basic building blocks that form the foundation of all deep learning models. Practice implementing simple neural networks from scratch." if num_modules <= 5 else "Learn fundamental neural network concepts including perceptrons and activation functions. Master the mathematical foundations behind network architecture and practice implementing basic networks."}
                
                Make sure to create exactly {num_modules} modules with substantive, detailed content.
                """

                loop = get_or_create_eventloop()
                
                # Retry logic for course generation
                max_retries = 3
                course_data = None
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            if num_modules >= 7:
                                status_text.text(f"Retrying course generation (attempt {attempt + 1}/{max_retries})...")
                                progress_bar.progress(attempt / max_retries)
                            st.info(f"Retrying course generation (attempt {attempt + 1}/{max_retries})...")
                        elif num_modules >= 7:
                            status_text.text(f"Generating course... (attempt {attempt + 1}/{max_retries})")
                            progress_bar.progress(0.3)
                        
                        course_data = loop.run_until_complete(
                            generate_content_with_gemini(
                                course_prompt,
                                temperature, # Use values from session state (updated by sidebar sliders)
                                course_tokens,  # Use optimized token allocation
                                top_k,       # Use values from session state (updated by sidebar sliders)
                                top_p,       # Use values from session state (updated by sidebar sliders)
                                response_schema=course_schema
                            )
                        )
                        
                        # Validate the generated course data
                        if course_data and isinstance(course_data, dict) and course_data.get("courseTitle"):
                            modules = course_data.get("modules", [])
                            if len(modules) == num_modules:
                                # Additional validation: check if each module has exactly 3 chapters
                                all_modules_valid = True
                                for module in modules:
                                    chapters = module.get("chapters", [])
                                    if len(chapters) != 3:
                                        all_modules_valid = False
                                        break
                                
                                if all_modules_valid:
                                    # Successful generation - break out of retry loop
                                    break
                                else:
                                    st.warning(f"Some modules don't have exactly 3 chapters. Retrying...")
                                    course_data = None
                            else:
                                st.warning(f"Generated {len(modules)} modules instead of {num_modules}. Retrying...")
                                course_data = None
                        else:
                            course_data = None
                            
                    except Exception as e:
                        st.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                        course_data = None
                        
                    if attempt < max_retries - 1:
                        # Wait before retry
                        time.sleep(2)
                
                loop.close() # Close the loop after use

                # Check if course generation was successful
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
                    
                    # Update progress bar for larger courses
                    if num_modules >= 7:
                        progress_bar.progress(1.0)
                        status_text.text("Course generated successfully!")
                    
                    # Mark as successfully generated in session state
                    st.session_state.last_generation_successful = True
                    st.success(f"✨ Successfully generated course: {new_course['courseTitle']}")
                    st.rerun()                
                else:
                    # Only show error message if course generation actually failed
                    st.session_state.last_generation_successful = False
                    st.error("❌ Unable to generate the course after multiple attempts. Please try again with fewer modules or a different topic.")
                    
            except Exception as e:
                # Only show error if generation wasn't marked as successful
                if not st.session_state.get('last_generation_successful', False):
                    st.error("❌ We encountered an unexpected error during course generation. Please try again.")
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

                # Enhanced chapter display with better formatting
                st.markdown(f"#### 📖 Chapter {c_idx + 1}: {chapter['chapterTitle']}")
                
                # Display chapter description in an info box for better visibility
                with st.container():
                    st.markdown("**📝 What you'll learn in this chapter:**")
                    st.info(chapter.get('description', 'No description available'))

                # Checkbox for completion with better spacing
                col1, col2 = st.columns([3, 1])
                with col1:
                    is_completed = st.checkbox(
                        f"✅ Mark chapter as completed",
                        value=course['completion_status'].get(chapter_id, False),
                        key=f"checkbox_{chapter_id}" # Unique key for each checkbox
                    )
                
                with col2:
                    # Generate detailed content button
                    if st.button(f"📚 Get Details", key=f"gen_content_{chapter_id}", help=f"Generate detailed content for '{chapter['chapterTitle']}'"):
                        with st.spinner("Generating detailed chapter content..."):
                            # Increased token allocation for more detailed content while ensuring completion
                            total_chapters = len(st.session_state.courses[st.session_state.selected_course_index].get('modules', [])) * 3
                            if total_chapters <= 9:  # 3 modules or fewer
                                content_tokens = min(max_tokens, 5120)  # Increased for more detailed content
                                content_depth = "comprehensive and highly detailed"
                                content_length = "extensive"
                            elif total_chapters <= 18:  # 6 modules or fewer
                                content_tokens = min(max_tokens, 4096)  # Increased for more detailed content
                                content_depth = "thorough and detailed"
                                content_length = "substantial"
                            else:  # 7+ modules
                                content_tokens = min(max_tokens, 3072)  # Increased for more detailed content
                                content_depth = "focused but comprehensive"
                                content_length = "comprehensive"
                            
                            content_prompt = f"""
Create comprehensive, detailed educational content for this chapter. You have substantial token allocation - use it to provide thorough coverage while ensuring you complete with a full Summary section.

**Chapter:** {chapter['chapterTitle']}
**Course:** {course['courseTitle']} ({difficulty} level)
**Token Allocation:** {content_tokens} tokens (Reserve 200-250 tokens for complete Summary)
**Chapter Focus:** {chapter['description']}

# {chapter['chapterTitle']}

## Introduction
Provide a comprehensive introduction explaining what this topic is, its significance, historical context (if relevant), and why it's important in the broader field. Include real-world relevance and motivation for learning this topic.

## Core Concepts and Theory
{f"Provide in-depth explanations of fundamental concepts, theoretical foundations, mathematical principles (where applicable), and detailed breakdowns of complex ideas. Use multiple examples and analogies to illustrate abstract concepts." if total_chapters <= 12 else "Explain key concepts thoroughly with clear examples and theoretical foundations."}

{f"Include step-by-step processes, detailed methodologies, algorithms, formulas, and comprehensive explanations of how things work at a deeper level." if total_chapters <= 9 else "Cover essential principles, methodologies, and step-by-step processes."}

## Detailed Examples and Applications
{f"Provide multiple detailed examples, case studies, real-world applications, worked solutions, and practical scenarios. Show different approaches and variations to demonstrate versatility of concepts." if total_chapters <= 12 else "Demonstrate practical applications with detailed examples and case studies."}

## Implementation and Practice
{f"Include detailed practical examples, code snippets (if applicable), calculations, procedures, exercises, and hands-on applications. Provide troubleshooting tips and common pitfalls to avoid." if total_chapters <= 12 else "Show practical implementations with examples and common applications."}

{f"## Advanced Considerations\nDiscuss advanced topics, edge cases, limitations, best practices, optimization techniques, and connections to other related concepts or fields." if total_chapters <= 9 else ""}

## Summary
**[REQUIRED - MUST COMPLETE THIS SECTION]**
Write a comprehensive summary covering:
- Key concepts and principles learned in this chapter
- Main theoretical and practical takeaways
- How this knowledge connects to the broader course and field
- Important formulas, processes, or methodologies to remember
- Next steps or how this leads into subsequent topics

**COMPLETION REQUIREMENTS:**
1. Focus ONLY on: {chapter['chapterTitle']}
2. Use the full token allocation to provide comprehensive coverage
3. MUST finish with a complete Summary section - never stop mid-sentence
4. Reserve 200-250 tokens for the Summary - ensure it's thorough but complete
5. End with a natural, complete conclusion that reinforces learning

Create detailed, {content_depth} educational content. Use your full token budget effectively while ensuring completion:
                            """
                            
                            # Simplified retry logic for better performance
                            max_retries = 2  # Reduced from 3 to avoid excessive retries
                            detailed_content = None
                            original_tokens = content_tokens
                            
                            for attempt in range(max_retries):
                                try:
                                    loop = get_or_create_eventloop()
                                    detailed_content = loop.run_until_complete(
                                        generate_content_with_gemini(content_prompt, temperature, content_tokens, top_k, top_p)
                                    )
                                    loop.close()
                                    
                                    # Strict completion check focused on proper endings
                                    if detailed_content and len(detailed_content.strip()) > 200:
                                        word_count = len(detailed_content.split())
                                        has_summary = any(keyword in detailed_content.lower() for keyword in ['summary', 'conclusion', 'in summary', 'to conclude', 'key points'])
                                        
                                        # Check for complete summary section
                                        is_complete = is_response_complete(detailed_content)
                                        
                                        # More conservative retry logic - expect longer content now
                                        if word_count < 400 or not has_summary or not is_complete:
                                            if attempt < max_retries - 1:
                                                reason = []
                                                if word_count < 400:
                                                    reason.append(f"too short ({word_count} words, expected 400+)")
                                                if not has_summary:
                                                    reason.append("missing summary")
                                                if not is_complete:
                                                    reason.append("incomplete ending")
                                                
                                                st.warning(f"Content incomplete: {', '.join(reason)}. Retrying...")
                                                # Moderate token increase for better coverage
                                                content_tokens = min(max_tokens, content_tokens + 512)
                                        else:
                                            break  # Content is sufficiently complete
                                    elif attempt < max_retries - 1:
                                        st.warning(f"Content too short ({len(detailed_content.strip()) if detailed_content else 0} chars). Retrying...")
                                        content_tokens = min(max_tokens, content_tokens + 768)
                                        
                                except Exception as e:
                                    if attempt < max_retries - 1:
                                        st.warning(f"Generation failed (attempt {attempt + 1}). Retrying...")
                                    else:
                                        st.error(f"Generation failed: {str(e)}")
                                    
                                if attempt < max_retries - 1:
                                    time.sleep(1)  # Shorter wait time
                            
                            if detailed_content and len(detailed_content.strip()) > 100:
                                st.session_state.chapter_contents[chapter_id] = detailed_content
                                word_count = len(detailed_content.split())
                                st.success(f"📚 Comprehensive chapter content generated!")
                            else:
                                st.error("Unable to generate content. Try increasing max tokens in sidebar or try a different chapter.")
                
                # Update completion status in session state if changed
                if is_completed != course['completion_status'].get(chapter_id, False):
                    course['completion_status'][chapter_id] = is_completed
                    st.session_state.courses[st.session_state.selected_course_index] = course # Update the course in session state
                    st.rerun() # Rerun to update the progress bar immediately
                
                # Display chapter content
                if chapter_id in st.session_state.chapter_contents:
                    st.markdown("---")
                    st.markdown("**📚 Detailed Chapter Content:**")
                    with st.expander("Click to expand/collapse detailed content", expanded=True):
                        st.markdown(st.session_state.chapter_contents[chapter_id])
                    st.markdown("---")
                else:
                    st.info("Click 'Get Details' to generate comprehensive content for this chapter.")

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
                        # Optimize quiz tokens based on total course size
                        total_modules = len(st.session_state.courses[st.session_state.selected_course_index].get('modules', []))
                        if total_modules <= 3:
                            quiz_tokens = min(max_tokens, 1536)
                            num_questions = 5
                        elif total_modules <= 6:
                            quiz_tokens = min(max_tokens, 1024)
                            num_questions = 4
                        else:
                            quiz_tokens = min(max_tokens, 768)
                            num_questions = 3
                            
                        loop = get_or_create_eventloop()
                        quiz_data = loop.run_until_complete(quiz_utils.generate_quiz_with_gemini(
                            module_content, API_KEY, temperature, quiz_tokens, top_k, top_p, num_questions=num_questions
                        ))
                        loop.close() # Close the loop after use
                        if quiz_data and "questions" in quiz_data:
                            st.session_state.quiz_progress[module_quiz_id] = {
                                "questions": quiz_data["questions"],
                                "completed": False,
                                "answers": [None] * len(quiz_data["questions"]),
                                "score": 0
                            }
                            st.success(f"Quiz generated with {num_questions} questions! Scroll down to attempt it.")
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
