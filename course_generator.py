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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

# --- Utility Import ---
try:
    quiz_utils = importlib.import_module("quiz_utils")
except ImportError:
    st.error("The 'quiz_utils.py' file was not found. Please make sure it's in the same directory.")
    st.stop()

# --- API Key Setup ---
# Use Streamlit secrets for deployment and fall back to environment variable
try:
    # Try Streamlit secrets first (for deployment)
    API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
except (AttributeError, FileNotFoundError):
    # Fall back to environment variables (for local development)
    API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Ensure we have a valid API key
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY

def build_chapter_content_prompt(chapter, course, difficulty, content_tokens, total_chapters, content_depth):
    """Build chapter content prompt without complex f-string expressions to avoid syntax errors."""
    # Build sections based on total chapters
    core_concepts_section = ""
    if total_chapters <= 12:
        core_concepts_section = "Provide in-depth explanations of fundamental concepts, theoretical foundations, mathematical principles (where applicable), and detailed breakdowns of complex ideas. Use multiple examples and analogies to illustrate abstract concepts."
    else:
        core_concepts_section = "Explain key concepts thoroughly with clear examples and theoretical foundations."
    
    implementation_section = ""
    if total_chapters <= 9:
        implementation_section = "Include step-by-step processes, detailed methodologies, algorithms, formulas, and comprehensive explanations of how things work at a deeper level."
    else:
        implementation_section = "Cover essential principles, methodologies, and step-by-step processes."
    
    examples_section = ""
    if total_chapters <= 12:
        examples_section = "Provide multiple detailed examples, case studies, real-world applications, worked solutions, and practical scenarios. Show different approaches and variations to demonstrate versatility of concepts."
    else:
        examples_section = "Demonstrate practical applications with detailed examples and case studies."
    
    practice_section = ""
    if total_chapters <= 12:
        practice_section = "Include detailed practical examples, code snippets (if applicable), calculations, procedures, exercises, and hands-on applications. Provide troubleshooting tips and common pitfalls to avoid."
    else:
        practice_section = "Show practical implementations with examples and common applications."
    
    advanced_section = ""
    if total_chapters <= 9:
        advanced_section = "\n\n## Advanced Considerations\nDiscuss advanced topics, edge cases, limitations, best practices, optimization techniques, and connections to other related concepts or fields."
    
    # Build the complete prompt
    prompt = f"""Create comprehensive, detailed educational content for this chapter. You have substantial token allocation - use it to provide thorough coverage while ensuring you complete with a full Summary section.

**Chapter:** {chapter['chapterTitle']}
**Course:** {course['courseTitle']} ({difficulty} level)
**Token Allocation:** {content_tokens} tokens (Reserve 200-250 tokens for complete Summary)
**Chapter Focus:** {chapter['description']}

# {chapter['chapterTitle']}

## Introduction
Provide a comprehensive introduction explaining what this topic is, its significance, historical context (if relevant), and why it's important in the broader field. Include real-world relevance and motivation for learning this topic.

## Core Concepts and Theory
{core_concepts_section}

{implementation_section}

## Detailed Examples and Applications
{examples_section}

## Implementation and Practice
{practice_section}{advanced_section}

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
    return prompt

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
    if "top_k" not in st.session_state:
        st.session_state.top_k = 32
    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0
    if "sidebar_rendered" not in st.session_state:
        st.session_state.sidebar_rendered = False
    if "last_generation_successful" not in st.session_state:
        st.session_state.last_generation_successful = False
    
    # Initialize enhanced tracking for existing courses
    for course in st.session_state.courses:
        if "chapter_viewed" not in course:
            course["chapter_viewed"] = {}
        if "chapter_completed_at" not in course:
            course["chapter_completed_at"] = {}
        if "study_sessions" not in course:
            course["study_sessions"] = []
        if "reading_times" not in course:
            course["reading_times"] = {}


def clear_chat_history():
    """Clear chat history when switching courses or generating a new one"""
    if "chat_history" in st.session_state:
        st.session_state.chat_history = []
    if "conversation" in st.session_state:
        st.session_state.conversation.memory.clear()
    st.session_state.sidebar_rendered = False


def show_navigation():
    """Display the navigation menu and AI settings in the sidebar."""
    with st.sidebar:
        # AI Generation Settings in sidebar
        st.header("⚙️ AI Generation Settings")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.01,
            help="Controls the randomness of the output. Higher values mean more creative."
        )
        st.session_state.max_tokens = st.slider(
            "Max Output Tokens",
            min_value=1024,
            max_value=8192,
            value=st.session_state.max_tokens,
            step=512,
            help="Maximum number of tokens to generate. Increase for larger courses (5-10 modules need 4096+ tokens)."
        )

        st.markdown("---")

        # Show saved courses section with progress indicators
        if st.session_state.courses:
            st.subheader("My Saved Courses")
            for idx, course in enumerate(st.session_state.courses):
                # Calculate progress for display
                completed = sum(1 for status in course.get("completion_status", {}).values() if status)
                total = course.get("total_chapters", 0)
                progress_percent = int((completed / total) * 100) if total > 0 else 0
                
                # Progress indicator emoji
                progress_emoji = "🎯" if progress_percent == 100 else "📈" if progress_percent > 50 else "📚" if progress_percent > 0 else "📖"
                
                # Use a consistent key for course selection buttons
                if st.button(
                    f"{progress_emoji} {course.get('courseTitle', 'Untitled Course')} ({progress_percent}%)",
                    key=f"sidebar_course_{idx}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_course_index == idx else "secondary",
                    help=f"Progress: {completed}/{total} chapters completed"
                ):
                    if st.session_state.selected_course_index != idx:
                        clear_chat_history()
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
    # Check API key at runtime
    if not API_KEY or API_KEY == "your_api_key_here":
        st.error("🔑 API Key is not configured properly. Please set your Gemini API key.")
        st.info("Get your API key from: https://aistudio.google.com/app/apikey")
        return None

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topK": int(top_k),
        "topP": top_p
    }
    payload = {"contents": chat_history, "generationConfig": generation_config}

    if response_schema:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = response_schema

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    
    try:
        async with httpx.AsyncClient() as client:
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
            
            if response.status_code == 400:
                st.error("🔑 **Invalid API Key!** Please check your Gemini API key is correct.")
                st.info("Get a new API key from: https://aistudio.google.com/app/apikey")
                return None
            elif response.status_code == 403:
                st.error("🚫 **API Access Forbidden!** Your API key may not have permission or quota exceeded.")
                return None
            elif response.status_code == 429:
                st.error("⏰ **Rate Limited!** Too many requests. Please wait a moment and try again.")
                return None
                
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
                    st.json(parsed_response["raw"])
                    return None
                return parsed_response
            return text_response
    except httpx.RequestError as e:
        st.error(f"🌐 **Network Error:** {str(e)}")
        st.info("Please check your internet connection and try again.")
        return None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            st.error("🔑 **Invalid API Key!** Please verify your Gemini API key is correct.")
        elif e.response.status_code == 403:
            st.error("🚫 **Access Forbidden!** Check your API key permissions.")
        elif e.response.status_code == 429:
            st.error("⏰ **Too Many Requests!** Please wait and try again.")
        else:
            st.error(f"🔴 **API Error:** HTTP {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"❌ **Unexpected Error:** {str(e)}")
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

    st.title("🎓 Course Generator")
    
    # Show API key status
    if not API_KEY or API_KEY == "your_api_key_here":
        st.error("🔑 **API Key Required!** Please set up your Gemini API key to use this application.")
        with st.expander("📋 **How to Set Up Your API Key**", expanded=True):
            st.markdown("""
            ### For Streamlit Cloud Deployment:
            1. Go to your Streamlit app settings
            2. Click on "Secrets"
            3. Add: `GEMINI_API_KEY = "your_actual_api_key_here"`
            
            ### For Local Development:
            **Option 1 - Environment Variable:**
            ```bash
            export GEMINI_API_KEY="your_actual_api_key_here"
            ```
            
            **Option 2 - .env file:**
            Create a `.env` file in your project directory:
            ```
            GEMINI_API_KEY=your_actual_api_key_here
            ```
            
            ### Get Your API Key:
            🔗 [Get Gemini API Key](https://aistudio.google.com/app/apikey)
            
            **Note:** Make sure to keep your API key secure and never commit it to version control!
            """)
    
    st.info("👈 Open the sidebar to explore exciting features")
    st.markdown("Generate custom course outlines, track your progress, and get detailed chapter content!")
    
    temperature = st.session_state.temperature
    max_tokens = st.session_state.max_tokens
    top_k = st.session_state.top_k 
    top_p = st.session_state.top_p 
    
    st.markdown("---")

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
        
        generate_button = st.form_submit_button(
            "Generate New Course Outline",
            type="primary",
            use_container_width=True
        )
        
    # Course Generation Logic
    if generate_button:
        if not course_topic.strip():
            st.warning("⚠️ Please enter a course topic!")
            return
        
        # Check API key before proceeding
        if not API_KEY or API_KEY == "your_api_key_here":
            st.error("🔑 **API Key Required!** Please configure your Gemini API key to generate courses.")
            st.info("""
            **How to set up your API key:**
            
            **For Streamlit Cloud:**
            1. Go to your app settings
            2. Add a secret called `GEMINI_API_KEY` or `GOOGLE_API_KEY`
            3. Paste your Gemini API key as the value
            
            **For Local Development:**
            1. Set environment variable: `GEMINI_API_KEY=your_actual_api_key`
            2. Or create a `.env` file with: `GEMINI_API_KEY=your_actual_api_key`
            
            **Get your API key from:** https://aistudio.google.com/app/apikey
            """)
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
                                top_p,       # Use values from session_state (updated by sidebar sliders)
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

                    # Create new course with enhanced tracking data
                    new_course = {
                        **course_data,
                        "completion_status": completion_status,
                        "total_chapters": total_chapters,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "chapter_viewed": {},  # Track when chapters were viewed
                        "chapter_completed_at": {},  # Track completion timestamps
                        "study_sessions": []  # Track study sessions
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
        
        # Enhanced course completion tracking with better visuals
        completed_chapters = sum(1 for status in course.get("completion_status", {}).values() if status)
        viewed_chapters = len(course.get("chapter_viewed", {}))
        total_chapters = course.get("total_chapters", 0)
        completion_percentage = (completed_chapters / total_chapters) * 100 if total_chapters > 0 else 0
        view_percentage = (viewed_chapters / total_chapters) * 100 if total_chapters > 0 else 0
        
        # Calculate quiz completion
        total_modules = len(course.get("modules", []))
        completed_quizzes = 0
        for module in course.get("modules", []):
            module_quiz_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_quiz"
            quiz_obj = st.session_state.quiz_progress.get(module_quiz_id, {})
            if quiz_obj.get("completed", False):
                completed_quizzes += 1
        
        quiz_completion_percentage = (completed_quizzes / total_modules) * 100 if total_modules > 0 else 0
        
        # Calculate overall progress (chapters 70% weight, quizzes 30% weight)
        overall_progress = (completion_percentage * 0.7) + (quiz_completion_percentage * 0.3) if total_chapters > 0 and total_modules > 0 else completion_percentage
        
        # Progress display with multiple metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📖 Chapters Viewed", f"{viewed_chapters}/{total_chapters}", f"{view_percentage:.0f}%")
        with col2:
            st.metric("✅ Chapters Completed", f"{completed_chapters}/{total_chapters}", f"{completion_percentage:.0f}%")
        with col3:
            st.metric("Quizzes Completed", f"{completed_quizzes}/{total_modules}", f"{quiz_completion_percentage:.0f}%")

        # Progress bar with overall progress
        st.markdown("**Learning Progress:**")
        st.progress(overall_progress / 100, text=f"Overall Progress: {overall_progress:.1f}% (Chapters: {completion_percentage:.0f}% + Quizzes: {quiz_completion_percentage:.0f}%)")
        
        # Course introduction
        st.markdown(f"**Introduction:** {course.get('introduction', '')}")
        
        # Modules and Chapters
        for m_idx, module in enumerate(course.get("modules", [])):
            st.markdown(f"### Module {module.get('moduleNumber', m_idx + 1)}: {module.get('moduleTitle', 'N/A')}")

            for c_idx, chapter in enumerate(module.get("chapters", [])):
                # Use the same chapter_id logic for consistency
                chapter_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_chapter_{chapter['chapterTitle'].replace(' ', '_').replace('.', '').replace(',', '')}"

                # Get tracking status
                is_completed = course['completion_status'].get(chapter_id, False)
                is_viewed = chapter_id in course.get("chapter_viewed", {})
                completion_time = course.get("chapter_completed_at", {}).get(chapter_id)
                view_time = course.get("chapter_viewed", {}).get(chapter_id)

                # Enhanced chapter display with status indicators
                status_icon = "✅" if is_completed else "👁️" if is_viewed else "📖"
                status_text = "Completed" if is_completed else "Viewed" if is_viewed else "Not Started"
                
                st.markdown(f"#### {status_icon} Chapter {c_idx + 1}: {chapter['chapterTitle']}")
                
                # Status and timing info
                status_col1, status_col2 = st.columns([3, 1])
                with status_col1:
                    if is_completed and completion_time:
                        reading_info = course.get("reading_times", {}).get(chapter_id, {})
                        time_spent = reading_info.get("time_spent", 0)
                        st.caption(f"✅ Completed on {completion_time} (Read: {time_spent:.1f}min)")
                    elif is_viewed and view_time:
                        st.caption(f"👁️ Viewed on {view_time}")
                    else:
                        st.caption("📖 Ready to study")
                
                with status_col2:
                    # Show estimated reading time if content exists
                    reading_info = course.get("reading_times", {}).get(chapter_id, {})
                    if reading_info:
                        est_time = reading_info.get("estimated_minutes", 5)
                        st.caption(f"~{est_time} min read")
                    else:
                        st.caption("~5 min read")
                
                # Display chapter description in an info box for better visibility
                with st.container():
                    st.markdown("**📝 What you'll learn in this chapter:**")
                    st.info(chapter.get('description', 'No description available'))

                # Generate detailed content button
                if st.button(f"📚 Study", key=f"gen_content_{chapter_id}", help=f"Generate detailed content for '{chapter['chapterTitle']}'", use_container_width=True):
                        with st.spinner("Generating detailed chapter content..."):
                            # Increased token allocation for more detailed content while ensuring completion
                            total_chapters = len(st.session_state.courses[st.session_state.selected_course_index].get('modules', [])) * 3
                            if total_chapters <= 9:
                                content_tokens = min(max_tokens, 5120)
                                content_depth = "comprehensive and highly detailed"
                            elif total_chapters <= 18:
                                content_tokens = min(max_tokens, 4096)
                                content_depth = "thorough and detailed"
                            else:
                                content_tokens = min(max_tokens, 3072)
                                content_depth = "focused but comprehensive"
                            
                            content_prompt = build_chapter_content_prompt(
                                chapter, course, difficulty, content_tokens, total_chapters, content_depth
                            )
                            
                            max_retries = 2
                            detailed_content = None
                            
                            for attempt in range(max_retries):
                                try:
                                    loop = get_or_create_eventloop()
                                    detailed_content = loop.run_until_complete(
                                        generate_content_with_gemini(content_prompt, temperature, content_tokens, top_k, top_p)
                                    )
                                    loop.close()
                                    
                                    if detailed_content and len(detailed_content.strip()) > 200:
                                        word_count = len(detailed_content.split())
                                        has_summary = any(keyword in detailed_content.lower() for keyword in ['summary', 'conclusion', 'in summary', 'to conclude', 'key points'])
                                        is_complete = is_response_complete(detailed_content)
                                        
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
                                                content_tokens = min(max_tokens, content_tokens + 512)
                                        else:
                                            break
                                    elif attempt < max_retries - 1:
                                        st.warning(f"Content too short ({len(detailed_content.strip()) if detailed_content else 0} chars). Retrying...")
                                        content_tokens = min(max_tokens, content_tokens + 768)
                                        
                                except Exception as e:
                                    if attempt < max_retries - 1:
                                        st.warning(f"Generation failed (attempt {attempt + 1}). Retrying...")
                                    else:
                                        st.error(f"Generation failed: {str(e)}")
                                    
                                if attempt < max_retries - 1:
                                    time.sleep(1)
                            
                            if detailed_content and len(detailed_content.strip()) > 100:
                                st.session_state.chapter_contents[chapter_id] = detailed_content
                                if chapter_id not in course.get("chapter_viewed", {}):
                                    course["chapter_viewed"][chapter_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.session_state.courses[st.session_state.selected_course_index] = course
                                
                                # Calculate and store estimated reading time
                                word_count = len(detailed_content.split())
                                estimated_time_minutes = max(2, word_count / 200)  # Average reading speed: 200 words per minute, minimum 2 minutes
                                
                                if "reading_times" not in course:
                                    course["reading_times"] = {}
                                course["reading_times"][chapter_id] = {
                                    "estimated_minutes": round(estimated_time_minutes, 1),
                                    "word_count": word_count,
                                    "start_time": None,
                                    "time_spent": 0
                                }
                                st.session_state.courses[st.session_state.selected_course_index] = course
                                
                                # Content generated successfully - no message needed
                            else:
                                st.error("Unable to generate content. Try increasing max tokens in sidebar or try a different chapter.")
                
                # Display chapter content
                if chapter_id in st.session_state.chapter_contents:
                    st.markdown("---")
                    st.markdown("**📚 Detailed Chapter Content:**")
                    
                    # Get reading time information
                    reading_info = course.get("reading_times", {}).get(chapter_id, {})
                    estimated_minutes = reading_info.get("estimated_minutes", 5)
                    word_count = reading_info.get("word_count", 0)
                    
                    # Initialize reading session tracking
                    if f"reading_start_{chapter_id}" not in st.session_state:
                        st.session_state[f"reading_start_{chapter_id}"] = datetime.now()
                    
                    # Display reading information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📖 Est. time: {estimated_minutes} min")
                    with col2:
                        # Calculate time spent in current session
                        current_session_time = (datetime.now() - st.session_state[f"reading_start_{chapter_id}"]).total_seconds() / 60
                        total_time_spent = reading_info.get("time_spent", 0) + current_session_time
                        
                        if total_time_spent >= estimated_minutes * 0.8:  # 80% of estimated time
                            st.caption("⏰ Reading time met!")
                        else:
                            st.caption(f"⏰ Read: {total_time_spent:.1f}min")
                    with col3:
                        # Show progress indicator
                        progress_percentage = min(100, (total_time_spent / estimated_minutes) * 100)
                        st.caption(f"📊 Progress: {progress_percentage:.0f}%")
                    
                    with st.expander("Click to expand/collapse detailed content", expanded=True):
                        st.markdown(st.session_state.chapter_contents[chapter_id])
                    
                    # Auto-completion logic based on time spent
                    if not is_completed and total_time_spent >= estimated_minutes * 0.8:
                        # Auto-complete if user has spent 80% of estimated reading time
                        course['completion_status'][chapter_id] = True
                        course["chapter_completed_at"][chapter_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Update time spent in reading_times
                        if "reading_times" in course and chapter_id in course["reading_times"]:
                            course["reading_times"][chapter_id]["time_spent"] = total_time_spent
                        
                        st.session_state.courses[st.session_state.selected_course_index] = course
                        st.success("🎉 Chapter automatically completed based on reading time!")
                        st.balloons()
                        st.rerun()
                    
                    # Manual completion option for chapters not auto-completed yet
                    if not is_completed:
                        progress_percentage = min(100, (total_time_spent / estimated_minutes) * 100)
                        st.progress(progress_percentage / 100, text=f"Reading Progress: {progress_percentage:.0f}%")
                        
                        if st.button(f"✅ Mark as Complete Manually", key=f"manual_complete_{chapter_id}", 
                                   help="Complete this chapter manually without waiting for reading time"):
                            course['completion_status'][chapter_id] = True
                            course["chapter_completed_at"][chapter_id] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Update time spent
                            if "reading_times" in course and chapter_id in course["reading_times"]:
                                course["reading_times"][chapter_id]["time_spent"] = total_time_spent
                            
                            st.session_state.courses[st.session_state.selected_course_index] = course
                            st.success("Chapter marked as completed!")
                            st.rerun()
                    else:
                        st.success(f"✅ Chapter completed! (Read for {reading_info.get('time_spent', total_time_spent):.1f} minutes)")
                        if st.button(f"↩️ Mark as Incomplete", key=f"uncomplete_{chapter_id}"):
                            course['completion_status'][chapter_id] = False
                            course["chapter_completed_at"].pop(chapter_id, None)
                            # Reset reading time tracking
                            if "reading_times" in course and chapter_id in course["reading_times"]:
                                course["reading_times"][chapter_id]["time_spent"] = 0
                            st.session_state.courses[st.session_state.selected_course_index] = course
                            st.info("Chapter marked as incomplete.")
                            st.rerun()
                    
                    st.markdown("---")
                else:
                    st.info("Click 'Study' to generate comprehensive content for this chapter.")

            # Quiz for this module
            module_quiz_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_quiz"
            if module_quiz_id not in st.session_state.quiz_progress:
                st.session_state.quiz_progress[module_quiz_id] = {"completed": False, "score": 0, "answers": []}

            if st.button(f"Take Quiz for Module {module['moduleNumber']} ({module['moduleTitle']})", key=f"quiz_btn_{module_quiz_id}"):
                with st.spinner("Generating quiz... This may take a moment."):
                    module_content = "\n".join([chapter['description'] for chapter in module.get('chapters', [])])
                    if not module_content.strip():
                        st.warning("Cannot generate quiz: No content available for this module's chapters.")
                    else:
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
                            
                        if not API_KEY:
                            st.error("🔑 API Key is required for quiz generation. Please configure your Gemini API key in Streamlit secrets.")
                            st.info("💡 **For Streamlit Cloud:** Add your API key in the app settings under 'Secrets management'")
                        else:
                            loop = get_or_create_eventloop()
                            quiz_data = loop.run_until_complete(quiz_utils.generate_quiz_with_gemini(
                                module_content, API_KEY, temperature, quiz_tokens, top_k, top_p, num_questions=num_questions
                            ))
                            loop.close()
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
                                f"Select answer for Q{i+1}",
                                options,
                                index=options.index(quiz_obj['answers'][i]) if quiz_obj['answers'][i] in options else None,
                                key=f"quiz_q_{module_quiz_id}_{i}"
                            )
                            current_answers.append(selected_option)
                        
                        submit_quiz_button = st.form_submit_button("Submit Quiz", help="Submit your answers for this quiz.")
                        if submit_quiz_button:
                            correct_answers = [q["answer"] for q in quiz_obj["questions"]]
                            quiz_utils.update_quiz_progress(st.session_state, module_quiz_id, current_answers, correct_answers)
                            st.rerun()
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
                        st.markdown("---")
        
        st.markdown(f"**Conclusion:** {course.get('conclusion', 'N/A')}")
    elif st.session_state.selected_course_index is None and not st.session_state.courses:
        st.info("Please generate a new course using the form above to get started!")
    elif st.session_state.selected_course_index is None and st.session_state.courses:
        st.info("Select an existing course from the sidebar to view its content.")

    # Chatbot Section
    if st.session_state.selected_course_index is not None:
        st.divider()
        st.subheader("💬 Course Assistant")
        st.write("Ask questions about the currently displayed course content!")
        
        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message["user"])
            with st.chat_message("assistant"):
                st.write(message["assistant"])
        
        if user_message := st.chat_input("Ask a question about the course..."):
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
