# # course_generator.py (Updated)
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain
# import streamlit as st
# import httpx
# import json
# import os
# import importlib
# import asyncio  # Import asyncio for event loop management
# from datetime import datetime  # Import datetime for timestamping courses
# from langchain.chat_models import ChatOpenAI
# from langchain.output_parsers import StructuredOutputParser

# # --- Utility Import ---
# try:
#     quiz_utils = importlib.import_module("quiz_utils")
# except ImportError:
#     st.error("The 'quiz_utils.py' file was not found. Please make sure it's in the same directory.")
#     st.stop()

# # --- API Key Setup ---
# API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# if API_KEY:
#     os.environ["GOOGLE_API_KEY"] = API_KEY # Ensure it's in os.environ for other modules that might use it

# # --- Session State Initialization (local to course_generator) ---
# # This is re-added as a safeguard to ensure state variables are initialized
# # even if appx.py's global initialization doesn't catch all cases immediately.
# def init_session_state():
#     """Initializes session state for the course generator."""
#     if "courses" not in st.session_state:
#         st.session_state.courses = []
#     if "selected_course_index" not in st.session_state:
#         st.session_state.selected_course_index = None
#     if "chapter_contents" not in st.session_state:
#         st.session_state.chapter_contents = {}
#     if "quiz_progress" not in st.session_state:
#         st.session_state.quiz_progress = {}
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []
#     if "conversation" not in st.session_state:
#         llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
#         memory = ConversationBufferMemory()
#         st.session_state.conversation = ConversationChain(llm=llm, memory=memory)
#     if "current_mode" not in st.session_state:
#         st.session_state.current_mode = "course_generator"
#     if "previous_course_index" not in st.session_state:
#         st.session_state.previous_course_index = None

# def clear_chat_history():
#     """Clear chat history when switching courses"""
#     if "chat_history" in st.session_state:
#         st.session_state.chat_history = []
#     if "conversation" in st.session_state:
#         st.session_state.conversation.memory.clear()

# def show_navigation():
#     """Display the navigation menu with clear visual separation"""
#     with st.sidebar:
#         st.title("Navigation")
#         st.subheader("Choose Application Mode")
#         modes = {
#             "course_generator": "Course Generator & Quizzes",
#             "rag_chat": "Chat with Your Documents (RAG)",
#             "doc_creator": "Course Creation from Documents"
#         }
        
#         for mode, label in modes.items():
#             if st.button(
#                 label,
#                 use_container_width=True,
#                 type="primary" if st.session_state.current_mode == mode else "secondary"
#             ):
#                 if st.session_state.current_mode != mode:
#                     clear_chat_history()
#                     st.session_state.current_mode = mode
#                     st.rerun()

#         if st.session_state.current_mode == "course_generator":
#             st.info("You are in the Course Generator mode.")
            
#             # Show saved courses section
#             if st.session_state.courses:
#                 st.subheader("My Saved Courses")
#                 for idx, course in enumerate(st.session_state.courses):
#                     if st.button(
#                         f"📚 {course.get('title', 'Untitled Course')}",
#                         key=f"course_{idx}",
#                         use_container_width=True,
#                         type="primary" if st.session_state.selected_course_index == idx else "secondary"
#                     ):
#                         if st.session_state.selected_course_index != idx:
#                             clear_chat_history()
#                             st.session_state.selected_course_index = idx
#                             st.rerun()

# # --- Helper function to get or create an asyncio loop ---
# def get_or_create_eventloop():
#     """Gets the running event loop or creates a new one."""
#     try:
#         return asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         return loop

# def safe_json_parse(text_response):
#     import re
#     import json
#     # Remove trailing commas before } or ]
#     text_response = re.sub(r',([\s\n]*[}}\]])', r'\1', text_response)
#     # Remove any markdown code block markers
#     text_response = re.sub(r'^```json|```$', '', text_response, flags=re.MULTILINE)
#     # Truncate to last closing curly brace if needed
#     last_brace = text_response.rfind('}')
#     if last_brace == -1 or last_brace < len(text_response) - 3:
#         # JSON is likely truncated
#         return {"error": "The response appears to be truncated. Try reducing the number of modules or content length.", "raw": text_response}
#     text_response = text_response[:last_brace+1]
#     try:
#         return json.loads(text_response)
#     except json.JSONDecodeError as e:
#         return {"error": f"Invalid JSON: {str(e)}", "raw": text_response}

# # --- Gemini API Call Function ---
# async def generate_content_with_gemini(prompt, response_schema=None):
#     """Calls the Gemini API to generate content with fallback for errors."""
#     if not API_KEY:
#         st.error("API Key is not configured. Please set it in Streamlit secrets or environment variables.")
#         return None

#     chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
#     generation_config = {
#         "temperature": 0.7,
#         "maxOutputTokens": 2048,
#         "topK": 32,
#         "topP": 1.0
#     }
#     payload = {"contents": chat_history, "generationConfig": generation_config}

#     if response_schema:
#         payload["generationConfig"]["responseMimeType"] = "application/json"
#         payload["generationConfig"]["responseSchema"] = response_schema

#     api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    
#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.post(api_url, json=payload, timeout=120)
#             response.raise_for_status()
#             result = response.json()

#             if not result.get("candidates"):
#                 st.error("No response generated. Please try again.")
#                 return None

#             text_response = result["candidates"][0]["content"]["parts"][0].get("text", "")

#             if response_schema:
#                 parsed_response = safe_json_parse(text_response)
#                 if isinstance(parsed_response, dict) and "error" in parsed_response:
#                     st.error(f"Failed to parse response as JSON: {parsed_response['error']}")
#                     st.error("Raw response: " + text_response[:500] + ("..." if len(text_response) > 500 else ""))
#                     return None
#                 # Validate required fields
#                 if not all(key in parsed_response for key in ["title", "introduction", "modules", "conclusion"]):
#                     st.error("Generated course is missing required fields")
#                     return None
#                 return parsed_response
#             return text_response
#     except httpx.RequestError as e:
#         st.error(f"API request error: {str(e)}")
#         return None
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")
#         return None

# # --- Course Assistant Chat Function ---
# def process_chat_message(user_message, course_content):
#     """Process chat messages with context about the course"""
#     if not user_message:
#         return

#     try:
#         # Prepare context about the course
#         context = f"""You are a helpful course assistant. Use the following course content to answer questions:
#         {course_content}
        
#         Only answer questions related to this course content. If the question is not related, politely redirect the user
#         to ask course-related questions. If you encounter any errors, provide a clear explanation of what went wrong.
        
#         User question: {user_message}
#         """
        
#         response = st.session_state.conversation.predict(input=context)
#         st.session_state.chat_history.append({"user": user_message, "assistant": response})
#         return response
#     except Exception as e:
#         error_message = f"I apologize, but I encountered an error: {str(e)}"
#         st.error(error_message)
#         st.session_state.chat_history.append({"user": user_message, "assistant": error_message})
#         return error_message

# def generate_course_outline(prompt, schema):
#     """
#     Generate a course outline using the OpenAI API
#     """
#     try:
#         # Initialize the chat client
#         chat = ChatOpenAI(temperature=0.7)
        
#         # Create a structured output parser
#         parser = StructuredOutputParser.from_json_schema(schema)
#         format_instructions = parser.get_format_instructions()
        
#         # Combine prompt with format instructions
#         full_prompt = f"{prompt}\n\nOutput should be in the following format:\n{format_instructions}"
        
#         # Get response from the model
#         response = chat.predict(full_prompt)
        
#         # Parse the response into structured format
#         try:
#             course_outline = parser.parse(response)
#             return course_outline
#         except Exception as e:
#             print(f"Error parsing response: {e}")
#             return None
            
#     except Exception as e:
#         print(f"Error generating course outline: {e}")
#         return None

# def save_course_outline(topic, difficulty, outline):
#     """
#     Save the generated course outline to a file
#     """
#     try:
#         # Create output directory if it doesn't exist
#         if not os.path.exists("output"):
#             os.makedirs("output")
            
#         # Create a filename with timestamp
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"{topic.lower().replace(' ', '_')}_{timestamp}.json"
#         filepath = os.path.join("output", filename)
        
#         # Add metadata to the outline
#         outline_with_metadata = {
#             "metadata": {
#                 "topic": topic,
#                 "difficulty": difficulty,
#                 "generated_at": timestamp
#             },
#             "content": outline
#         }
        
#         # Save to file
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(outline_with_metadata, f, indent=4)
            
#         return filepath
#     except Exception as e:
#         print(f"Error saving course outline: {e}")
#         return None

# def run_app():
#     """Runs the course generator application."""
#     init_session_state()
    
#     st.title("🎓 Course Generator")
#     st.markdown("Generate custom course outlines, track your progress, and get detailed chapter content!")
    
#     # AI Generation Settings in sidebar
#     with st.sidebar:
#         st.header("⚙️ AI Generation Settings")
#         temperature = st.slider(
#             "Temperature",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.7,
#             step=0.01,
#             help="Controls the randomness of the output. Higher values mean more creative."
#         )
#         max_tokens = st.slider(
#             "Max Output Tokens",
#             min_value=50,
#             max_value=2048,
#             value=1024,
#             step=50,
#             help="Maximum number of tokens to generate in the response."
#         )
    
#     # Course generation form
#     with st.form(key="new_course_form"):
#         st.subheader("Generate New Course")
        
#         # Course inputs in columns
#         col1, col2 = st.columns(2)
#         with col1:
#             course_topic = st.text_input(
#                 "Course Topic", 
#                 key="new_course_topic",
#                 help="Enter the main topic for your course",
#                 placeholder="e.g., Machine Learning, Web Development"
#             )
#             num_modules = st.number_input(
#                 "Number of Modules", 
#                 min_value=1, 
#                 max_value=10,
#                 value=5, 
#                 key="new_course_modules",
#                 help="Choose how many modules you want"
#             )
#         with col2:
#             difficulty = st.selectbox(
#                 "Difficulty Level",
#                 ["Beginner", "Intermediate", "Advanced"],
#                 key="new_course_difficulty",
#                 help="Select the target audience level"
#             )
        
#         # Generate button with enhanced prominence
#         generate_button = st.form_submit_button(
#             "Generate New Course Outline",
#             type="primary",
#             use_container_width=True
#         )
#           # Course generation
#         if generate_button:
#             if not course_topic.strip():
#                 st.warning("⚠️ Please enter a course topic")
#                 return
            
#             with st.spinner(f"🔄 Generating {difficulty} level course on {course_topic}..."):
#                 try:
#                     # Course schema definition
#                     course_schema = {
#                         "type": "OBJECT",
#                         "properties": {
#                             "title": {
#                                 "type": "STRING",
#                                 "description": "The title of the course"
#                             },
#                             "introduction": {
#                                 "type": "STRING",
#                                 "description": "A comprehensive introduction to the course"
#                             },
#                             "prerequisites": {
#                                 "type": "ARRAY",
#                                 "items": {"type": "STRING"},
#                                 "description": "List of prerequisites for the course"
#                             },
#                             "modules": {
#                                 "type": "ARRAY",
#                                 "description": "List of course modules",
#                                 "items": {
#                                     "type": "OBJECT",
#                                     "properties": {
#                                         "moduleNumber": {
#                                             "type": "INTEGER",
#                                             "description": "The module number"
#                                         },
#                                         "moduleTitle": {
#                                             "type": "STRING",
#                                             "description": "The title of the module"
#                                         },
#                                         "moduleDescription": {
#                                             "type": "STRING",
#                                             "description": "Description of the module content"
#                                         },
#                                         "learningObjectives": {
#                                             "type": "ARRAY",
#                                             "items": {"type": "STRING"},
#                                             "description": "Learning objectives for this module"
#                                         },
#                                         "chapters": {
#                                             "type": "ARRAY",
#                                             "description": "List of chapters in this module",
#                                             "items": {
#                                                 "type": "OBJECT",
#                                                 "properties": {
#                                                     "chapterTitle": {
#                                                         "type": "STRING",
#                                                         "description": "Title of the chapter"
#                                                     },
#                                                     "content": {
#                                                         "type": "STRING",
#                                                         "description": "Main content of the chapter"
#                                                     },
#                                                     "keyPoints": {
#                                                         "type": "ARRAY",
#                                                         "items": {"type": "STRING"},
#                                                         "description": "Key learning points of the chapter"
#                                                     }
#                                                 },
#                                                 "required": ["chapterTitle", "content", "keyPoints"]
#                                             }
#                                         }
#                                     },
#                                     "required": ["moduleNumber", "moduleTitle", "moduleDescription", "learningObjectives", "chapters"]
#                                 }
#                             },
#                             "conclusion": {
#                                 "type": "STRING",
#                                 "description": "A concluding summary of the course"
#                             }
#                         },
#                         "required": ["title", "introduction", "prerequisites", "modules", "conclusion"]
#                     }

#                     # Generate course content                    # Create the course generation prompt
#                     course_prompt = (
#                         f"Generate a structured course on '{course_topic}' at {difficulty} level with {num_modules} modules.\n\n"
#                         "IMPORTANT: Return ONLY valid JSON matching the schema, with no additional text or formatting.\n\n"
#                         "If the output is too long, make module and chapter content shorter so the entire JSON fits in a single response.\n"
#                         "Do not truncate the JSON.\n"
#                         "Do not include any text before or after the JSON.\n\n"
#                         "Requirements:\n"
#                         "1. Title should be clear and descriptive\n"
#                         "2. Introduction should explain course value and prerequisites\n"
#                         "3. Each module must have:\n"
#                         "   - A descriptive title\n"
#                         "   - Clear learning objectives\n"
#                         "   - 2-3 chapters with detailed content\n"
#                         "   - Key points for each chapter\n"
#                         f"4. Content should be appropriate for {difficulty} level\n"
#                         "5. Ensure natural progression of concepts\n"
#                         "6. Include a meaningful conclusion\n\n"
#                         "Remember:\n"
#                         "- Return ONLY the JSON object\n"
#                         "- Follow the schema exactly\n"
#                         "- No markdown or formatting\n"
#                         "- All required fields must be included\n"
#                         "- Content should be practical and actionable"
#                     )

#                     loop = asyncio.new_event_loop()
#                     asyncio.set_event_loop(loop)
#                     course_data = loop.run_until_complete(
#                         generate_content_with_gemini(course_prompt, response_schema=course_schema)
#                     )
#                     loop.close()

#                     if course_data and isinstance(course_data, dict) and course_data.get("title"):
#                         # Initialize completion status for chapters
#                         completion_status = {}
#                         total_chapters = 0
                        
#                         for m_idx, module in enumerate(course_data.get("modules", [])):
#                             for c_idx, _ in enumerate(module.get("chapters", [])):
#                                 chapter_id = f"course_{len(st.session_state.courses)}_m{m_idx}_ch{c_idx}"
#                                 completion_status[chapter_id] = False
#                                 total_chapters += 1

#                         # Create new course with tracking data
#                         new_course = {
#                             **course_data,
#                             "completion_status": completion_status,
#                             "total_chapters": total_chapters,
#                             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                         }
                        
#                         st.session_state.courses.append(new_course)
#                         st.session_state.selected_course_index = len(st.session_state.courses) - 1
#                         st.success(f"✨ Successfully generated course: {new_course['title']}")
#                         st.rerun()
#                     else:
#                         st.error("Failed to generate a valid course outline. Please try again.")
#                 except Exception as e:
#                     st.error(f"Error generating course: {str(e)}")
#                     st.error("Please try again or contact support if the problem persists.")
#                 return
            
#             with st.spinner(f"Generating {difficulty} level course on {course_topic}..."):
#                 try:
#                     # Define the course generation prompt
#                     course_prompt = f"""Create a detailed course outline for '{course_topic}' at {difficulty} level with exactly {num_modules} modules.
#                     The course should be comprehensive, well-structured, and include:
#                     1. A clear title that accurately reflects the course topic
#                     2. A thorough introduction explaining the course value and objectives
#                     3. Exactly {num_modules} modules, each with:
#                        - A clear module title and description
#                        - 2-4 detailed chapters per module
#                        - Specific learning objectives
#                        - Practical examples and exercises
#                     4. Key points and takeaways for each chapter
#                     5. A concluding summary tying everything together
#                     6. Required prerequisites and recommended background

#                     Make the content detailed, practical, and appropriate for {difficulty} level learners.
#                     Ensure proper progression of concepts from basic to advanced within the course structure.
#                     Focus on making the content engaging and actionable.
#                     """

#                     # Define the schema for course structure
#                     course_schema = {
#                         "type": "object",
#                         "properties": {
#                             "title": {
#                                 "type": "string",
#                                 "description": "The title of the course"
#                             },
#                             "introduction": {
#                                 "type": "string",
#                                 "description": "A comprehensive introduction to the course"
#                             },
#                             "prerequisites": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "string"
#                                 },
#                                 "description": "List of prerequisites for the course"
#                             },
#                             "modules": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "object",
#                                     "properties": {
#                                         "moduleNumber": {
#                                             "type": "integer",
#                                             "description": "The module number"
#                                         },
#                                         "moduleTitle": {
#                                             "type": "string",
#                                             "description": "The title of the module"
#                                         },
#                                         "moduleDescription": {
#                                             "type": "string",
#                                             "description": "Description of the module content"
#                                         },
#                                         "chapters": {
#                                             "type": "array",
#                                             "items": {
#                                                 "type": "object",
#                                                 "properties": {
#                                                     "chapterTitle": {
#                                                         "type": "string",
#                                                         "description": "Title of the chapter"
#                                                     },
#                                                     "content": {
#                                                         "type": "string",
#                                                         "description": "Detailed chapter content"
#                                                     },
#                                                     "keyPoints": {
#                                                         "type": "array",
#                                                         "items": {
#                                                             "type": "string"
#                                                         },
#                                                         "description": "Key learning points from the chapter"
#                                                     }
#                                                 },
#                                                 "required": ["chapterTitle", "content", "keyPoints"]
#                                             }
#                                         },
#                                         "learningObjectives": {
#                                             "type": "array",
#                                             "items": {
#                                                 "type": "string"
#                                             },
#                                             "description": "Learning objectives for the module"
#                                         }
#                                     },
#                                     "required": ["moduleNumber", "moduleTitle", "moduleDescription", "chapters", "learningObjectives"]
#                                 }
#                             },
#                             "conclusion": {
#                                 "type": "string",
#                                 "description": "A concluding summary of the course"
#                             }
#                         },
#                         "required": ["title", "introduction", "prerequisites", "modules", "conclusion"]
#                     }

#                     # Generate course outline using the LLM
#                     try:
#                         course_outline = generate_course_outline(course_prompt, course_schema)
#                         if course_outline:
#                             # Save the course outline
#                             save_course_outline(course_topic, difficulty, course_outline)
#                             st.success("Course outline generated successfully!")
#                             st.write(course_outline)
#                         else:
#                             st.error("Failed to generate a valid course outline. Please try again.")
#                     except Exception as e:
#                         st.error(f"An error occurred while generating the course: {str(e)}")
#                         st.error("Failed to generate a valid course outline. Please try again.")
#                 except Exception as e:
#                     st.error(f"Error generating course: {str(e)}")
#                     st.error("Please try again or contact support if the problem persists.")
    
#     # Display selected course and chat interface
#     if st.session_state.selected_course_index is not None:
#         course = st.session_state.courses[st.session_state.selected_course_index]
#         # Only show the course title ONCE
#         st.subheader(f"📚 {course.get('title', 'Course Content')}")
        
#         # Course completion tracking
#         completed_chapters = sum(1 for status in course.get("completion_status", {}).values() if status)
#         total_chapters = course.get("total_chapters", 0)
#         completion_percentage = (completed_chapters / total_chapters) * 100 if total_chapters > 0 else 0
#         st.markdown("**Course Progress:**")
#         st.progress(completion_percentage / 100, text=f"{completion_percentage:.1f}% Completed ({completed_chapters}/{total_chapters} chapters)")
        
#         # Course introduction
#         st.markdown(f"**Introduction:** {course.get('introduction', '')}")
        
#         # Modules and Chapters
#         for m_idx, module in enumerate(course.get("modules", [])):
#             with st.expander(f"Module {module.get('moduleNumber', m_idx + 1)}: {module.get('moduleTitle', 'N/A')}", expanded=True):
#                 st.markdown(f"*{module.get('moduleDescription', '')}*")
#                 st.markdown("**Learning Objectives:**")
#                 for obj in module.get("learningObjectives", []):
#                     st.markdown(f"- {obj}")
#                 for c_idx, chapter in enumerate(module.get("chapters", [])):
#                     chapter_id = f"course_{st.session_state.selected_course_index}_m{m_idx}_ch{c_idx}"
#                     # Use a unique key for each checkbox
#                     completed = st.checkbox(
#                         f"{chapter['chapterTitle']}",
#                         value=course["completion_status"].get(chapter_id, False),
#                         key=f"check_{st.session_state.selected_course_index}_{m_idx}_{c_idx}"
#                     )
#                     if completed != course["completion_status"].get(chapter_id, False):
#                         course["completion_status"][chapter_id] = completed
#                         st.rerun()
                    
#                     # Generate detailed content button
#                     if st.button(f"Generate Detailed Content", key=f"gen_content_{chapter_id}"):
#                         with st.spinner("Generating detailed chapter content..."):
#                             content_prompt = f"Generate detailed content for chapter '{chapter['chapterTitle']}' in the {difficulty} level course on {course['title']}."
#                             loop = quiz_utils.get_or_create_eventloop()
#                             detailed_content = loop.run_until_complete(
#                                 generate_content_with_gemini(content_prompt, temperature, max_tokens, 32, 1.0)
#                             )
#                             if detailed_content:
#                                 st.session_state.chapter_contents[chapter_id] = detailed_content
                    
#                     # Display chapter content
#                     if chapter_id in st.session_state.chapter_contents:
#                         st.markdown(st.session_state.chapter_contents[chapter_id])
                    
#                     # Quiz section
#                     if st.button(f"Generate Quiz for {chapter['chapterTitle']}", key=f"quiz_{chapter_id}"):
#                         with st.spinner("Generating quiz questions..."):
#                             chapter_content = st.session_state.chapter_contents.get(chapter_id, chapter.get('content', ''))
#                             loop = quiz_utils.get_or_create_eventloop()
#                             quiz_data = loop.run_until_complete(
#                                 quiz_utils.generate_quiz_with_gemini(
#                                     chapter_content, API_KEY, temperature, max_tokens, 32, 1.0, 5
#                                 )
#                             )
#                             if quiz_data:
#                                 st.session_state.quiz_progress[chapter_id] = {
#                                     "questions": quiz_data["questions"],
#                                     "completed": False,
#                                     "current_question": 0,
#                                     "score": 0
#                                 }
                    
#                     # Display Quiz
#                     if chapter_id in st.session_state.quiz_progress:
#                         quiz = st.session_state.quiz_progress[chapter_id]
#                         st.markdown("### Chapter Quiz")
                        
#                         if not quiz["completed"]:
#                             question = quiz["questions"][quiz["current_question"]]
#                             answer = st.radio(
#                                 question["question"],
#                                 question["options"],
#                                 key=f"quiz_q_{chapter_id}_{quiz['current_question']}"
#                             )
                            
#                             col1, col2 = st.columns([1, 3])
#                             with col1:
#                                 if st.button("Submit Answer", key=f"submit_{chapter_id}"):
#                                     if answer == question["answer"]:
#                                         quiz["score"] += 1
#                                         st.success("Correct!")
#                                     else:
#                                         st.error(f"Incorrect. The answer was: {question['answer']}")
#                                     st.markdown(f"*Explanation: {question['explanation']}*")
                                    
#                                     quiz["current_question"] += 1
#                                     if quiz["current_question"] >= len(quiz["questions"]):
#                                         quiz["completed"] = True
#                                         st.success(f"Quiz completed! Score: {quiz['score']}/{len(quiz['questions'])}")
#                                     st.rerun()
                            
#                         else:
#                             st.success(f"Quiz completed! Final score: {quiz['score']}/{len(quiz['questions'])}")
#                             if st.button("Retake Quiz", key=f"retake_{chapter_id}"):
#                                 del st.session_state.quiz_progress[chapter_id]
#                                 st.rerun()        # Display course completion and content
#         if course.get("modules"):
#             # Display progress
#             st.markdown("### Course Progress")
#             st.progress(completion_percentage / 100, text=f"{completion_percentage:.1f}% Completed ({completed_chapters}/{total_chapters} chapters)")
            
#             # Display introduction
#             st.markdown("### Course Overview")
#             st.markdown(course.get("introduction", ""))
            
#             # Display modules
#             for mod_idx, module in enumerate(course.get("modules", [])):
#                 with st.expander(f"📚 Module {module.get('moduleNumber', mod_idx + 1)}: {module.get('moduleTitle', 'N/A')}", expanded=True):
#                     for chap_idx, chapter in enumerate(module.get("chapters", [])):
#                         # Unique ID for chapter checkbox and content keys
#                         chapter_unique_id = f"course_{st.session_state.selected_course_index}_m{mod_idx}_ch{chap_idx}"
                        
#                         is_completed = st.checkbox(
#                             f"**{chapter['chapterTitle']}**",
#                             value=course['completion_status'].get(chapter_unique_id, False),
#                             key=f"checkbox_{chapter_unique_id}"
#                         )
#                         # Update completion status if checkbox state changes
#                         if is_completed != course['completion_status'].get(chapter_unique_id, False):
#                             course['completion_status'][chapter_unique_id] = is_completed
#                             st.rerun() # Rerun to update the progress bar

#                         st.markdown(f"*{chapter.get('description', '')}*")
                        
#                         if st.button(f"Generate Content for '{chapter['chapterTitle']}'", key=f"gen_btn_{chapter_unique_id}"):
#                             with st.spinner("Generating detailed chapter content..."):
#                                 content_prompt = f"Elaborate in detail on the chapter '{chapter['chapterTitle']}' for the course '{course['courseTitle']}'. Description: {chapter['description']}. Provide comprehensive, readable content with examples if relevant."
#                                 loop = quiz_utils.get_or_create_eventloop()
#                                 generated_text = loop.run_until_complete(
#                                     generate_content_with_gemini(content_prompt)
#                                 )
#                                 if generated_text:
#                                     st.session_state.chapter_contents[chapter_unique_id] = generated_text
#                                 else:
#                                     st.error("Failed to generate content.")
                        
#                         if chapter_unique_id in st.session_state.chapter_contents:
#                             st.markdown("---")
#                             st.info(st.session_state.chapter_contents[chapter_unique_id])
#                             st.markdown("---") # Add a separator after content for readability

#                     # Quiz Logic for the module
#                     module_quiz_id = f"quiz_c{st.session_state.selected_course_index}_m{mod_idx}"
#                     if module_quiz_id not in st.session_state.quiz_progress:
#                         st.session_state.quiz_progress[module_quiz_id] = {}

#                     if st.button(f"Take Quiz for Module {module.get('moduleNumber', mod_idx + 1)}", key=f"quiz_btn_{module_quiz_id}"):
#                         with st.spinner("Generating quiz... This may take a moment."):
#                             module_content = "\n".join([f"{ch['chapterTitle']}: {ch['description']}" for ch in module['chapters']])
#                             # Ensure there's content to generate a quiz from
#                             if not module_content.strip():
#                                 st.warning("Cannot generate quiz: No content available for this module's chapters.")
#                             else:
#                                 loop = quiz_utils.get_or_create_eventloop()
#                                 quiz_data = loop.run_until_complete(quiz_utils.generate_quiz_with_gemini(
#                                     module_content, API_KEY, 0.5, 2048, 1, 1, 5
#                                 ))
#                                 if quiz_data and "questions" in quiz_data:
#                                     st.session_state.quiz_progress[module_quiz_id] = {
#                                         "questions": quiz_data["questions"],
#                                         "completed": False,
#                                         "answers": [None] * len(quiz_data["questions"]),
#                                         "score": 0
#                                     }
#                                     st.success("Quiz generated!")
#                                 else:
#                                     st.error("Failed to generate quiz. Please try again.")
                    
#                     quiz_obj = st.session_state.quiz_progress.get(module_quiz_id)
#                     if quiz_obj and quiz_obj.get("questions"):
#                         st.subheader(f"Module {module.get('moduleNumber', mod_idx + 1)} Quiz")
#                         if not quiz_obj.get("completed"):
#                             with st.form(f"quiz_form_{module_quiz_id}"):
#                                 # Use a list to store answers for the form submission
#                                 current_answers = []
#                                 for i, q in enumerate(quiz_obj["questions"]):
#                                     selected_option = st.radio(f"**Q{i+1}: {q['question']}**", q['options'], key=f"quiz_{module_quiz_id}_q{i}")
#                                     current_answers.append(selected_option)
                                
#                                 if st.form_submit_button("Submit Quiz", help="Submit your answers for this quiz."):
#                                     correct_answers = [q["answer"] for q in quiz_obj["questions"]]
#                                     quiz_utils.update_quiz_progress(st.session_state, module_quiz_id, current_answers, correct_answers)
#                                     st.rerun() # Rerun to display quiz results
#                         else:
#                             st.success(f"Quiz completed! Score: {quiz_obj['score']}/{len(quiz_obj['questions'])}")
#                             for i, q in enumerate(quiz_obj["questions"]):
#                                 st.markdown(f"**Q{i+1}: {q['question']}**")
#                                 user_ans = quiz_obj['answers'][i]
#                                 correct_ans = q['answer']
                                
#                                 # Display user's answer and correctness
#                                 if user_ans == correct_ans:
#                                     st.markdown(f"> ✅ Your answer: **{user_ans}** (Correct)")
#                                 else:
#                                     st.markdown(f"> ❌ Your answer: **{user_ans}** (Incorrect, Correct was: **{correct_ans}**)")
#                                 st.markdown(f"> *Explanation: {q['explanation']}*")
#                                 st.markdown("---") # Separator between questions
        
#         st.markdown(f"**Conclusion:** {course.get('conclusion', 'N/A')}")
#     elif st.session_state.selected_course_index is None:
#         st.info("Please generate a new course or select an existing one from the sidebar to view its content.")

#     # --- Chatbot Section ---
#     if st.session_state.selected_course_index is not None:
#         st.divider()
#         st.subheader("💬 Course Assistant")
#         st.write("Ask questions about the course content!")
        
#         # Display chat history
#         for message in st.session_state.chat_history:
#             with st.chat_message("user"):
#                 st.write(message["user"])
#             with st.chat_message("assistant"):
#                 st.write(message["assistant"])
        
#         # Chat input
#         if user_message := st.chat_input("Ask a question about the course..."):
#             # Get current course content
#             course = st.session_state.courses[st.session_state.selected_course_index]
#             course_content = json.dumps(course)  # Convert course data to string
            
#             with st.chat_message("user"):
#                 st.write(user_message)
            
#             with st.chat_message("assistant"):
#                 response = process_chat_message(user_message, course_content)
#                 st.write(response)

# course_generator.py (Updated)
# course_generator.py (Updated)
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
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
                placeholder="e.g., Machine Learning, Web Development"
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
                    st.error("Failed to generate a valid course outline. Please check the topic and try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred during course generation: {str(e)}")
                st.error("Please try again or contact support if the problem persists.")

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
                            st.error("Failed to generate detailed content.")
                
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
                            st.error("Failed to generate quiz for this module. Ensure enough content is available.")
            
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
                                index=options.index(quiz_obj['answers'][i]) if quiz_obj['answers'][i] in options else 0, # Pre-select if already answered
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

