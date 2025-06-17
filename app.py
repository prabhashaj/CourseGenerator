import streamlit as st
import json
import asyncio
from datetime import datetime
import base64
import os
from dotenv import load_dotenv
import httpx
import importlib
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader

quiz_utils = importlib.import_module("quiz_utils")

# Load environment variables
load_dotenv()

# --- Configuration ---
st.set_page_config(
    page_title="Course Creator & Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Key Configuration ---
API_KEY = "AIzaSyDHyPyFtNn0TUaGcVgHPlTU1JgG17sIF_A"

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Course Creator & Learning Assistant")
st.markdown("Generate custom courses, track progress, get detailed content, and interact with your learning materials!")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "courses" not in st.session_state:
    st.session_state.courses = []
if "selected_course_index" not in st.session_state:
    st.session_state.selected_course_index = None
if "chapter_contents" not in st.session_state:
    st.session_state.chapter_contents = {}
if "quiz_progress" not in st.session_state:
    st.session_state.quiz_progress = {}
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Gemini API Call Function ---
async def generate_content_with_gemini(prompt, temperature, max_tokens, top_k, top_p, response_schema=None):
    """
    Calls the Gemini API to generate content with specified parameters.
    """
    # Prepare the API request payload
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            "topK": int(top_k),
            "topP": float(top_p)
        }
    }    # Add schema if provided
    if response_schema:
        payload["generationConfig"].update({
            "candidateCount": 1,
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        })
        
    if not validate_api_key(API_KEY):
        st.error("API key validation failed. Please check your API key configuration.")
        return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            st.write("Sending request with payload:", payload)  # Debug output
            response = await client.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=120
            )
            if response.status_code != 200:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
            result = response.json()
            st.write("Received response:", result)  # Debug output
            
            if result.get("candidates") and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if candidate.get("content") and candidate["content"].get("parts"):
                    text_response = candidate["content"]["parts"][0].get("text", "")
                    if response_schema:
                        try:
                            return json.loads(text_response)
                        except json.JSONDecodeError as e:
                            st.error(f"Failed to parse JSON response. Error: {e}")
                            st.code(text_response)  # Show the raw response for debugging
                            return None
                    return text_response
            
            st.error(f"Unexpected response structure from API")
            st.write(result)  # Show the full response for debugging
            return None
            
    except httpx.RequestError as e:
        st.error(f"Network or API request error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Utility Functions ---
def validate_api_key(api_key):
    """Validate the API key before making requests"""
    if not api_key:
        st.error("API key is missing")
        return False
    if not isinstance(api_key, str):
        st.error("API key must be a string")
        return False
    if not api_key.startswith("AIza"):
        st.error("Invalid API key format")
        return False
    return True

# --- Helper function to get or create an asyncio loop ---
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# --- Sidebar for Configuration and Course Inputs ---
with st.sidebar:
    st.header("⚙️ AI Generation Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="Controls the randomness of the output. Higher values mean more creative."
    )
    max_tokens = st.slider(
        "Max Output Tokens",
        min_value=50,
        max_value=2048,
        value=1024,
        step=50,
        help="Maximum number of tokens to generate in the response."
    )
    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=40,
        value=32,
        step=1,
        help="Considers the top K most likely next tokens."
    )
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help="Considers the smallest set of tokens whose cumulative probability exceeds Top P."
    )

    st.header("📚 New Course Details")
    course_topic = st.text_input("Course Topic", "Introduction to Quantum Computing", key="new_course_topic_input")
    num_modules = st.number_input(
        "Number of Modules",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many modules/chapters should the course have?",
        key="new_course_modules_input"
    )
    difficulty_level = st.selectbox(
        "Difficulty Level",
        ["Beginner", "Intermediate", "Advanced", "Expert"],
        index=0,
        help="Select the target audience's expertise level.",
        key="new_course_difficulty_input"
    )
    read_time_per_module = st.radio(
        "Approx. Read Time per Module",
        ["2 minutes", "5 minutes", "10 minutes"],
        index=1,
        horizontal=True,
        help="Estimate the reading time for each module.",
        key="new_course_read_time_input"
    )

    generate_course_button = st.button("Generate New Course Outline", use_container_width=True, type="primary")

    st.markdown("---")
    st.header("My Saved Courses")
    if st.session_state.courses:
        course_titles = [f"{i+1}. {course['courseTitle']}" for i, course in enumerate(st.session_state.courses)]
        # Use a unique key for the radio button
        selected_course_display = st.radio(
            "Select a Course to View",
            course_titles,
            index=st.session_state.selected_course_index if st.session_state.selected_course_index is not None else 0,
            key="course_selector_radio"
        )
        # Update selected_course_index based on user selection
        st.session_state.selected_course_index = course_titles.index(selected_course_display)
    else:
        st.info("No courses generated yet. Generate one above!")

# --- Course Generation Logic ---
if generate_course_button:
    if not course_topic:
        st.error("Please enter a course topic to generate an outline.")
    else:
        # Log the input parameters
        input_params = {
            "course_topic": course_topic,
            "difficulty_level": difficulty_level,
            "num_modules": num_modules,
            "read_time_per_module": read_time_per_module,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p
        }
        print("\n=== Course Generation Input Parameters ===")
        print(json.dumps(input_params, indent=2))
        print("=====================================")
        
        st.session_state.messages.append({"role": "user", "content": f"Request: Generate a {difficulty_level} course outline on '{course_topic}' with {num_modules} modules, each taking approx. {read_time_per_module} to read."})

        # Define the JSON schema for the expected course outline
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
        The course should be designed for a '{difficulty_level}' level audience.
        It must have exactly {num_modules} modules.
        Each module should have chapters, and the content for each module should be designed to take approximately {read_time_per_module} to read.

        The JSON output should strictly follow this schema:
        {{
            "courseTitle": "string",
            "introduction": "string",
            "modules": [
                {{
                    "moduleNumber": integer,
                    "moduleTitle": "string",
                    "chapters": [
                        {{
                            "chapterTitle": "string",
                            "description": "string"
                        }}
                    ]
                }}
            ],
            "conclusion": "string"        }}
        Ensure the JSON is valid and complete. Do not include any additional text outside the JSON.
        """
        with st.spinner("Generating new course outline..."):
            loop = get_or_create_eventloop()
            course_data = loop.run_until_complete(
                generate_content_with_gemini(
                    course_prompt,
                    temperature,
                    max_tokens,
                    top_k,
                    top_p,
                    response_schema=course_schema
                )
            )
            if course_data and isinstance(course_data, dict) and "courseTitle" in course_data:
                # Log the generated course data
                print("\n=== Generated Course Data ===")
                print(json.dumps(course_data, indent=2))
                print("============================")
                
                # Save the input and output to JSONL file
                dataset_entry = {
                    "timestamp": str(datetime.now()),
                    "input": input_params,
                    "output": course_data
                }
                
                # Append to JSONL file
                with open("my_dataset.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(dataset_entry) + "\n")
                
                # Initialize completion status for the new course
                completion_status = {}
                for module in course_data.get("modules", []):
                    for chapter in module.get("chapters", []):
                        # Create a unique ID for each chapter
                        chapter_id = f"course_{len(st.session_state.courses)}_module_{module['moduleNumber']}_chapter_{chapter['chapterTitle'].replace(' ', '_').replace('.', '').replace(',', '')}"
                        completion_status[chapter_id] = False # Mark as not completed initially

                new_course_entry = {
                    "courseTitle": course_data["courseTitle"],
                    "introduction": course_data["introduction"],
                    "modules": course_data["modules"],
                    "conclusion": course_data["conclusion"],
                    "completion_status": completion_status # Add completion tracker
                }
                st.session_state.courses.append(new_course_entry)
                st.session_state.selected_course_index = len(st.session_state.courses) - 1 # Select the newly created course
                st.session_state.messages.append({"role": "assistant", "content": f"Course '{course_data['courseTitle']}' generated and added to your list!"})
                st.rerun() # Rerun to update the sidebar and main display
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Failed to generate course outline. Please check the error message above."})

# --- Display Selected Course Outline and Progress ---
if st.session_state.selected_course_index is not None and st.session_state.selected_course_index < len(st.session_state.courses):
    current_course = st.session_state.courses[st.session_state.selected_course_index]

    st.subheader(f"Viewing Course: {current_course['courseTitle']}")

    # Calculate and display progress
    total_chapters = 0
    completed_chapters = 0
    total_quizzes = 0
    completed_quizzes = 0
    for module in current_course.get("modules", []):
        for chapter in module.get("chapters", []):
            total_chapters += 1
            chapter_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_chapter_{chapter['chapterTitle'].replace(' ', '_').replace('.', '').replace(',', '')}"
            if current_course['completion_status'].get(chapter_id, False):
                completed_chapters += 1
        module_quiz_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_quiz"
        if module_quiz_id in st.session_state.quiz_progress and st.session_state.quiz_progress[module_quiz_id].get("completed", False):
            total_quizzes += 1
            completed_quizzes += 1
        elif module_quiz_id in st.session_state.quiz_progress and st.session_state.quiz_progress[module_quiz_id].get("questions"):
            total_quizzes += 1
    total_items = total_chapters + total_quizzes
    completed_items = completed_chapters + completed_quizzes
    progress_percentage = (completed_items / total_items) * 100 if total_items > 0 else 0
    st.progress(progress_percentage / 100, text=f"Course Progress: {progress_percentage:.0f}% ({completed_items}/{total_items} items completed, including quizzes)")

    st.markdown(f"**Introduction:** {current_course['introduction']}")

    for module in current_course.get("modules", []):
        st.markdown(f"### Module {module['moduleNumber']}: {module['moduleTitle']}")
        for chapter in module.get("chapters", []):
            # Re-create chapter_id consistently
            chapter_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_chapter_{chapter['chapterTitle'].replace(' ', '_').replace('.', '').replace(',', '')}"

            # Checkbox for completion
            is_completed = st.checkbox(
                f"Mark as complete: **{chapter['chapterTitle']}**",
                value=current_course['completion_status'].get(chapter_id, False),
                key=f"checkbox_{chapter_id}" # Unique key for each checkbox
            )

            # Update completion status in session state if changed
            if is_completed != current_course['completion_status'].get(chapter_id, False):
                current_course['completion_status'][chapter_id] = is_completed
                st.session_state.courses[st.session_state.selected_course_index] = current_course # Update the course in session state
                st.rerun() # Rerun to update the progress bar immediately

            with st.expander(f"Chapter Details: {chapter['chapterTitle']}", expanded=False):
                st.markdown(f"**Description:** {chapter['description']}")
                # --- Custom Red Button Style for Chapter Content and Quiz ---
                st.markdown("""
                    <style>
                    .red-btn button {
                        background-color: #e53935 !important;
                        color: #fff !important;
                        font-weight: bold !important;
                        border-radius: 8px !important;
                        border: 2px solid #e53935 !important;
                        box-shadow: 0 2px 8px rgba(229,57,53,0.15);
                        margin-bottom: 10px;
`                        transition: background 0.2s, color 0.2s;
                    }
                    .red-btn button:hover {
                        background-color: #b71c1c !important;
                        border-color: #b71c1c !important;
                        color: #fff !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                # --- Red Button for Generate Detailed Content ---
                with st.container():
                    gen_content_btn = st.button(f"Generate Detailed Content for '{chapter['chapterTitle']}'", key=f"gen_content_btn_{chapter_id}", type="secondary")
                    st.markdown('<style>div[data-testid="stButton"] button {background-color: #e53935 !important; color: #fff !important; font-weight: bold !important; border-radius: 8px !important; border: 2px solid #e53935 !important;}</style>', unsafe_allow_html=True)
                if gen_content_btn:
                    with st.spinner(f"Generating detailed content for '{chapter['chapterTitle']}'..."):
                        chapter_content_prompt = f"""
                        Elaborate in detail on the following chapter from a course titled '{current_course['courseTitle']}' (Difficulty: {difficulty_level}, Read Time: {read_time_per_module}):
                        Module: {module['moduleTitle']}
                        Chapter: {chapter['chapterTitle']}
                        Description: {chapter['description']}

                        Provide a comprehensive explanation, aiming for 3-5 paragraphs of good knowledge. Include examples if relevant.Don't just only create paragraphs but make the content more appealing and readable.
                        """
                        loop = get_or_create_eventloop()
                        generated_chapter_text = loop.run_until_complete(
                            generate_content_with_gemini(
                                chapter_content_prompt,
                                temperature,
                                max_tokens,
                                top_k,
                                top_p,
                                response_schema=None # No schema for free-form text
                            )
                        )
                        if generated_chapter_text:
                            st.session_state.chapter_contents[chapter_id] = generated_chapter_text
                            st.success(f"Content for '{chapter['chapterTitle']}' generated!")
                        else:
                            st.error(f"Failed to generate content for '{chapter['chapterTitle']}'.")
                
                # Display generated content if available
                if chapter_id in st.session_state.chapter_contents:
                    st.markdown("---")
                    st.markdown(st.session_state.chapter_contents[chapter_id])
                else:
                    st.info("Click 'Generate Detailed Content' to get more information for this chapter.")
        # --- Quiz for this module (after chapters) ---
        module_quiz_id = f"course_{st.session_state.selected_course_index}_module_{module['moduleNumber']}_quiz"
        if module_quiz_id not in st.session_state.quiz_progress:
            st.session_state.quiz_progress[module_quiz_id] = {"completed": False, "score": 0, "answers": []}
        # --- Red Button for Take Quiz ---
        with st.container():
            take_quiz_btn = st.button(f"Take Quiz for Module {module['moduleNumber']} ({module['moduleTitle']})", key=f"quiz_btn_{module_quiz_id}", type="secondary")
            st.markdown('<style>div[data-testid="stButton"] button {background-color: #e53935 !important; color: #fff !important; font-weight: bold !important; border-radius: 8px !important; border: 2px solid #e53935 !important;}</style>', unsafe_allow_html=True)
        if take_quiz_btn:
            module_content = "\n".join([chapter['description'] for chapter in module.get('chapters', [])])
            quiz_prompt = f"Module: {module['moduleTitle']}\n{module_content}"
            with st.spinner(f"Generating quiz for Module {module['moduleNumber']}..."):
                loop = quiz_utils.get_or_create_eventloop()
                quiz_data = loop.run_until_complete(
                    quiz_utils.generate_quiz_with_gemini(
                        quiz_prompt,
                        API_KEY,
                        temperature,
                        max_tokens,
                        top_k,
                        top_p,
                        num_questions=5
                    )
                )
                if quiz_data and "questions" in quiz_data:
                    st.session_state.quiz_progress[module_quiz_id]["questions"] = quiz_data["questions"]
                    st.session_state.quiz_progress[module_quiz_id]["completed"] = False
                    st.session_state.quiz_progress[module_quiz_id]["answers"] = [None] * len(quiz_data["questions"])
                    st.success("Quiz generated! Scroll down to attempt it.")
                else:
                    st.error("Failed to generate quiz for this module.")
        # Display quiz if available (immediately after module)
        quiz_obj = st.session_state.quiz_progress.get(module_quiz_id, {})
        if quiz_obj.get("questions"):
            st.markdown(f"#### Quiz for Module {module['moduleNumber']} ({module['moduleTitle']})")
            answers = quiz_obj.get("answers", [None]*len(quiz_obj["questions"]))
            submitted = False
            with st.form(f"quiz_form_{module_quiz_id}"):
                for idx, q in enumerate(quiz_obj["questions"]):
                    st.markdown(f"**Q{idx+1}: {q['question']}**")
                    options = q["options"]
                    answers[idx] = st.radio(
                        f"Select answer for Q{idx+1}",
                        options,
                        index=options.index(answers[idx]) if answers[idx] in options else 0,
                        key=f"quiz_{module_quiz_id}_q{idx}"
                    )
                submitted = st.form_submit_button("Submit Quiz")
            if submitted and not quiz_obj.get("completed", False):
                correct_answers = [q["answer"] for q in quiz_obj["questions"]]
                score = quiz_utils.update_quiz_progress(st.session_state, module_quiz_id, answers, correct_answers)
                st.success(f"Quiz submitted! Your score: {score}/{len(correct_answers)}")
                for idx, q in enumerate(quiz_obj["questions"]):
                    st.markdown(f"**Q{idx+1} Explanation:** {q['explanation']}")
            elif quiz_obj.get("completed", False):
                st.info(f"Quiz already completed. Score: {quiz_obj['score']}/{len(quiz_obj['questions'])}")
                for idx, q in enumerate(quiz_obj["questions"]):
                    st.markdown(f"**Q{idx+1} Explanation:** {q['explanation']}")
    st.markdown(f"**Conclusion:** {current_course['conclusion']}")
else:
    st.info("Select a course from the sidebar or generate a new one to get started!")


# --- Display Chat Messages (for general chatbot interaction) ---
st.subheader("Chat History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input for general questions ---
st.subheader("Ask a general question (optional)")
if prompt := st.chat_input("What else can I help you with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        # For general chat, we don't need a specific schema
        loop = get_or_create_eventloop()
        response = loop.run_until_complete(
            generate_content_with_gemini(
                prompt,
                temperature,
                max_tokens,
                top_k,
                top_p
            )
        )
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- RAG Helper Functions ---
@st.cache_resource
def get_vector_store(file_paths):
    """
    Loads documents, splits them, creates embeddings using Gemini's embedding model,
    and stores them in a FAISS vector store.
    """
    all_documents = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file_extension}. Skipping {os.path.basename(file_path)}")
                continue
            all_documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {os.path.basename(file_path)}: {e}")
            continue

    if not all_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(all_documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    return vector_store

@st.cache_resource
def get_conversational_chain(_vector_store):
    """
    Initializes and returns a ConversationalRetrievalChain using Gemini's chat model.
    """
    if not _vector_store:
        return None

    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": None}
    )
    return conversation_chain

def handle_user_input(user_question):
    """
    Processes the user's question and displays the response.
    """
    if st.session_state.conversation and user_question:
        with st.spinner("Generating response..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)
    elif not st.session_state.conversation:
        st.error("Please upload and process documents first.")

# --- Content Generation Functions ---
async def generate_course_content(topic, chapter_title):
    """Generate detailed content for a specific chapter"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    
    prompt = f"""Create detailed content for the chapter '{chapter_title}' in the course about '{topic}'.
    Include:
    1. Comprehensive explanation of concepts
    2. Real-world examples
    3. Code snippets or practical demonstrations where applicable
    4. Best practices and common pitfalls
    5. Summary of key points
    
    Format the content with proper Markdown headings and formatting."""
    
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        st.error(f"Error generating chapter content: {str(e)}")
        return None

async def generate_quiz(topic, chapter_content):
    """Generate a quiz based on chapter content"""
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.7)
    
    quiz_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "correct_answer": {"type": "integer"},
                        "explanation": {"type": "string"}
                    }
                }
            }
        }
    }
    
    prompt = f"""Create a quiz to test understanding of: {topic}
    Based on this content: {chapter_content}
    Include 5 multiple-choice questions with explanations for the correct answers."""
    
    try:
        response = await generate_content_with_gemini(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            top_k=40,
            top_p=0.95,
            response_schema=quiz_schema
        )
        return response
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return None

async def generate_course_outline(topic, num_modules, difficulty="intermediate"):
    """Generate a complete course structure"""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "duration": {"type": "string"},
            "prerequisites": {
                "type": "array",
                "items": {"type": "string"}
            },
            "learning_objectives": {
                "type": "array",
                "items": {"type": "string"}
            },
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "key_topics": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "exercises": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["title", "description", "key_topics", "exercises"]
                }
            }
        },
        "required": ["title", "description", "prerequisites", "learning_objectives", "chapters"]
    }
    
    prompt = f"""Generate a detailed course outline for the topic: {topic}
    Number of modules/chapters: {num_modules}
    Difficulty level: {difficulty}
    
    Please provide:
    1. A clear course title
    2. A comprehensive description
    3. List of prerequisites
    4. Specific learning objectives
    5. {num_modules} chapters/modules, each with:
       - Title
       - Description
       - Key topics to cover
       - Practical exercises
    
    Make the content engaging and focused on practical applications."""
    
    try:
        response = await generate_content_with_gemini(
            prompt=prompt,
            temperature=0.7,
            max_tokens=2048,
            top_k=40,
            top_p=0.95,
            response_schema=schema
        )
        
        if not response:
            st.error("Failed to generate course content")
            return None
            
        # Add metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "id": f"{topic.lower().replace(' ', '_')}_{timestamp}",
            "created_at": timestamp,
            "progress": 0,
            "completed_chapters": []
        }
        # Combine the course data with metadata
        final_data = {**response, **metadata}
        return final_data
            
    except Exception as e:
        st.error(f"Error generating course outline: {str(e)}")
        st.error("Raw response: " + str(response) if response else "No response received")
        return None

def main():
    tabs = st.tabs(["Course Creator", "Learning Assistant", "Study Progress"])
    
    with tabs[0]:
        st.header("🎓 Course Creator")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            course_topic = st.text_input("Enter Course Topic:", 
                placeholder="e.g., Advanced Machine Learning, Web Development...")
            num_modules = st.number_input("Number of Modules:", 
                min_value=1, max_value=10, value=5)
                
            generate_button = st.button("Generate Course", type="primary")
            if generate_button and course_topic:
                with st.spinner("Creating your custom course..."):
                    loop = get_or_create_eventloop()
                    course_data = loop.run_until_complete(
                        generate_course_outline(
                            topic=course_topic,
                            num_modules=num_modules
                        )
                    )
                    
                    if course_data:
                        st.session_state.courses.append(course_data)
                        st.success(f"Course '{course_data['title']}' generated successfully!")
                        
                        st.subheader("📚 Generated Course")
                        st.markdown(f"### {course_data['title']}")
                        st.markdown(course_data['description'])
                        
                        st.markdown("### Prerequisites")
                        for prereq in course_data['prerequisites']:
                            st.markdown(f"- {prereq}")
                        
                        st.markdown("### Learning Objectives")
                        for obj in course_data['learning_objectives']:
                            st.markdown(f"- {obj}")
                        
                        st.markdown("### Chapters")
                        for i, chapter in enumerate(course_data['chapters'], 1):
                            with st.expander(f"Chapter {i}: {chapter['title']}"):
                                st.markdown(chapter['description'])
                                st.markdown("#### Key Topics")
                                for topic in chapter['key_topics']:
                                    st.markdown(f"- {topic}")
                                st.markdown("#### Exercises")
                                for exercise in chapter['exercises']:
                                    st.markdown(f"- {exercise}")
                    else:
                        st.error("Failed to generate course. Please try again.")
            elif generate_button:
                st.warning("Please enter a course topic")
        
        with col2:
            st.markdown("""
            ### Tips for Best Results
            - Be specific with your course topic
            - Consider your target audience
            - Specify realistic duration
            - Include practical examples
            """)
    
    # Tab 2: Learning Assistant (RAG)
    with tabs[1]:
        st.header("📚 Learning Assistant")
        
        # File uploader section
        with st.expander("Upload Learning Materials", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files to enable intelligent Q&A",
                type=["pdf", "txt"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Process Documents"):
                    with st.spinner("Processing documents..."):
                        # Save uploaded files temporarily
                        temp_paths = []
                        for file in uploaded_files:
                            temp_path = f"temp_{file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(file.getvalue())
                            temp_paths.append(temp_path)
                        
                        # Create vector store
                        st.session_state.vector_store = get_vector_store(temp_paths)
                        
                        # Clean up temp files
                        for path in temp_paths:
                            os.remove(path)
                        
                        if st.session_state.vector_store:
                            st.session_state.conversation = get_conversational_chain(st.session_state.vector_store)
                            st.success("Documents processed successfully! You can now ask questions.")
        
        # Chat interface
        st.markdown("### Ask Questions About Your Learning Materials")
        user_question = st.text_input("Your question:")
        if user_question:
            handle_user_input(user_question)
    
    # Tab 3: Study Progress
    with tabs[2]:
        st.header("📊 Study Progress")
        
        if st.session_state.courses:
            course_names = [course["title"] for course in st.session_state.courses]
            selected_course = st.selectbox("Select Course:", course_names)
            
            if selected_course:
                course_idx = course_names.index(selected_course)
                course = st.session_state.courses[course_idx]
                
                # Display progress metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chapters Completed", "0/{}".format(len(course["chapters"])))
                with col2:
                    st.metric("Time Spent", "0 hours")
                with col3:
                    st.metric("Quiz Score", "N/A")
                
                # Display chapters with progress
                st.markdown("### Chapter Progress")
                for i, chapter in enumerate(course["chapters"]):
                    with st.expander(f"Chapter {i+1}: {chapter['title']}", expanded=False):
                        st.markdown(chapter["description"])
                        if st.button(f"Mark Chapter {i+1} as Complete", key=f"complete_{i}"):
                            st.success(f"Chapter {i+1} marked as complete!")
        else:
            st.info("No courses created yet. Start by creating a course in the Course Creator tab!")

if __name__ == "__main__":
    main()

