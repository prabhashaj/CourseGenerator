import streamlit as st
import json
import asyncio # Required for async fetch
import base64 # Required for encoding image data if needed, though not directly used for text generation
import os # For environment variables, if API key was stored there
import httpx # Using httpx for async requests
import importlib
quiz_utils = importlib.import_module("quiz_utils")

# --- Configuration ---
# Set page configuration for the Streamlit app
st.set_page_config(
    page_title="Course Creator & Tracker",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Course Creator & Tracker")
st.markdown("Generate custom course outlines, track your progress, and get detailed chapter content!")

# --- API Key (from Streamlit secrets) ---
API_KEY = st.secrets["GEMINI_API_KEY"]  # Use Streamlit secrets for deployment

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "courses" not in st.session_state:
    st.session_state.courses = [] # List to store all generated courses
if "selected_course_index" not in st.session_state:
    st.session_state.selected_course_index = None # Index of the currently viewed course
if "chapter_contents" not in st.session_state:
    st.session_state.chapter_contents = {} # Stores generated content for chapters, keyed by chapter_id
if "quiz_progress" not in st.session_state:
    st.session_state.quiz_progress = {}

# --- Gemini API Call Function ---
async def generate_content_with_gemini(prompt, temperature, max_tokens, top_k, top_p, response_schema=None):
    """
    Calls the Gemini API to generate content with specified parameters.
    """
    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]

    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topK": int(top_k), # Ensure integer
        "topP": top_p
    }

    payload = {
        "contents": chat_history,
        "generationConfig": generation_config
    }

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
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("text"):
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            if response_schema:
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON response from LLM. Please try again. Raw response might be:")
                    st.json(text_response) # Show raw response for debugging
                    return None
            return text_response
        else:
            st.error(f"LLM response structure unexpected: {result}")
            return "Error: Could not get a valid response from the AI model."
    except httpx.RequestError as e:
        st.error(f"Network or API request error: {e}")
        return "Error: Failed to connect to the AI model. Please check your network or API key."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "Error: An unexpected issue occurred during AI generation."

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
            "conclusion": "string"
        }}
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
                
                # Button to generate content for this specific chapter
                if st.button(f"Generate Detailed Content for '{chapter['chapterTitle']}'", key=f"gen_content_btn_{chapter_id}"):
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
        if st.button(f"Take Quiz for Module {module['moduleNumber']} ({module['moduleTitle']})", key=f"quiz_btn_{module_quiz_id}"):
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

