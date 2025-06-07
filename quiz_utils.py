import asyncio
import httpx
import streamlit as st
import json

# Utility to generate quiz questions for a module or chapter
async def generate_quiz_with_gemini(prompt, api_key, temperature, max_tokens, top_k, top_p, num_questions=5):
    quiz_schema = {
        "type": "OBJECT",
        "properties": {
            "questions": {
                "type": "ARRAY",
                "description": "A list of quiz questions.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {"type": "STRING", "description": "The quiz question."},
                        "options": {
                            "type": "ARRAY",
                            "description": "List of answer options.",
                            "items": {"type": "STRING"}
                        },
                        "answer": {"type": "STRING", "description": "The correct answer from the options."},
                        "explanation": {"type": "STRING", "description": "Explanation for the correct answer."}
                    },
                    "required": ["question", "options", "answer", "explanation"]
                }
            }
        },
        "required": ["questions"]
    }

    quiz_prompt = f"""
    Create a quiz of {num_questions} multiple-choice questions based on the following content. Each question should have 4 options, one correct answer, and a brief explanation. Return only valid JSON as per the schema.
    Content:
    {prompt}
    """

    chat_history = [{"role": "user", "parts": [{"text": quiz_prompt}]}]
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topK": int(top_k),
        "topP": top_p,
        "responseMimeType": "application/json",
        "responseSchema": quiz_schema
    }
    payload = {
        "contents": chat_history,
        "generationConfig": generation_config
    }
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("text"):
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            try:
                return json.loads(text_response)
            except json.JSONDecodeError:
                st.error("Failed to parse quiz JSON response.")
                st.json(text_response)
                return None
        else:
            st.error(f"Quiz LLM response structure unexpected: {result}")
            return None
    except httpx.RequestError as e:
        st.error(f"Quiz API request error: {e}")
        return None
    except Exception as e:
        st.error(f"Quiz generation error: {e}")
        return None

# Helper to get or create event loop (for Streamlit compatibility)
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Quiz progress tracker helpers
def initialize_quiz_progress(session_state, quiz_id, num_questions):
    if "quiz_progress" not in session_state:
        session_state.quiz_progress = {}
    if quiz_id not in session_state.quiz_progress:
        session_state.quiz_progress[quiz_id] = {
            "completed": False,
            "score": 0,
            "answers": [None] * num_questions
        }

def update_quiz_progress(session_state, quiz_id, answers, correct_answers):
    score = sum([a == c for a, c in zip(answers, correct_answers)])
    session_state.quiz_progress[quiz_id]["score"] = score
    session_state.quiz_progress[quiz_id]["completed"] = True
    session_state.quiz_progress[quiz_id]["answers"] = answers
    return score
