import streamlit as st
from typing import Dict, List
import json
import random
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# --- API Key Setup ---
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("API Key is not configured. Please set it in Streamlit secrets or environment variables.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = API_KEY

def init_session_state():
    """Initialize session state for score tracking"""
    if "correct_count" not in st.session_state:
        st.session_state.correct_count = 0
    if "incorrect_count" not in st.session_state:
        st.session_state.incorrect_count = 0
    if "cards_answered" not in st.session_state:
        st.session_state.cards_answered = set()

def init_llm():
    """Initialize the Gemini 2.0 Flash model"""
    if "llm" not in st.session_state:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            top_k=40,
            top_p=0.8,
            max_output_tokens=2048
        )
        memory = ConversationBufferMemory()
        st.session_state.conversation = ConversationChain(llm=llm, memory=memory)
    return st.session_state.conversation

def generate_flashcards(topic: str, num_cards: int) -> List[Dict]:
    """Generate flashcards using Gemini 2.0 Flash"""
    conversation = init_llm()
    
    prompt = f"""Generate {num_cards} educational flashcards for studying {topic}.
    Each flashcard should be clear, concise, and test the user's knowledge.
    
    Requirements:
    1. Front: A clear question about {topic}
    2. Back: A concise but informative answer
    3. Difficulty: Specify as 'easy', 'medium', or 'hard'
    
    Return the flashcards in this JSON format:
    [
        {{
            "front": "question",
            "back": "answer",
            "difficulty": "level"
        }}
    ]
    
    Return ONLY the JSON array, no other text."""
    
    try:
        response = conversation.predict(input=prompt)
        # Extract JSON from response
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            # Sanitize invalid backslashes before parsing
            import re
            def fix_json_escapes(s):
                # Replace single backslashes not followed by valid escape chars with double backslash
                return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
            json_str = fix_json_escapes(json_str)
            return json.loads(json_str)
        else:
            st.error("Could not find valid JSON in the response")
            return []
    except Exception as e:
        st.error(f"Error generating flashcards: {str(e)}")
        return []

def display_flashcard(flashcard: Dict, index: int, total: int):
    """Display a single flashcard with enhanced UI and score tracking"""
    init_session_state()

    # Enhanced CSS for flashcard design
    st.markdown("""
        <style>
        .score-value.correct {
            color: #28a745;
        }
        .score-value.incorrect {
            color: #dc3545;
        }
        .final-score {
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            margin-top: 20px;
        }
        .score-percent {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .flashcard-container {
            padding: 20px;
            margin: 20px auto;
            max-width: 800px;
        }
        .card-header {
            padding: 10px 0;
            margin-bottom: 20px;
            text-align: center;
        }
        .progress-text {
            font-size: 1.2em;
            color: #333;
        }
        .flashcard {
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.12);
            padding: 36px 24px;
            min-height: 250px;
            position: relative;
            margin-bottom: 20px;
            transition: box-shadow 0.2s;
        }
        .flashcard:hover {
            box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        }
        .card-content {
            font-size: 1.4em;
            text-align: center;
            padding: 20px;
            margin: 20px 0;
        }
        .card-type {
            position: absolute;
            top: 20px;
            left: 20px;
            background-color: #f0f0f0;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
        }
        .difficulty-badge {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
            color: white;
        }
        .difficulty-easy {
            background-color: #28a745;
        }
        .difficulty-medium {
            background-color: #ffc107;
            color: #333;
        }
        .difficulty-hard {
            background-color: #dc3545;
        }
        .score-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .score-box {
            text-align: center;
            padding: 10px;
        }
        .score-label {
            font-size: 0.9em;
            color: #666;
        }
        .score-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

    # Card container
    st.markdown('<div class="flashcard-container">', unsafe_allow_html=True)

    # Score tracking display (side by side, visually combined, above flashcard)
    st.markdown('''
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 16px;">
            <div style="background: #f8f9fa; border-radius: 10px 0 0 10px; padding: 12px 24px; text-align: center; min-width: 90px; border: 1px solid #e0e0e0; border-right: none;">
                <div style="font-size: 0.9em; color: #666;">Correct</div>
                <div style="font-size: 1.3em; font-weight: bold; color: #28a745;">{correct}</div>
            </div>
            <div style="width: 2px; height: 40px; background: #e0e0e0;"></div>
            <div style="background: #f8f9fa; border-radius: 0 10px 10px 0; padding: 12px 24px; text-align: center; min-width: 90px; border: 1px solid #e0e0e0; border-left: none;">
                <div style="font-size: 0.9em; color: #666;">Incorrect</div>
                <div style="font-size: 1.3em; font-weight: bold; color: #dc3545;">{incorrect}</div>
            </div>
        </div>
    '''.format(correct=st.session_state.correct_count, incorrect=st.session_state.incorrect_count), unsafe_allow_html=True)

    # Difficulty badge color
    diff = flashcard.get("difficulty", "medium").lower()
    diff_class = {
        "easy": "difficulty-badge difficulty-easy",
        "medium": "difficulty-badge difficulty-medium",
        "hard": "difficulty-badge difficulty-hard"
    }.get(diff, "difficulty-badge difficulty-medium")

    # Flashcard content
    st.markdown(f'''
        <div class="flashcard">
            <div class="card-type">{"Back" if st.session_state.get(f"card_flipped_{index}", False) else "Front"}</div>
            <div class="{diff_class}">{diff.capitalize()}</div>
            <div class="card-content">
                {flashcard["back"] if st.session_state.get(f"card_flipped_{index}", False) else flashcard["front"]}
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Flip button
    if st.button("↻ Flip Card", key=f"flip_{index}"):
        st.session_state[f"card_flipped_{index}"] = not st.session_state.get(f"card_flipped_{index}", False)
        st.rerun()

    # --- RETAIN ONLY THE HTML BUTTON ROW, REMOVE STREAMLIT BUTTON ROW ---
    st.markdown('''
        <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 10px; margin: 18px 0 10px 0;">
            <form style="display:inline; margin:0; padding:0;">
                <button type="submit" name="prev" style="padding:6px 14px; font-size:1em; border-radius:7px; border:1px solid #e0e0e0; background:#f8f9fa; cursor:pointer; min-width:36px;">←</button>
            </form>
            <form style="display:inline; margin:0; padding:0;">
                <button type="submit" name="missed" style="padding:6px 14px; font-size:1em; border-radius:7px; border:1px solid #e0e0e0; background:#fff3f3; color:#dc3545; cursor:pointer; min-width:90px;">❌ Missed It</button>
            </form>
            <form style="display:inline; margin:0; padding:0;">
                <button type="submit" name="knew" style="padding:6px 14px; font-size:1em; border-radius:7px; border:1px solid #e0e0e0; background:#f3fff3; color:#28a745; cursor:pointer; min-width:90px;">✓ Knew It!</button>
            </form>
            <form style="display:inline; margin:0; padding:0;">
                <button type="submit" name="next" style="padding:6px 14px; font-size:1em; border-radius:7px; border:1px solid #e0e0e0; background:#f8f9fa; cursor:pointer; min-width:36px;">→</button>
            </form>
        </div>
    ''', unsafe_allow_html=True)

    # Show final score if all cards are answered
    total_answered = len(st.session_state.cards_answered)
    if total_answered == total:
        score_percentage = (st.session_state.correct_count / total) * 100
        st.markdown(f'''
            <div class="final-score">
                <h2>🎉 Congratulations! You've completed all flashcards!</h2>
                <p class="score-percent">{score_percentage:.1f}%</p>
                <p>You got {st.session_state.correct_count} out of {total} cards correct!</p>
                <p>• Correct: {st.session_state.correct_count}</p>
                <p>• Incorrect: {st.session_state.incorrect_count}</p>
            </div>
        ''', unsafe_allow_html=True)
        # Reset button
        if st.button("Start Over", use_container_width=True):
            st.session_state.correct_count = 0
            st.session_state.incorrect_count = 0
            st.session_state.cards_answered = set()
            st.session_state.current_card_idx = 0
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def show_play_zone():
    st.title("🎮 PlayZone - Interactive Learning")
    
    # Tabs for different features
    tab1, tab2, tab3 = st.tabs(["📝 Flashcards", "🗺️ Mind Maps", "🎯 Learning Games"])
    
    with tab1:
        st.header("Flashcards Generator")
        
        # User input for topic and number of cards
        col1, col2 = st.columns([2, 1])
        with col1:
            topic = st.text_input("Enter a topic for flashcards", 
                                placeholder="e.g., Python Programming, Machine Learning, etc.")
        with col2:
            num_cards = st.number_input("Number of flashcards", 
                                      min_value=1, max_value=64, value=5)
        
        if st.button("Generate Flashcards", type="primary", use_container_width=True) and topic:
            with st.spinner(f"Generating flashcards for {topic}..."):
                flashcards = generate_flashcards(topic, num_cards)
                if flashcards:
                    st.session_state.flashcards = flashcards
                    st.session_state.current_card_idx = 0
                    # Reset scores when generating new flashcards
                    st.session_state.correct_count = 0
                    st.session_state.incorrect_count = 0
                    st.session_state.cards_answered = set()
                    st.success("Flashcards generated successfully!")
                    st.rerun()

        # Show intro message if no flashcards
        if not hasattr(st.session_state, 'flashcards'):
            st.info("""
            ### Flash Cards
            These are auto-generated flash cards! You can...
            - Swipe between cards
            - Flip and track which questions you got right or wrong
            - Quickly refresh your memory on the content!
            """)
        
        # Display flashcards if they exist
        elif st.session_state.flashcards:
            display_flashcard(
                st.session_state.flashcards[st.session_state.current_card_idx],
                st.session_state.current_card_idx,
                len(st.session_state.flashcards)
            )

    with tab2:
        st.info("🚧 Mind Maps feature coming soon!")
    
    with tab3:
        st.info("🚧 Learning Games feature coming soon!")

def run_app():
    """Main entry point for the PlayZone application"""
    show_play_zone()
