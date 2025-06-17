import json
import asyncio
import httpx
from datetime import datetime
import random
import os
import re
from quiz_utils import get_or_create_eventloop

API_KEY = "AIzaSyDv0Cc2_I6k4V-m1xAim_HG4jyu63b_qCc"

TOPICS = ["Python for Data Science",
        "Web Development Fundamentals",
        "Mobile App Development",
        "Cloud Computing Basics",
        "Cybersecurity Essentials",
        "Machine Learning Fundamentals",
        "Blockchain Technology",
        "DevOps Practices", 
        "UI/UX Design",
        "Database Management", "Thermodynamics", "Fluid Mechanics", "Strength of Materials", "Engineering Mechanics", "Theory of Machines",
    "Machine Design", "Heat Transfer", "IC Engines", "Refrigeration and Air Conditioning", "Workshop Technology",
    "Engineering Drawing", "Mechanical Measurements", "Production Engineering", "CNC Machines", "Automobile Engineering",

    # ⚡ Electrical & Electronics
    "Basic Electrical Engineering", "Network Analysis", "Electrical Machines", "Power Systems", 
    "Control Systems", "Electrical Measurements", "Power Electronics", "Microprocessors", "Digital Electronics",
    "Analog Electronics", "Electric Circuits", "Electromagnetic Fields", "Switchgear and Protection", 
    "Transformers and Induction Motors", "Utilization of Electrical Energy",

    # 📡 Electronics & Communication
    "Signals and Systems", "Communication Systems", "Microwave Engineering", "Radar and Satellite Systems",
    "Wireless Communication", "Analog Communication", "Digital Signal Processing", "Fiber Optics", 
    "VLSI Design", "Embedded Systems", "Antenna and Wave Propagation", "Image Processing", "Telecommunication Switching",

    # 🏗️ Civil Engineering
    "Surveying and Levelling", "Building Materials", "Structural Analysis", "Concrete Technology",
    "Transportation Engineering", "Soil Mechanics", "Water Resources Engineering", "Environmental Engineering",
    "Estimation and Costing", "Construction Management", "Hydraulics", "Design of Steel Structures", 
    "Urban Planning", "AutoCAD for Civil","Digital Photography",
        "Graphic Design", 
        "Video Editing",
        "Creative Writing",
        "Music Production",
        "Interior Design",
        "Animation Basics",
        "Fashion Design",
        "Content Creation",
        "Drawing Fundamentals",
        "3D Modeling",
    # 💉 Medical & Health Sciences
    "Human Anatomy", "Human Physiology", "Pharmacology", "Pathology", "Microbiology", 
    "Biochemistry", "First Aid & Emergency Care", "Community Medicine", "General Surgery Basics",
    "Nutrition and Dietetics", "Medical Terminology", "Patient Care Techniques", "Basic Nursing Procedures",
    "Physiotherapy Fundamentals", "Public Health and Sanitation", "OBG (Obstetrics & Gynecology)",
    "Mental Health Awareness", "Blood Banking and Transfusion", "Radiography Basics", "Pharmaceutical Chemistry",

    # 🔬 Science (Physics, Chem, Bio)
    "Kinematics", "Laws of Motion", "Work, Energy and Power", "Electricity and Magnetism", "Optics",
    "Atomic Physics", "Organic Chemistry", "Inorganic Chemistry", "Chemical Bonding", "Periodic Table",
    "Acids, Bases, and Salts", "Stoichiometry", "Cell Biology", "Genetics", "Ecology and Environment",
    "Evolution", "Photosynthesis", "Human Digestive System", "Endocrine System", "Microorganisms and Diseases"
    "Financial Planning",
        "Digital Marketing",
        "Project Management",
        "Entrepreneurship Basics",
        "Business Analytics",
        "Investment Strategies",
        "Supply Chain Management",
        "Business Communication",
        "Risk Management",
        "Leadership Skills"
]



DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]
READ_TIMES = ["2 minutes", "5 minutes", "10 minutes"]

COURSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "courseTitle": {"type": "STRING"},
        "introduction": {"type": "STRING"},
        "modules": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "moduleNumber": {"type": "INTEGER"},
                    "moduleTitle": {"type": "STRING"},
                    "chapters": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "chapterTitle": {"type": "STRING"},
                                "description": {"type": "STRING"}
                            },
                            "required": ["chapterTitle", "description"]
                        }
                    }
                },
                "required": ["moduleNumber", "moduleTitle", "chapters"]
            }
        },
        "conclusion": {"type": "STRING"}
    },
    "required": ["courseTitle", "introduction", "modules", "conclusion"]
}

async def generate_content_with_gemini(prompt, temperature, max_tokens, top_k, top_p, response_schema=None):
    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    generation_config = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topK": int(top_k),
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
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and \
           result["candidates"][0]["content"].get("parts") and \
           result["candidates"][0]["content"]["parts"][0].get("text"):
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            text_response = text_response.strip()
            if text_response.startswith("```"):
                text_response = re.sub(r"^```(\w+)?", "", text_response).strip()
            if text_response.endswith("```"):
                text_response = text_response[:-3].strip()

            json_match = re.search(r"\{.*\}", text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        else:
            print(f"LLM response structure unexpected: {result}")
    except Exception as e:
        print(f"Error generating content: {e}")
    return None

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def save_course_to_files(dataset_entry):
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    dataset_file = os.path.join(output_dir, "my_dataset.jsonl")
    # Always print the dataset file path for debugging
    print(f"Appending to dataset file: {dataset_file}")
    try:
        with open(dataset_file, "a", encoding="utf-8") as f:
            json_str = json.dumps(dataset_entry, ensure_ascii=False)
            f.write(json_str + "\n")
        print(f"Successfully wrote to {dataset_file}")
    except Exception as e:
        print(f"Error writing to {dataset_file}: {e}")
    course_title = sanitize_filename(dataset_entry["output"]["courseTitle"].replace(" ", "_").lower())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    course_file = os.path.join(output_dir, f"{course_title}_{timestamp}.json")
    try:
        with open(course_file, "w", encoding="utf-8") as f:
            json.dump(dataset_entry, f, indent=2, ensure_ascii=False)
        print(f"Saved individual course file: {course_file}")
    except Exception as e:
        print(f"Error writing individual course file: {e}")
    course_list_file = os.path.join(output_dir, "course_list.txt")
    try:
        with open(course_list_file, "a", encoding="utf-8") as f:
            f.write(f"{dataset_entry['timestamp']} - {dataset_entry['output']['courseTitle']}\n")
        print(f"Updated course list: {course_list_file}")
    except Exception as e:
        print(f"Error writing course list: {e}")
    return True

async def generate_random_course():
    topic = random.choice(TOPICS)
    difficulty = random.choice(DIFFICULTY_LEVELS)
    num_modules = random.randint(3, 6)
    read_time = random.choice(READ_TIMES)
    temperature = round(random.uniform(0.3, 0.7), 2)
    max_tokens = random.choice([1024, 2048])
    top_k = random.randint(10, 32)
    top_p = round(random.uniform(0.5, 1.0), 2)
    input_params = {
        "course_topic": topic,
        "difficulty_level": difficulty,
        "num_modules": num_modules,
        "read_time_per_module": read_time,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "top_p": top_p
    }
    course_prompt = f"""
    Generate a detailed course outline in JSON format for a '{topic}' course.
    The course should be designed for a '{difficulty}' level audience.
    It must have exactly {num_modules} modules.
    Each module should have chapters, and the content for each module should be designed to take approximately {read_time} to read.
    """
    course_data = await generate_content_with_gemini(
        course_prompt, temperature, max_tokens, top_k, top_p, response_schema=COURSE_SCHEMA)
    if course_data and isinstance(course_data, dict) and "courseTitle" in course_data:
        dataset_entry = {
            "timestamp": str(datetime.now()),
            "input": input_params,
            "output": course_data
        }
        if save_course_to_files(dataset_entry):
            print(f"Generated course: {course_data['courseTitle']}")
            return True
    print("Failed to generate or save course")
    return False

def initialize_output_directory():
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    dataset_file = os.path.join(output_dir, "my_dataset.jsonl")
    if not os.path.exists(dataset_file):
        with open(dataset_file, "w", encoding="utf-8") as f:
            f.write("")
    print(f"Output directory initialized at: {output_dir}")

def count_existing_courses():
    dataset_file = os.path.join(os.path.dirname(__file__), "output", "my_dataset.jsonl")
    if not os.path.exists(dataset_file):
        return 0
    with open(dataset_file, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

async def main():
    num_courses = int(input("How many courses would you like to generate? "))
    initialize_output_directory()
    dataset_file = os.path.join(os.path.dirname(__file__), "output", "my_dataset.jsonl")
    existing_courses = sum(1 for _ in open(dataset_file, 'r', encoding='utf-8') if _.strip()) if os.path.exists(dataset_file) else 0
    print(f"Found {existing_courses} existing courses in the dataset.")
    print(f"Starting generation of {num_courses} new courses...")
    successful = 0
    for i in range(num_courses):
        print(f"Generating course {i+1}/{num_courses}...")
        if await generate_random_course():
            successful += 1
            existing_courses += 1
        await asyncio.sleep(1)
    print(f"\nGeneration complete! Successfully generated {successful}/{num_courses} courses.")
    print(f"Total courses in dataset: {existing_courses}")

if __name__ == "__main__":
    if not API_KEY:
        print("Please set your API key in the script first!")
    else:
        loop = get_or_create_eventloop()
        loop.run_until_complete(main())