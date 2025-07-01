import os
import re
import uuid
import random
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()
# Constants
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Input schema
class QuizInput(BaseModel):
    subject: str
    topic: str
    program: str
    difficulty: str  # easy, intermediate, difficult
    number_of_questions: int


# Step 1: Create dynamic prompt with UUID and high randomness
def generate_prompt(data: QuizInput) -> str:
    session_id = uuid.uuid4()
    return f"""
You are a quiz generator AI.

✅ Input:
Subject: {data.subject}
Topic: {data.topic}
Program: {data.program}
Difficulty: {data.difficulty.capitalize()}
Number of Questions: {data.number_of_questions}
Session ID: {session_id}  # Force variation on each run

🎯 Task:
Generate exactly {data.number_of_questions} unique multiple choice questions.
Each question must have:
- A clear and concise question statement
- 4 options labeled option-1 to option-4
- A correct answer line in the format: ✅ Correct Answer: option-X

🧠 Format:
Question 1
 What is...?
option-1: ...
option-2: ...
option-3: ...
option-4: ...
 ✅ Correct Answer: option-2

Only return the questions in this format, nothing else.
    """

# Step 2: Call Groq API
def call_groq_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a quiz generator for students."},
            {"role": "user", "content": prompt}
        ],
        "temperature": round(random.uniform(0.85, 1.0), 2)
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"GROQ API Error: {response.status_code} - {response.text}")


# Step 3: Parse LLM output to JSON
def parse_quiz(raw_text: str):
    questions = {}
    parts = re.split(r"Question \d+\n", raw_text)
    parts = [p.strip() for p in parts if p.strip()]

    for idx, part in enumerate(parts):
        lines = part.splitlines()
        if not lines:
            continue

        title = lines[0].strip()
        options = []
        correct_index = None

        for line in lines[1:]:
            if line.lower().startswith("option-"):
                match = re.match(r"option-(\d+): (.+)", line)
                if match:
                    options.append(match.group(2).strip())
            elif "✅ Correct Answer" in line:
                match = re.search(r"option-(\d+)", line)
                if match:
                    correct_index = int(match.group(1)) - 1  # Convert to 0-based index

        questions[str(idx)] = {
            "title": title,
            "options": options,
            "correct": correct_index
        }

    return {"questions": questions}

# Step 4: API Endpoint
@app.post("/generate-quiz")
def generate_quiz(data: QuizInput):
    prompt = generate_prompt(data)
    raw_output = call_groq_api(prompt)
    parsed = parse_quiz(raw_output)
    return parsed
