from fastapi import FastAPI
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Read the API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

class QuizInput(BaseModel):
    subject: str
    topic: str
    program: str
    difficulty: str  # easy, intermediate, difficult
    number_of_questions: int

def generate_prompt(data: QuizInput) -> str:
    return f"""
You are a quiz generator AI. Based on the following inputs, generate a multiple choice quiz.

✅ Input:
Topic: {data.topic}
Program: {data.program}
Number of Questions: {data.number_of_questions}
Difficulty: {data.difficulty.capitalize()}

🧠 Output format (exactly like this):

Question 1
 What does LAN stand for?
option-1: Local Access Network
option-2: Local Area Network
option-3: Large Area Network
option-4: Long Access Node
 ✅ Correct Answer: option-2

Repeat the above format for all {data.number_of_questions} questions related to {data.topic} in {data.program} with {data.difficulty} difficulty.
    """

def call_groq_api(prompt: str):
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
        "temperature": 0.7
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.post("/generate-quiz")
def generate_quiz(data: QuizInput):
    prompt = generate_prompt(data)
    quiz_output = call_groq_api(prompt)
    return {"quiz": quiz_output}
