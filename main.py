from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
import PyPDF2
import os
import json
from dotenv import load_dotenv
import spacy

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://lakshya-ai.vercel.app/"  # Next.js Frontend
    "https://lakshya-learn-ai.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_FILE = 'database.json'

def read_data():
    try:
        with open(DATABASE_FILE, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

def write_data(data):
    with open(DATABASE_FILE, 'w') as file:
        json.dump(data, file, indent=4)

# Function to extract details from the resume using spaCy
def extract_resume_details(resume_text: str):
    doc = nlp(resume_text)
    experience, languages, projects, achievements = [], [], [], []

    for sentence in doc.sents:
        text = sentence.text.lower()
        if 'experience' in text:
            experience.append(sentence.text)
        if any(lang in text for lang in ['python', 'javascript', 'java', 'c++', 'typescript']):
            languages.append(sentence.text)
        if 'project' in text:
            projects.append(sentence.text)
        if 'achievement' in text or 'certification' in text:
            achievements.append(sentence.text)

    return experience, languages, projects, achievements

# Function to generate a single interview question based on resume details
def generate_single_question(experience, languages, projects, role):
    messages = [
        {"role": "system", "content": "You are an expert interviewer."},
        {"role": "user", "content": (
            f"Create a single interview question for a {role} based on the following details:\n"
            f"Experience: {', '.join(experience)}\n"
            f"Languages: {', '.join(languages)}\n"
            f"Projects: {', '.join(projects)}\n"
            "Ask one short, relevant question based on the details."
        )}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )
        question = response['choices'][0]['message']['content'].strip()
        return question
    except Exception as e:
        return f"Error generating question: {e}"

# Endpoint to upload resume and generate an interview question
@app.post("/upload")
async def upload_resume(resume: UploadFile = File(...), role: str = Form(...)):
    try:
        # Extract text from the uploaded PDF resume
        reader = PyPDF2.PdfReader(resume.file)
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + "\n"

        # Extract relevant details from the resume text
        experience, languages, projects, achievements = extract_resume_details(resume_text)
        
        # Generate a single interview question based on the extracted details
        question = generate_single_question(experience, languages, projects, role)

        # Store the resume, role, and question in the database
        interview_data = {
            "resume": resume_text,
            "role": role,
            "questions": [question],
            "answers": []
        }
        
        data = read_data()
        data.append(interview_data)
        write_data(data)

        return {"question": question}
    except Exception as e:
        return {"error": f"Failed to process upload: {e}"}

# Endpoint to submit an answer to a question
@app.post("/submit-answer")
async def submit_answer(
    question: str = Form(...),
    answer: str = Form(...)
):
    try:
        data = read_data()
        
        if not data:
            return {"error": "No data found in the database."}

        # Retrieve the most recent interview record
        if len(data) > 0:
            interview_data = data[-1]  # Use the most recent record
            interview_data["questions"].append(question)
            interview_data["answers"].append(answer)
            
            write_data(data)
            return {"message": "Answer saved successfully."}
        else:
            return {"error": "No interview data found."}
    except Exception as e:
        return {"error": f"Failed to save answer: {e}"}

# Endpoint to get the next question based on the resume details
@app.get("/get-next-question")
async def get_next_question():
    try:
        data = read_data()
        
        if not data:
            return {"error": "No interview data found in the database."}

        # Retrieve the most recent interview record
        if len(data) > 0:
            interview_data = data[-1]
            experience, languages, projects, _ = extract_resume_details(interview_data["resume"])
            role = interview_data["role"]
            
            # Generate a new question based on the existing context
            new_question = generate_single_question(
                experience,
                languages,
                projects,
                role
            )
            
            interview_data["questions"].append(new_question)
            write_data(data)
            
            return {"question": new_question}
        else:
            return {"error": "No interview data found."}
    except Exception as e:
        return {"error": f"Failed to retrieve next question: {e}"}

# Endpoint to clear the database
@app.post("/clear-database")
async def clear_database():
    try:
        open(DATABASE_FILE, 'w').close()  # Clear the file
        return {"message": "Database cleared successfully"}
    except Exception as e:
        return {"error": f"Failed to clear database: {e}"}

@app.get("/")
async def root():
    return {"message": "AI Mock Interview Platform is live!"}