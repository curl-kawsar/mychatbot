import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from typing import List, Dict
import uuid
from datetime import datetime, timedelta
import base64
from io import BytesIO
import requests

# Load environment variables - only in development
if os.path.exists(".env"):
    load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv("API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("No API_KEY found in environment variables")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Store conversations in memory
conversations: Dict[str, List[Dict]] = {}

class Question(BaseModel):
    text: str
    session_id: str = None

class ConversationHistory:
    def __init__(self):
        self.messages = []
        self.last_updated = datetime.now()

def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = datetime.now()
    expired_sessions = [
        session_id for session_id, conv in conversations.items()
        if current_time - conv.last_updated > timedelta(hours=1)
    ]
    for session_id in expired_sessions:
        del conversations[session_id]

def extract_text_from_pdf():
    """Extract text content from PDF file"""
    try:
        text = ""
        with open("Kawsar-Resume.pdf", 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {str(e)}")

def create_context(resume_text, conversation_history=None):
    """Create a context prompt for the model"""
    context = f"""You are Kawsar's AI assistant that helps answer questions about him. Use this information about Kawsar:
    {resume_text}
    
    Only answer questions based on the information provided above.
    If you don't have enough information to answer accurately, politely say so.
    Keep answers concise and professional.
    Do not mention that this information comes from a resume."""

    if conversation_history:
        context += "\n\nPrevious conversation:\n"
        for msg in conversation_history:
            context += f"User: {msg['question']}\nAssistant: {msg['answer']}\n"

    return context

@app.post("/ask")
async def ask_question(question: Question):
    try:
        # Create or get session ID
        if not question.session_id:
            question.session_id = str(uuid.uuid4())
        
        # Cleanup old sessions
        cleanup_old_sessions()
        
        # Get or create conversation history
        if question.session_id not in conversations:
            conversations[question.session_id] = ConversationHistory()
        
        conversation = conversations[question.session_id]
        conversation.last_updated = datetime.now()

        # Extract resume text
        resume_text = extract_text_from_pdf()
        
        # Create context with conversation history
        context = create_context(resume_text, conversation.messages)
        
        # Get response from Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Context: {context}
        
        Question: {question.text}
        
        Answer as if you are Kawsar's personal AI assistant, without mentioning any source documents: """
        
        response = model.generate_content(prompt)
        
        # Store conversation
        conversation.messages.append({
            "question": question.text,
            "answer": response.text
        })
        
        return {
            "response": response.text,
            "session_id": question.session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome! I'm Kawsar's AI Assistant. How can I help you?"}

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    if session_id in conversations:
        del conversations[session_id]
        return {"message": "Session ended successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

# Before deploying, add PDF content to Heroku config:
# First convert your PDF to base64
with open("Kawsar-Resume.pdf", "rb") as f:
    pdf_base64 = base64.b64encode(f.read()).decode()

# Then set it in Heroku
# heroku config:set PDF_CONTENT="<your-base64-pdf-content>"

# Then set PDF_URL in Heroku:
# heroku config:set PDF_URL="https://your-cloud-storage-url/Kawsar-Resume.pdf"
