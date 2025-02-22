import os
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2
import textwrap
from typing import List, Dict

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

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

def get_gemini_response(user_input, context):
    """Get response from Gemini model"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""Context: {context}
    
    Question: {user_input}
    
    Answer as if you are Kawsar's personal AI assistant, without mentioning any source documents: """
    
    response = model.generate_content(prompt)
    return response.text

def main():
    # Extract resume text
    resume_text = extract_text_from_pdf('Kawsar-Resume.pdf')
    conversation_history = []
    
    print("👋 Hi! I'm Kawsar's AI assistant. How can I help you learn more about him?")
    print("Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye! 👋")
            break
            
        if user_input:
            try:
                context = create_context(resume_text, conversation_history)
                response = get_gemini_response(user_input, context)
                
                # Store conversation
                conversation_history.append({
                    "question": user_input,
                    "answer": response
                })
                
                print("\nAssistant:", textwrap.fill(response, width=80))
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try asking your question again.")

if __name__ == "__main__":
    main()
