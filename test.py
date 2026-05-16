from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load .env
load_dotenv()

# Configure API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load model
model = genai.GenerativeModel("models/gemini-2.5-flash")

# Generate response
response = model.generate_content("Hello")

print(response.text)