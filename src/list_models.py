import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")


genai.configure(api_key=GOOGLE_API_KEY)

print("Models supporting embeddings:\n")

for model in genai.list_models():
    if "embedContent" in model.supported_generation_methods:
        print(model.name)