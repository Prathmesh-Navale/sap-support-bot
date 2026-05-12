import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyC8vL5VAe_QAL66AOcdZhUhMK5lPEjkYmQ"))

print("Models supporting embeddings:")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"-> {m.name}")