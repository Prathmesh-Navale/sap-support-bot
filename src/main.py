from dotenv import load_dotenv
load_dotenv() # Load this FIRST

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.engine import SAPChatEngine

app = FastAPI()
engine = SAPChatEngine()

class ChatRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: ChatRequest):
    try:
        return engine.run_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)