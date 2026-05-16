from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.engine import SAPChatEngine


app = FastAPI(
    title="SAP Support Bot",
    version="1.0.0"
)


engine = SAPChatEngine()


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "SAP Support Bot API Running"
    }


@app.post("/ask")
async def ask_question(request: ChatRequest):

    try:
        response = engine.run_query(request.query)
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )