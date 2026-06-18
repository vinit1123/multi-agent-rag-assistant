from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2"
)
app = FastAPI()


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest):

    response = llm.invoke(
        req.question
    )

    return {
        "answer": response.content
    }