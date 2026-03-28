from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import agent

app = FastAPI(
    title="Research Assistant Agent",
    description="An agentic AI that searches the web and reasons over results",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    iterations: int

@app.get("/")
def root():
    return {"message": "Research Assistant Agent is running!"}

@app.post("/research", response_model=AnswerResponse)
def research(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Initial state
    initial_state = {
        "question": request.question,
        "search_results": "",
        "answer": "",
        "is_sufficient": False,
        "iterations": 0
    }
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    return AnswerResponse(
        question=request.question,
        answer=final_state["answer"],
        iterations=final_state["iterations"]
    )