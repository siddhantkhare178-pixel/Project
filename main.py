from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field

from agent import ask

app = FastAPI(
    title="Agentic RAG Assistant API",
    description="Ask about the PDF, do math, or convert currency. The agent picks the tool.",
     version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question:str = Field(min_length=3,max_length=1000)
    session_id:str = Field(
        default="default",min_length=1,max_length=50,
        description="same id means same conversation memory"
    )

    model_config = {
        "json_schema_extra":{
            "examples":[
                {"question": "What is the main topic of the PDF?", "session_id": "user1"}
            ]
        }
    }  

class AskResponse(BaseModel):
    answer:str
    session_id:str 

@app.get("/health",tags=['system'])
def health():
    return {"status": "ok"} 


@app.post("/ask",response_model=AskResponse,tags=["Assistant"]) 
def ask_endpoint(request:AskRequest):
    try:
        answer = ask(request.question,session_id=request.session_id)
    except Exception as e:
        print(f"ask() failed: {e}") 
        raise HTTPException(
            status_code=500,
            detail="The assistant failed to answer. Please try again.",
        )
    return AskResponse(answer=answer,session_id=request.session_id) 


