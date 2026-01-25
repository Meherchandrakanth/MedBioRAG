import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pipeline import MedBioRAGPipeline

from fastapi.responses import RedirectResponse

load_dotenv()

app = FastAPI(title="MediBioRAG Production API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/health")

pipeline = None

class QueryRequest(BaseModel):
    query: str
    task_type: Optional[str] = "long-form"
    github_token: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    status: str

class IndexRequest(BaseModel):
    documents: List[str]

@app.on_event("startup")
async def startup_event():
    global pipeline
    token = os.environ.get("GITHUB_TOKEN") or "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"
    
    # Try to load existing index first
    if os.path.exists("storage/docs.pkl"):
        print("Found existing index, loading...")
        pipeline = MedBioRAGPipeline(github_token=token)
    else:
        print("No index found, initializing with seed data...")
        seed_docs = [
            "Insulin is a hormone produced by the pancreas that regulates glucose levels in the blood.",
            "Type 1 diabetes occurs when the body destroys insulin-producing beta cells in the pancreas.",
            "Metformin is a common medication used to treat Type 2 diabetes by improving insulin sensitivity."
        ]
        pipeline = MedBioRAGPipeline(documents=seed_docs, github_token=token)

@app.post("/index")
async def add_documents(request: IndexRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        pipeline.add_new_documents(request.documents)
        return {"status": "success", "message": f"Added {len(request.documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        answer = pipeline.run(request.query, task_type=request.task_type)
        return QueryResponse(answer=answer, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "MediBioRAG"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
