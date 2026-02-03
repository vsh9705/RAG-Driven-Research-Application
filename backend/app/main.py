from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import json
from typing import List
import traceback
import logging

from .models import Query, QueryResponse, IngestionStatus, ManualEvaluation, EvaluationStats
from .rag_engine import rag_engine
from .config import settings

app = FastAPI(
    title="RAG Research Assistant",
    description="AI-powered literature review with manual evaluation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_path = settings.BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

# In-memory storage for evaluations
evaluations_store: List[ManualEvaluation] = []

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve frontend"""
    index_path = frontend_path / "templates" / "index.html"
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/ingest", response_model=IngestionStatus)
async def ingest_papers():
    """Ingest all PDF papers with detailed error reporting"""
    try:
        result = rag_engine.ingest_documents()
        return IngestionStatus(**result)
    except Exception as e:
        print("=" * 80)
        print("ERROR DURING INGESTION:")
        print(str(e))
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 80)
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "type": type(e).__name__
            }
        )

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Upload single PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    try:
        file_path = settings.PAPERS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": f"Uploaded {file.filename}", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_rag(query: Query):
    """Query RAG - searches across all papers"""
    try:
        response = rag_engine.query(query.question, top_k=query.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate")
async def submit_evaluation(evaluation: ManualEvaluation):
    """Submit manual evaluation of a response"""
    try:
        evaluations_store.append(evaluation)
        
        # Save to file
        eval_file = settings.OUTPUTS_DIR / "evaluations" / f"eval_{evaluation.response_id}.json"
        eval_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(eval_file, "w") as f:
            json.dump(evaluation.dict(), f, indent=2, default=str)
        
        return {
            "message": "Evaluation saved",
            "response_id": evaluation.response_id,
            "total_evaluations": len(evaluations_store)
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluations", response_model=EvaluationStats)
async def get_evaluations():
    """Get all evaluations and statistics"""
    if not evaluations_store:
        return EvaluationStats(
            total_evaluations=0,
            avg_relevance=0.0,
            avg_hallucination=0.0,
            avg_completeness=None,
            avg_faithfulness=None,
            evaluations=[]
        )
    
    avg_rel = sum(e.relevance_score for e in evaluations_store) / len(evaluations_store)
    avg_hal = sum(e.hallucination_score for e in evaluations_store) / len(evaluations_store)
    
    # Calculate optional metrics
    completeness_scores = [e.completeness_score for e in evaluations_store if e.completeness_score is not None]
    faithfulness_scores = [e.faithfulness_score for e in evaluations_store if e.faithfulness_score is not None]
    
    avg_comp = sum(completeness_scores) / len(completeness_scores) if completeness_scores else None
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    
    return EvaluationStats(
        total_evaluations=len(evaluations_store),
        avg_relevance=avg_rel,
        avg_hallucination=avg_hal,
        avg_completeness=avg_comp,
        avg_faithfulness=avg_faith,
        evaluations=evaluations_store
    )

@app.get("/api/papers")
async def list_papers():
    """List all papers"""
    papers = rag_engine.list_documents()
    return {
        "count": len(papers),
        "papers": papers
    }

@app.get("/api/health")
async def health_check():
    """Health check"""
    vectorstore_exists = Path(settings.VECTOR_STORE_PATH).exists()
    return {
        "status": "healthy",
        "vectorstore_loaded": vectorstore_exists,
        "llm_provider": settings.LLM_PROVIDER,
        "papers_count": len(rag_engine.list_documents()),
        "evaluations_count": len(evaluations_store)
    }

@app.get("/api/vectorstore-stats")
async def get_vectorstore_stats():
    """Get vector store statistics"""
    try:
        stats = rag_engine.get_vectorstore_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/api/clear-all")
async def clear_all_data():
    """Clear all documents and vector store"""
    try:
        import shutil
        
        # Remove vector store
        vector_store_path = Path(settings.VECTOR_STORE_PATH)
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
        
        # Remove all PDFs
        pdf_files = list(settings.PAPERS_DIR.glob("*.pdf"))
        for pdf in pdf_files:
            pdf.unlink()

        # Reset RAG engine
        rag_engine.vectorstore = None
        rag_engine.qa_chain = None
        
        return {
            "status": "success",
            "message": f"Cleared {len(pdf_files)} documents and vector store",
            "documents_removed": len(pdf_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)