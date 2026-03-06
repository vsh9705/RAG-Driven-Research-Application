from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import shutil
import traceback
import logging

from .models import Query, QueryResponse, IngestionStatus
from .rag_engine import rag_engine
from .config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Research Assistant - Production Grade",
    description="AI-powered literature review with parallel processing and caching",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = settings.BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = frontend_path / "templates" / "index.html"
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/ingest", response_model=IngestionStatus)
async def ingest_papers():
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
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__})

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
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
    try:
        response = rag_engine.query(query.question, top_k=query.top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers")
async def list_papers():
    papers = rag_engine.list_documents()
    return {"count": len(papers), "papers": papers}

@app.post("/api/clear-all")
async def clear_all_data():
    try:
        vector_store_path = Path(settings.VECTOR_STORE_PATH)
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
            logger.info("🗑️  Removed vector store")

        pdf_files = list(settings.PAPERS_DIR.glob("*.pdf"))
        for pdf in pdf_files:
            pdf.unlink()
            logger.info(f"🗑️  Removed {pdf.name}")

        rag_engine.vectorstore = None
        rag_engine.qa_chain = None
        rag_engine.clear_cache()

        return {
            "status": "success",
            "message": f"Cleared {len(pdf_files)} documents and vector store",
            "documents_removed": len(pdf_files)
        }
    except Exception as e:
        logger.error(f"❌ Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-cache")
async def clear_query_cache():
    try:
        rag_engine.clear_cache()
        return {"status": "success", "message": "Query cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    vectorstore_exists = Path(settings.VECTOR_STORE_PATH).exists()
    return {
        "status": "healthy",
        "vectorstore_loaded": vectorstore_exists,
        "llm_provider": settings.LLM_PROVIDER,
        "papers_count": len(rag_engine.list_documents()),
        "version": "2.0.0-production"
    }

@app.get("/api/vectorstore-stats")
async def get_vectorstore_stats():
    try:
        stats = rag_engine.get_vectorstore_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)