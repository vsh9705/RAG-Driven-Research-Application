import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Project paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    BACKEND_DIR = BASE_DIR / "backend"
    DATA_DIR = BACKEND_DIR / "data"
    PAPERS_DIR = DATA_DIR / "papers"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = BACKEND_DIR / "models"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # RAG Settings - OPTIMIZED FOR LARGE DATASETS
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
    EMBEDDING_DEVICE = "cpu"  # Change to "cuda" if you have GPU
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    
    # Batch Processing Settings
    INGESTION_BATCH_SIZE = 10  # Process 10 PDFs at a time
    EMBEDDING_BATCH_SIZE = 32  # Embed 32 chunks at a time
    MAX_WORKERS = 4  # Parallel workers for PDF processing
    
    # Vector Store - OPTIMIZED
    VECTOR_STORE_PATH = str(MODELS_DIR / "chroma_db")
    COLLECTION_NAME = "research_papers"
    CHROMA_PERSIST_INTERVAL = 100  # Persist every 100 chunks
    
    # Memory Management
    MAX_MEMORY_MB = 4096  # Max 4GB for vector store operations
    ENABLE_COMPRESSION = True
    
    # LLM Settings
    LLM_PROVIDER = "groq"
    LLM_MODEL = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE = 0.3
    MAX_TOKENS = 2000
    
    # Query Optimization
    SIMILARITY_THRESHOLD = 0.7  # Filter low-relevance results
    ENABLE_RERANKING = False  # Set True for better accuracy (slower)
    
    # Cache Settings
    ENABLE_QUERY_CACHE = True
    CACHE_TTL_SECONDS = 3600

settings = Settings()

# Create directories
for dir_path in [settings.PAPERS_DIR, settings.PROCESSED_DIR, 
                 settings.MODELS_DIR, settings.OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)