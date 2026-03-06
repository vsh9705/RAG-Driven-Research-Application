import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    BACKEND_DIR = BASE_DIR / "backend"
    DATA_DIR = BACKEND_DIR / "data"
    PAPERS_DIR = DATA_DIR / "papers"
    PROCESSED_DIR = DATA_DIR / "processed"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BACKEND_DIR / "models"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
    EMBEDDING_DEVICE = "cpu"

    # ── Chunking ──────────────────────────────────────────────────────────────
    # 900 chars captures a full paragraph in most academic PDFs.
    # 200 overlap preserves cross-sentence context at boundaries.
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 200

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # 6 gives the LLM enough material without hitting context limits.
    # query() fetches TOP_K_RETRIEVAL * 3 candidates before filtering.
    TOP_K_RETRIEVAL = 6

    # ── Parallelism / batching ────────────────────────────────────────────────
    MAX_WORKERS = 4
    EMBEDDING_BATCH_SIZE = 32
    STORAGE_BATCH_SIZE = 100
    PERSIST_INTERVAL = 500
    MAX_MEMORY_MB = 4096

    # ── Vector store ──────────────────────────────────────────────────────────
    VECTOR_STORE_PATH = str(MODELS_DIR / "faiss_index")
    COLLECTION_NAME = "faiss_index"

    # ── LLM ───────────────────────────────────────────────────────────────────
    LLM_PROVIDER = "groq"
    LLM_MODEL = "llama-3.3-70b-versatile"
    # 0.1 keeps answers factual and consistent; raise to 0.4 for more variety
    LLM_TEMPERATURE = 0.1
    MAX_TOKENS = 2000

    # ── Cache ─────────────────────────────────────────────────────────────────
    ENABLE_QUERY_CACHE = True
    CACHE_TTL_SECONDS = 3600
    MAX_CACHE_SIZE = 100


settings = Settings()

for dir_path in [settings.PAPERS_DIR, settings.PROCESSED_DIR,
                 settings.MODELS_DIR, settings.OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)