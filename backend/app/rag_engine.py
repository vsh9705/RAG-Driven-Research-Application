import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm import tqdm

from .config import settings
from .models import CitedSource, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


# ── Prompt template defined at module level so query() can access it ──────────
RAG_PROMPT_TEMPLATE = """You are a research assistant specializing in secure file-sharing systems, with emphasis on client-side cryptography, minimal server trust, and ephemeral data protection.

For every question, strictly follow this structure:

**Major Finding:**
[Give ONE clear sentence summarizing the main insight from the provided context.]

**Technical Analysis:**
- [Bullet 1: key technical detail or mechanism]
- [Bullet 2: supporting evidence or comparison from the papers]
- [Bullet 3: practical implication or limitation]
- [Bullet 4 (optional): additional nuance if relevant]

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


class RAGEngine:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.query_cache = LRUCache(max_size=settings.MAX_CACHE_SIZE)
        self._initialize_embeddings()
        self._initialize_llm()

    def _initialize_embeddings(self):
        logger.info("🔄 Loading embedding model...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            raise

    def _initialize_llm(self):
        logger.info("🔄 Initializing Groq LLM...")
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found")
        try:
            self.llm = ChatGroq(
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            logger.info("✅ Groq LLM initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise

    def _load_single_pdf(self, pdf_path: Path) -> List:
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            logger.info(f"✅ Loaded {pdf_path.name}: {len(docs)} pages")
            return docs
        except Exception as e:
            logger.error(f"❌ Failed to load {pdf_path.name}: {e}")
            return []

    def _process_pdf_batch_parallel(self, pdf_paths: List[Path]) -> List:
        all_docs = []
        logger.info(f"🚀 Processing {len(pdf_paths)} PDFs with {settings.MAX_WORKERS} workers")
        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            future_to_pdf = {
                executor.submit(self._load_single_pdf, path): path
                for path in pdf_paths
            }
            for future in tqdm(as_completed(future_to_pdf),
                               total=len(future_to_pdf),
                               desc="Loading PDFs",
                               unit="pdf"):
                docs = future.result()
                all_docs.extend(docs)
        return all_docs

    def ingest_documents(self, paper_dir: Optional[Path] = None) -> Dict[str, Any]:
        paper_dir = paper_dir or settings.PAPERS_DIR
        try:
            pdf_files = list(paper_dir.glob("*.pdf"))
            if not pdf_files:
                return {
                    "status": "error",
                    "message": "No PDF documents found",
                    "total_documents": 0,
                    "processed_documents": 0,
                    "total_chunks": 0
                }

            logger.info(f"📚 Found {len(pdf_files)} PDFs")
            all_documents = self._process_pdf_batch_parallel(pdf_files)

            if not all_documents:
                return {
                    "status": "error",
                    "message": "Failed to load documents",
                    "total_documents": 0,
                    "processed_documents": 0,
                    "total_chunks": 0
                }

            logger.info(f"📄 Loaded {len(all_documents)} pages total")

            logger.info("✂️  Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            chunks = []
            for doc in tqdm(all_documents, desc="Splitting", unit="page"):
                chunks.extend(text_splitter.split_documents([doc]))

            logger.info(f"✅ Created {len(chunks)} chunks")

            logger.info("🔄 Creating FAISS vector store...")
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )

            vector_path = Path(settings.VECTOR_STORE_PATH)
            vector_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"💾 Saving FAISS index to {vector_path}...")
            self.vectorstore.save_local(str(vector_path))
            logger.info("✅ Vector store created successfully")

            self._initialize_qa_chain()
            self.query_cache.clear()

            unique_docs = len(set([doc.metadata.get("source", "") for doc in all_documents]))
            return {
                "status": "success",
                "message": "Documents ingested with FAISS (parallel processing)",
                "total_documents": unique_docs,
                "processed_documents": unique_docs,
                "total_chunks": len(chunks)
            }

        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Ingestion failed: {str(e)}",
                "total_documents": 0,
                "processed_documents": 0,
                "total_chunks": 0
            }

    def load_vectorstore(self) -> bool:
        try:
            logger.info("🔄 Loading FAISS vector store...")
            vector_path = Path(settings.VECTOR_STORE_PATH)
            if not vector_path.exists():
                logger.error("Vector store not found")
                return False
            self.vectorstore = FAISS.load_local(
                str(vector_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self._initialize_qa_chain()
            logger.info("✅ Vector store loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading: {e}")
            return False

    def _initialize_qa_chain(self):
        """
        QA chain is kept for potential direct use, but query() now handles
        retrieval + prompt formatting explicitly so the template is always applied.
        """
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",                          # MMR for result diversity
            search_kwargs={
                "k": settings.TOP_K_RETRIEVAL,
                "fetch_k": settings.TOP_K_RETRIEVAL * 3,  # fetch more, then diversify
                "lambda_mult": 0.6                       # 0=max diversity, 1=max relevance
            }
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def _get_query_hash(self, question: str, top_k: int) -> str:
        return hashlib.md5(f"{question}_{top_k}".encode()).hexdigest()

    def query(self, question: str, top_k: int = None) -> QueryResponse:
        """
        Retrieves relevant chunks, filters by cosine similarity, then calls the
        LLM with the structured RAG prompt so the response always follows the
        Major Finding / Technical Analysis template.
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL

        if not self.vectorstore:
            if not self.load_vectorstore():
                raise ValueError("No vector store found. Please ingest documents first.")

        # ── Cache check ───────────────────────────────────────────────────────
        if settings.ENABLE_QUERY_CACHE:
            cache_key = self._get_query_hash(question, top_k)
            cached = self.query_cache.get(cache_key)
            if cached:
                logger.info("✅ Cache HIT")
                return cached
            logger.info("❌ Cache MISS")

        # ── Retrieval with similarity filtering ───────────────────────────────
        # Fetch more candidates than needed so filtering doesn't leave us empty
        fetch_k = max(top_k * 3, 10)
        docs_scores = self.vectorstore.similarity_search_with_score(question, k=fetch_k)

        # FAISS returns L2 distances; for unit-normalised embeddings:
        #   cosine_similarity = 1 - (L2_distance² / 2)
        SIMILARITY_THRESHOLD = 0.30   # cosine similarity floor (tune between 0.25–0.45)

        filtered_docs = []
        for doc, l2_dist in docs_scores:
            cos_sim = 1.0 - (l2_dist ** 2) / 2.0
            logger.debug(f"  doc={Path(doc.metadata.get('source','')).name} "
                         f"l2={l2_dist:.4f} cos_sim={cos_sim:.4f}")
            if cos_sim >= SIMILARITY_THRESHOLD:
                filtered_docs.append((doc, cos_sim))

        # Fallback: if threshold removes everything, use the top-k raw results
        if not filtered_docs:
            logger.warning("⚠️  No docs passed threshold — using raw top-k fallback")
            filtered_docs = [
                (doc, round(1.0 - (score ** 2) / 2.0, 4))
                for doc, score in docs_scores[:top_k]
            ]

        # Keep only the top `top_k` after filtering
        filtered_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)[:top_k]
        source_documents = [doc for doc, _ in filtered_docs]
        scores_map = {id(doc): sim for doc, sim in filtered_docs}

        logger.info(f"🔍 Using {len(source_documents)} chunks after filtering (fetch_k={fetch_k})")

        # ── Build context ─────────────────────────────────────────────────────
        context_parts = []
        for i, doc in enumerate(source_documents, 1):
            src = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[Source {i} — {src}, p.{page}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        # ── Invoke LLM using the structured prompt template ───────────────────
        formatted_prompt = PROMPT.format(context=context, question=question)
        result = self.llm.invoke(formatted_prompt)
        answer_text = result.content

        # ── Build cited sources ───────────────────────────────────────────────
        cited_sources = []
        for doc in source_documents:
            source_path = doc.metadata.get("source", "Unknown")
            source_name = Path(source_path).name if source_path != "Unknown" else "Unknown"
            cited_sources.append(CitedSource(
                document=source_name,
                page=doc.metadata.get("page", None),
                chunk_content=doc.page_content[:500],
                relevance_score=round(scores_map.get(id(doc), 0.0), 4)
            ))

        response = QueryResponse(
            question=question,
            answer=answer_text,
            cited_sources=cited_sources,
            timestamp=datetime.now(),
            response_id=str(uuid.uuid4())[:8]
        )

        if settings.ENABLE_QUERY_CACHE:
            self.query_cache.set(cache_key, response)
            logger.info("💾 Cached response")

        return response
    
    def list_documents(self) -> List[str]:
        """Return a list of unique source document names in the vector store."""
        if not self.vectorstore:
            return []
        try:
            docstore = self.vectorstore.docstore._dict
            sources = set()
            for doc in docstore.values():
                source = doc.metadata.get("source", "")
                if source:
                    sources.add(Path(source).name)
            return sorted(sources)
        except Exception as e:
            logger.error(f"❌ Failed to list documents: {e}")
            return []

    def clear_cache(self):
        self.query_cache.clear()
        logger.info("🗑️  Cache cleared")


rag_engine = RAGEngine()