import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import uuid
import logging
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .config import settings
from .models import CitedSource, QueryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self._initialize_embeddings()
        self._initialize_llm()
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        logger.info("🔄 Loading embedding model...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            raise
        
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        logger.info(f"🔄 Initializing Groq LLM ({settings.LLM_MODEL})...")
        
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
    
    def ingest_documents(self, paper_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Simple ingestion with proper error handling"""
        paper_dir = paper_dir or settings.PAPERS_DIR
        
        try:
            # Get all PDFs
            pdf_files = list(paper_dir.glob("*.pdf"))
            
            if not pdf_files:
                return {
                    "status": "error",
                    "message": f"No PDF documents found in {paper_dir}",
                    "total_documents": 0,
                    "processed_documents": 0,
                    "total_chunks": 0
                }
            
            logger.info(f"📚 Found {len(pdf_files)} PDFs to process")
            
            # Load all documents
            all_documents = []
            for pdf_file in pdf_files:
                try:
                    logger.info(f"Loading {pdf_file.name}...")
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    all_documents.extend(docs)
                    logger.info(f"✅ Loaded {pdf_file.name}: {len(docs)} pages")
                except Exception as e:
                    logger.error(f"❌ Failed to load {pdf_file.name}: {e}")
                    continue
            
            if not all_documents:
                return {
                    "status": "error",
                    "message": "Failed to load any documents",
                    "total_documents": 0,
                    "processed_documents": 0,
                    "total_chunks": 0
                }
            
            logger.info(f"📄 Total pages loaded: {len(all_documents)}")
            
            # Split documents
            logger.info("✂️  Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len
            )
            chunks = text_splitter.split_documents(all_documents)
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Create vector store - with permission fix
            logger.info("🔄 Creating vector store (this may take a few minutes)...")
            
            # Delete old store if exists to avoid readonly issues
            vector_store_path = Path(settings.VECTOR_STORE_PATH)
            if vector_store_path.exists():
                logger.info("🗑️  Removing old vector store...")
                try:
                    shutil.rmtree(vector_store_path)
                except Exception as e:
                    logger.warning(f"⚠️  Could not remove old store: {e}")
                    # Try to make it writable first
                    os.chmod(vector_store_path, 0o755)
                    shutil.rmtree(vector_store_path)
            
            # Ensure parent directory exists and is writable
            vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(vector_store_path.parent, 0o755)
            
            # Create new store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=settings.VECTOR_STORE_PATH,
                collection_name=settings.COLLECTION_NAME
            )
            
            logger.info("✅ Vector store created successfully")
            
            # Initialize QA chain
            self._initialize_qa_chain()
            
            unique_docs = len(set([doc.metadata.get('source', '') for doc in all_documents]))
            
            return {
                "status": "success",
                "message": "Documents ingested successfully",
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
        """Load existing vector store"""
        try:
            logger.info("🔄 Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=settings.VECTOR_STORE_PATH,
                embedding_function=self.embeddings,
                collection_name=settings.COLLECTION_NAME
            )
            self._initialize_qa_chain()
            logger.info("✅ Vector store loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            return False
    
    def _initialize_qa_chain(self):
        """Initialize QA chain"""
        
        template = """You are a research assistant analyzing academic papers on secure file sharing, encryption, and related security topics.

Use the following retrieved context to answer the question thoroughly.
IMPORTANT: Always mention which specific papers/documents you are citing in your answer.

If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer (mention document names when citing):"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.TOP_K_RETRIEVAL}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question: str, top_k: int = 5) -> QueryResponse:
        """Query the RAG system"""
        if not self.qa_chain:
            if not self.load_vectorstore():
                raise ValueError("No vector store found. Please ingest documents first.")
        
        # Update retriever k value
        self.qa_chain.retriever.search_kwargs["k"] = top_k
        
        # Get answer
        logger.info(f"🔍 Querying with top_k={top_k}")
        result = self.qa_chain({"query": question})
        
        # Extract cited sources
        cited_sources = []
        for doc in result.get("source_documents", []):
            source_path = doc.metadata.get("source", "Unknown")
            source_name = Path(source_path).name if source_path != "Unknown" else "Unknown"
            
            cited_sources.append(CitedSource(
                document=source_name,
                page=doc.metadata.get("page", None),
                chunk_content=doc.page_content[:500],
                relevance_score=None
            ))
        
        response_id = str(uuid.uuid4())[:8]
        
        return QueryResponse(
            question=question,
            answer=result["result"],
            cited_sources=cited_sources,
            timestamp=datetime.now(),
            response_id=response_id
        )
    
    def list_documents(self) -> List[str]:
        """List all documents"""
        papers = list(settings.PAPERS_DIR.glob("*.pdf"))
        return [p.name for p in papers]
    
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.vectorstore:
            return {"status": "not_loaded"}
        
        try:
            collection = self.vectorstore._collection
            return {
                "status": "loaded",
                "total_chunks": collection.count(),
                "collection_name": settings.COLLECTION_NAME,
                "embedding_model": settings.EMBEDDING_MODEL
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Global instance
rag_engine = RAGEngine()
