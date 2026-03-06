from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Query(BaseModel):
    question: str = Field(..., description="The research question")
    top_k: Optional[int] = Field(5, description="Number of sources to retrieve")

class CitedSource(BaseModel):
    document: str
    page: Optional[int] = None
    chunk_content: str
    relevance_score: Optional[float] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    cited_sources: List[CitedSource]
    timestamp: datetime = Field(default_factory=datetime.now)
    response_id: Optional[str] = None

class IngestionStatus(BaseModel):
    total_documents: int
    processed_documents: int
    total_chunks: int
    status: str
    message: str