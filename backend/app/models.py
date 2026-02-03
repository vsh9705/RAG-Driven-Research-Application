from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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

class ManualEvaluation(BaseModel):
    response_id: str
    question: str
    answer: str
    cited_sources: List[Dict[str, Any]]  # Accept as dict for flexibility
    relevance_score: int = Field(..., ge=1, le=5, description="1-5: How relevant are the sources?")
    hallucination_score: int = Field(..., ge=1, le=5, description="1-5: How accurate are citations? (1=many errors, 5=perfect)")
    completeness_score: Optional[int] = Field(None, ge=1, le=5, description="1-5: How complete is the answer?")
    faithfulness_score: Optional[int] = Field(None, ge=1, le=5, description="1-5: Is answer grounded in sources?")
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class IngestionStatus(BaseModel):
    total_documents: int
    processed_documents: int
    total_chunks: int
    status: str
    message: str

class EvaluationStats(BaseModel):
    total_evaluations: int
    avg_relevance: float
    avg_hallucination: float
    avg_completeness: Optional[float] = None
    avg_faithfulness: Optional[float] = None
    evaluations: List[ManualEvaluation]