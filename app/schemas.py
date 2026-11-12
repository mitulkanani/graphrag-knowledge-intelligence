from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class DocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=2, max_length=200)
    content: str = Field(min_length=30, max_length=200_000)
    source: str | None = Field(default=None, max_length=500)


class DocumentOut(BaseModel):
    id: str
    title: str
    source: str | None
    chunk_count: int
    entity_count: int
    relation_count: int
    created_at: datetime


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=5, max_length=2000)
    mode: Literal["auto", "local", "global"] = "auto"
    top_k: int = Field(default=8, ge=1, le=20)
    max_hops: int = Field(default=2, ge=1, le=3)


class Citation(BaseModel):
    source_id: str
    document_id: str
    title: str
    source: str | None = None
    chunk_id: str
    excerpt: str


class GraphPath(BaseModel):
    entities: list[str]
    relationships: list[str]
    source_id: str | None = None


class QueryOut(BaseModel):
    run_id: str
    question: str
    mode: str
    answer: str
    citations: list[Citation]
    graph_paths: list[GraphPath]
    communities: list[str]
    retrieval_debug: dict[str, Any]
    created_at: datetime


class GraphNode(BaseModel):
    id: str
    name: str
    entity_type: str
    community_id: str | None = None
    degree: int = 0


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation_type: str
    description: str
    source_document_id: str


class GraphOut(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    communities: list[dict[str, Any]]
