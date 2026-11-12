from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.config import settings
from app.db import Database
from app.schemas import DocumentIn, QueryIn
from app.services.pipeline import GraphRAGService


def create_app(database_path: str | None = None) -> FastAPI:
    db=Database(database_path or settings.database_path)
    service=GraphRAGService(db)
    application=FastAPI(title=settings.app_name,version="0.1.0",description="Graph-enhanced retrieval with graph relationships, community context, and cited answers.")
    application.state.db=db
    application.state.service=service
    static_dir=Path(__file__).parent/"static"
    application.mount("/static",StaticFiles(directory=static_dir),name="static")

    @application.get("/",response_class=HTMLResponse,tags=["dashboard"])
    def dashboard():
        return (static_dir/"index.html").read_text(encoding="utf-8")

    @application.get("/health",tags=["meta"])
    def health():
        return {"status":"ok","app":settings.app_name}

    @application.post("/api/documents",tags=["knowledge base"])
    def ingest(payload:DocumentIn):
        try:
            result=service.ingest_document(payload.title,payload.content,payload.source)
            return {"status":"indexed",**result}
        except ValueError as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc

    @application.get("/api/documents",tags=["knowledge base"])
    def list_documents():
        return {"documents":service.list_documents()}

    @application.post("/api/query",tags=["GraphRAG"])
    def query(payload:QueryIn):
        try:
            return service.query(payload.question,payload.mode,payload.top_k,payload.max_hops)
        except Exception as exc:
            raise HTTPException(status_code=500,detail=f"Query failed: {type(exc).__name__}") from exc

    @application.get("/api/graph",tags=["knowledge graph"])
    def graph(limit:int=Query(default=200,ge=1,le=500)):
        return service.graph.graph(limit)

    @application.get("/api/history",tags=["GraphRAG"])
    def history(limit:int=Query(default=20,ge=1,le=100)):
        return {"runs":service.history(limit)}

    @application.post("/api/demo/load",tags=["demo"])
    def load_demo():
        if service.list_documents():
            return {"status":"already_loaded","documents":service.list_documents()}
        examples=Path(__file__).parent.parent/"examples"
        for path in sorted(examples.glob("*.md")):
            service.ingest_document(path.stem.replace("_"," ").title(),path.read_text(encoding="utf-8"),f"example://{path.name}")
        return {"status":"loaded","documents":service.list_documents()}

    return application

app=create_app()
