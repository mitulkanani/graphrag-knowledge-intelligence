import json
from datetime import UTC, datetime
from uuid import uuid4
from app.config import settings
from app.db import Database
from app.services.answering import get_answer_provider
from app.services.chunking import chunk_text
from app.services.graph_store import GraphStore
from app.services.embedding import get_embedding_provider
from app.services.retrieval import GraphRetriever

class GraphRAGService:
    def __init__(self, db: Database):
        self.db=db
        self.graph=GraphStore(db)
        self.embedding_provider=get_embedding_provider()
        self.retriever=GraphRetriever(db,self.embedding_provider)

    def ingest_document(self,title:str,content:str,source:str|None=None)->dict:
        chunks=chunk_text(content,settings.max_chunk_chars,settings.chunk_overlap_chars)
        if not chunks:
            raise ValueError("Document content is empty after normalization.")
        result=self.graph.ingest(title,content,source,chunks)
        if self.embedding_provider:
            vectors=self.embedding_provider.embed([x["content"] for x in chunks])
            with self.db.session() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO chunk_embeddings(chunk_id,model,vector_json,dimensions) VALUES(?,?,?,?)",
                    [(chunk["id"],settings.embedding_model,json.dumps(vector),len(vector)) for chunk,vector in zip(chunks,vectors)],
                )
            result["embedding_count"]=len(vectors)
        else:
            result["embedding_count"]=0
        return result

    def list_documents(self)->list[dict]:
        with self.db.session() as conn:
            docs=conn.execute("""SELECT d.id,d.title,d.source,d.created_at,
                (SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.id) AS chunk_count,
                (SELECT COUNT(DISTINCT ce.entity_id) FROM chunk_entities ce JOIN chunks c ON c.id=ce.chunk_id WHERE c.document_id=d.id) AS entity_count,
                (SELECT COUNT(*) FROM relations r WHERE r.document_id=d.id) AS relation_count
                FROM documents d ORDER BY d.created_at DESC""").fetchall()
            return [dict(x) for x in docs]

    def query(self,question:str,mode:str="auto",top_k:int=8,max_hops:int=2)->dict:
        retrieved=self.retriever.retrieve(question,mode,top_k,max_hops)
        evidence=retrieved["chunks"]
        citations=[]
        for idx,item in enumerate(evidence,1):
            citations.append({"source_id":f"S{idx}","document_id":item["document_id"],"title":item["title"],"source":item["source"],"chunk_id":item["chunk_id"],"excerpt":item["content"][:500]})
        provider=get_answer_provider()
        answer=provider.answer(question,evidence,retrieved["paths"],retrieved["communities"])
        run_id=str(uuid4()); created_at=datetime.now(UTC).isoformat()
        with self.db.session() as conn:
            conn.execute("INSERT INTO query_runs(id,question,mode,answer,citations_json,paths_json,debug_json,created_at) VALUES(?,?,?,?,?,?,?,?)",(run_id,question,retrieved["debug"].get("mode",mode),answer,json.dumps(citations),json.dumps(retrieved["paths"]),json.dumps(retrieved["debug"]),created_at))
        return {"run_id":run_id,"question":question,"mode":retrieved["debug"].get("mode",mode),"answer":answer,"citations":citations,"graph_paths":retrieved["paths"],"communities":[x["summary"] for x in retrieved["communities"]],"retrieval_debug":retrieved["debug"],"created_at":created_at}

    def history(self,limit:int=20)->list[dict]:
        with self.db.session() as conn:
            rows=conn.execute("SELECT id,question,mode,answer,created_at FROM query_runs ORDER BY created_at DESC LIMIT ?",(limit,)).fetchall()
            return [dict(x) for x in rows]
