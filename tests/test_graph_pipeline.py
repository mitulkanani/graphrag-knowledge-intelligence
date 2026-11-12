from app.db import Database
from app.services.pipeline import GraphRAGService

DOC = "Acme Robotics partnered with Northstar Logistics. Northstar Logistics collaborates with Helios Systems. Helios Systems works with Atlas Platform. Acme Robotics uses Atlas Platform to track shipment events."


def service(tmp_path):
    return GraphRAGService(Database(str(tmp_path / "graph.sqlite3")))


def test_ingestion_extracts_entities_relations_and_communities(tmp_path):
    svc=service(tmp_path)
    result=svc.ingest_document("Acme network",DOC,"unit://acme")
    graph=svc.graph.graph()
    names={node["name"] for node in graph["nodes"]}
    assert result["chunk_count"] >= 1
    assert "Acme Robotics" in names
    assert "Northstar Logistics" in names
    assert result["relation_count"] >= 3
    assert graph["communities"]


def test_local_retrieval_finds_multi_hop_relationships(tmp_path):
    svc=service(tmp_path)
    svc.ingest_document("Acme network",DOC,"unit://acme")
    result=svc.query("How are Acme Robotics and Helios Systems connected?",mode="local",max_hops=2)
    assert result["citations"]
    all_entities=[name for path in result["graph_paths"] for name in path["entities"]]
    assert "Acme Robotics" in all_entities
    assert "Helios Systems" in all_entities
    assert "S1" in result["answer"]


def test_global_mode_returns_community_context(tmp_path):
    svc=service(tmp_path)
    svc.ingest_document("Acme network",DOC,"unit://acme")
    result=svc.query("Summarize the Acme Logistics platform ecosystem",mode="global")
    assert result["mode"] == "global"
    assert result["citations"]
    assert result["retrieval_debug"]["retrieval_strategy"].startswith("BM25-inspired")


def test_query_history_is_persisted(tmp_path):
    svc=service(tmp_path)
    svc.ingest_document("Acme network",DOC,"unit://acme")
    svc.query("How does Acme use Atlas Platform?",mode="local")
    history=svc.history()
    assert len(history)==1
    assert "Acme" in history[0]["question"]


def test_empty_knowledge_base_returns_no_evidence_message(tmp_path):
    svc=service(tmp_path)
    result=svc.query("What is the relationship between Acme Robotics and Helios Systems?")
    assert result["citations"] == []
    assert "could not find relevant evidence" in result["answer"].lower()


def test_graph_paths_reference_a_returned_source_citation(tmp_path):
    svc=service(tmp_path)
    svc.ingest_document("Acme network",DOC,"unit://acme")
    result=svc.query("How are Acme Robotics and Helios Systems connected?",mode="local",max_hops=2)
    citation_ids={c["source_id"] for c in result["citations"]}
    cited_paths=[p for p in result["graph_paths"] if p["source_id"]]
    assert cited_paths
    assert all(p["source_id"] in citation_ids for p in cited_paths)


def test_optional_dense_embedding_retrieval_is_combined_with_graph(tmp_path):
    import json
    from app.services.retrieval import GraphRetriever
    db=Database(str(tmp_path/"vector.sqlite3"))
    svc=GraphRAGService(db)
    svc.ingest_document("Acme network",DOC,"unit://acme")
    class StubEmbeddings:
        def embed(self,texts):
            return [[1.0,0.0] for _ in texts]
    with db.session() as conn:
        rows=conn.execute("SELECT id FROM chunks").fetchall()
        for row in rows:
            conn.execute("INSERT INTO chunk_embeddings(chunk_id,model,vector_json,dimensions) VALUES(?,?,?,?)",(row["id"],"stub",json.dumps([1.0,0.0]),2))
    result=GraphRetriever(db,StubEmbeddings()).retrieve("How does Acme connect to Helios?",mode="local")
    assert result["debug"]["semantic_chunks"] > 0
    assert "dense vector" in result["debug"]["retrieval_strategy"]


def test_demo_markdown_headings_do_not_become_entities(tmp_path):
    svc=service(tmp_path)
    from pathlib import Path
    sample=Path(__file__).parent.parent/"examples"/"acme_ecosystem.md"
    svc.ingest_document("Acme ecosystem",sample.read_text(),"unit://demo")
    names={node["name"] for node in svc.graph.graph()["nodes"]}
    assert "Acme Ecosystem Overview" not in names
    assert "The Atlas Platform" not in names
