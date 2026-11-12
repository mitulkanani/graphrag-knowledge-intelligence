from fastapi.testclient import TestClient
from app.main import create_app

DOC="Acme Robotics partnered with Northstar Logistics. Northstar Logistics collaborates with Helios Systems. Helios Systems works with Atlas Platform."


def test_api_document_ingestion_query_graph_and_history(tmp_path):
    app=create_app(str(tmp_path/"api.sqlite3"))
    with TestClient(app) as client:
        response=client.post("/api/documents",json={"title":"Acme network","content":DOC,"source":"unit://acme"})
        assert response.status_code==200
        assert response.json()["relation_count"]>=2
        q=client.post("/api/query",json={"question":"How are Acme Robotics and Helios Systems connected?","mode":"local"})
        assert q.status_code==200
        assert q.json()["citations"]
        assert client.get("/api/graph").json()["nodes"]
        assert client.get("/api/history").json()["runs"]


def test_api_validation_rejects_short_document_and_question(tmp_path):
    app=create_app(str(tmp_path/"api.sqlite3"))
    with TestClient(app) as client:
        assert client.post("/api/documents",json={"title":"x","content":"short"}).status_code==422
        assert client.post("/api/query",json={"question":"hi"}).status_code==422


def test_demo_load_is_idempotent(tmp_path):
    app=create_app(str(tmp_path/"api.sqlite3"))
    with TestClient(app) as client:
        first=client.post("/api/demo/load")
        second=client.post("/api/demo/load")
        assert first.json()["status"]=="loaded"
        assert second.json()["status"]=="already_loaded"


def test_demo_load_builds_a_nonempty_knowledge_graph(tmp_path):
    app=create_app(str(tmp_path/"demo.sqlite3"))
    with TestClient(app) as client:
        loaded=client.post("/api/demo/load").json()
        graph=client.get("/api/graph").json()
        assert loaded["status"]=="loaded"
        assert len(graph["nodes"]) >= 5
        assert len(graph["edges"]) >= 5
