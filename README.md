# GraphRAG Knowledge Intelligence

A practical GraphRAG starter that turns documents into an entity-relationship graph and retrieves evidence through a combination of lexical ranking, local graph traversal, and community context. Answers include source citations.

> **Scope:** this repository includes a runnable SQLite-backed graph engine and a deterministic answer provider. Optional OpenAI answer generation is included. Entity and relationship extraction is heuristic in the default mode; it is not a production-grade LLM extraction pipeline or Microsoft's full GraphRAG implementation.

## Features

- Document ingestion and overlapping chunking.
- Heuristic entity and relationship extraction with source-chunk provenance.
- Canonical entity reuse across documents.
- Graph neighborhoods and bounded multi-hop traversal.
- Connected-component community grouping and global community-context retrieval.
- Hybrid retrieval: BM25-inspired lexical ranking + graph-expansion scores; optional OpenAI dense embeddings can be enabled for vector similarity.
- Source-grounded answers with `[S1]`, `[S2]` citations.
- Query history and graph inspection API.
- Browser dashboard for ingestion, questions, graph nodes/edges, and communities.
- Optional OpenAI entity/relation extraction and answer generation; offline heuristic/mock providers for development and CI.
- Neo4j schema reference for future migration.

## Architecture

```text
Documents
   |
   v
Chunker -> Entity / Relation Extractor -> SQLite Knowledge Graph
                                             |
Question -> Query Mode Router ----------------+
              |                               |
              +--> Local Search: entity seeds + bounded graph traversal
              +--> Global Search: community summaries + related evidence
              +--> Lexical Search: BM25-inspired chunk relevance
                                             |
                                             v
                                   Ranked evidence + paths
                                             |
                                             v
                                  Answer provider (mock/OpenAI)
                                             |
                                             v
                              Cited answer + graph context + history
```

The architecture borrows the high-level GraphRAG concepts of entity/relationship extraction, graph communities, and local/global query modes. The community algorithm here is connected components—not Leiden—and the lexical scorer is a compact BM25-inspired implementation. Optional OpenAI dense embeddings add cosine-similarity retrieval, but the starter does not require or ship a dedicated vector database. Graph paths include citations to their originating chunks when those sources are returned.

## Quick start

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

Open:
- Dashboard: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Click **Load demo knowledge** to index the three example documents.

## Ask a question

```bash
curl -X POST http://127.0.0.1:8000/api/query \\
  -H "Content-Type: application/json" \\
  -d '{"question":"How are Acme Robotics and Helios Systems connected?","mode":"auto","top_k":8,"max_hops":2}'
```

Try:
- `How are Acme Robotics and Helios Systems connected?`
- `What technologies are used in the supply chain?`
- `Which organizations collaborate with Beacon Analytics?`
- `Summarize the Acme ecosystem.` (global mode)

## API overview

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/documents` | Index document text |
| GET | `/api/documents` | List indexed documents |
| POST | `/api/query` | Ask a GraphRAG question |
| GET | `/api/graph` | Inspect nodes, edges, communities |
| GET | `/api/history` | Recent query history |
| POST | `/api/demo/load` | Load bundled demo documents |

## Optional OpenAI answer provider

Install the optional dependency:

```bash
python -m pip install -e ".[dev,openai]"
```

Set in `.env`:

```env
EXTRACTION_PROVIDER=openai
EMBEDDING_PROVIDER=openai
ANSWER_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
```

The extraction model is instructed to treat source text as untrusted data and extract only explicit facts. When enabled, dense embeddings are stored per chunk and added to the lexical + graph score. The answer model is instructed to answer only from supplied evidence and cite only source IDs provided by the application. Validate generated triples and answers before using them in high-stakes workflows.

## Run with Docker

```bash
cp .env.example .env
docker compose up --build
```

SQLite data is persisted in the `graphrag_data` Docker volume.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest --cov=app --cov-report=term-missing
```

## Data model

- `documents`: original text and source metadata.
- `chunks`: overlapping text units.
- `entities`: canonical entities.
- `relations`: directed relationships and source provenance.
- `chunk_entities`: mention index.
- `communities`: graph connected components with generated summaries.
- `community_entities`: community membership.
- `query_runs`: question, answer, citations, paths, and retrieval diagnostics.

## Important limitations and production roadmap

- The default entity extraction uses capitalization heuristics and a small relation-phrase dictionary. Optional OpenAI extraction is included for stronger extraction, but ambiguous names, pronouns, aliases, and complex relation statements still need entity resolution and human validation.
- Community detection uses connected components. For large graphs, use Leiden or another graph clustering algorithm and hierarchical community summaries.
- Default retrieval combines BM25-inspired lexical ranking with graph traversal. Optional OpenAI embeddings add dense cosine similarity. A dedicated vector database, reranking, and GraphRAG DRIFT search are future enhancements.
- SQLite is suitable for this demo, not large multi-user workloads. Use Neo4j or another graph database for production graph operations, and a vector store for semantic retrieval.
- The included Neo4j Cypher file is a schema reference, not a wired Neo4j persistence adapter.
- OpenAI extraction, embeddings, and answer generation are optional; the default extractor and answer provider are deterministic and do not call a model.
- Add document-level authorization, tenant isolation, encryption, retention policies, ingestion limits, observability, and background jobs before production use.
- Treat documents as untrusted content. They may contain prompt injection or misleading claims; do not let indexed text grant tool permissions.

## License

MIT. See [LICENSE](LICENSE).
