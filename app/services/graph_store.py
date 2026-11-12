from collections import defaultdict, deque
from datetime import UTC, datetime
from uuid import uuid4
from app.db import Database
from app.services.extraction import extract_knowledge

class GraphStore:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def now():
        return datetime.now(UTC).isoformat()

    def ingest(self, title: str, content: str, source: str | None, chunks: list[dict[str, str]]) -> dict:
        from uuid import uuid4
        document_id = str(uuid4())
        entity_count = relation_count = 0
        with self.db.session() as conn:
            conn.execute("INSERT INTO documents(id,title,source,content,created_at) VALUES(?,?,?,?,?)", (document_id, title, source, content, self.now()))
            for idx, chunk in enumerate(chunks):
                chunk_id = chunk["id"]
                conn.execute("INSERT INTO chunks(id,document_id,chunk_index,content,created_at) VALUES(?,?,?,?,?)", (chunk_id, document_id, idx, chunk["content"], self.now()))
                entities, extracted_relations = extract_knowledge(chunk["content"])
                # Reuse canonical entities across all documents.
                resolved = []
                for entity in entities:
                    row = conn.execute("SELECT id,name,normalized_name,entity_type FROM entities WHERE normalized_name=?", (entity.normalized_name,)).fetchone()
                    if row:
                        entity_id = row["id"]
                    else:
                        entity_id = entity.id
                        conn.execute("INSERT INTO entities(id,name,normalized_name,entity_type,created_at) VALUES(?,?,?,?,?)", (entity_id, entity.name, entity.normalized_name, entity.entity_type, self.now()))
                    resolved.append(type("ResolvedEntity", (), {"id":entity_id,"name":entity.name,"normalized_name":entity.normalized_name,"entity_type":entity.entity_type})())
                    conn.execute("INSERT OR IGNORE INTO chunk_entities(chunk_id,entity_id) VALUES(?,?)", (chunk_id, entity_id))
                entity_count += len(resolved)
                resolved_by_norm = {x.normalized_name: x for x in resolved}
                original_by_id = {x.id: x for x in entities}
                # Rebind extracted relation endpoint IDs to canonical entity IDs.
                relations = []
                for rel in extracted_relations:
                    original_source = original_by_id.get(rel.source)
                    original_target = original_by_id.get(rel.target)
                    source_entity = resolved_by_norm.get(original_source.normalized_name) if original_source else None
                    target_entity = resolved_by_norm.get(original_target.normalized_name) if original_target else None
                    if source_entity and target_entity:
                        rel.source = source_entity.id
                        rel.target = target_entity.id
                        relations.append(rel)
                for rel in relations:
                    conn.execute("INSERT INTO relations(id,source_entity_id,target_entity_id,relation_type,description,source_chunk_id,document_id,confidence) VALUES(?,?,?,?,?,?,?,?)", (rel.id, rel.source, rel.target, rel.relation_type, rel.description, chunk_id, document_id, rel.confidence))
                relation_count += len(relations)
        self.rebuild_communities()
        return {"id": document_id, "chunk_count": len(chunks), "entity_count": entity_count, "relation_count": relation_count}

    def rebuild_communities(self) -> None:
        with self.db.session() as conn:
            entities = conn.execute("SELECT id,name FROM entities ORDER BY name").fetchall()
            edges = conn.execute("SELECT source_entity_id,target_entity_id,relation_type FROM relations").fetchall()
            adjacency = defaultdict(set)
            for edge in edges:
                adjacency[edge["source_entity_id"]].add(edge["target_entity_id"])
                adjacency[edge["target_entity_id"]].add(edge["source_entity_id"])
            names = {e["id"]: e["name"] for e in entities}
            visited = set()
            groups = []
            for entity in entities:
                start = entity["id"]
                if start in visited:
                    continue
                queue = deque([start]); visited.add(start); group=[]
                while queue:
                    node = queue.popleft(); group.append(node)
                    for neighbor in adjacency[node]:
                        if neighbor not in visited:
                            visited.add(neighbor); queue.append(neighbor)
                groups.append(group)
            conn.execute("DELETE FROM community_entities")
            conn.execute("DELETE FROM communities")
            for group in groups:
                cid = str(uuid4())
                label = ", ".join(names[x] for x in sorted(group, key=lambda k:names[k].lower())[:4])
                rels = conn.execute("SELECT DISTINCT relation_type FROM relations WHERE source_entity_id IN (%s) OR target_entity_id IN (%s)" % (','.join('?'*len(group)), ','.join('?'*len(group))), tuple(group)+tuple(group)).fetchall() if group else []
                rel_labels = ", ".join(x["relation_type"].lower().replace("_", " ") for x in rels[:5])
                summary = f"Connected topic group containing {label}." + (f" Observed relationship types: {rel_labels}." if rel_labels else "")
                conn.execute("INSERT INTO communities(id,label,summary,entity_count,created_at) VALUES(?,?,?,?,?)", (cid,label,summary,len(group),self.now()))
                conn.executemany("INSERT INTO community_entities(community_id,entity_id) VALUES(?,?)", [(cid,x) for x in group])

    def graph(self, limit: int = 200) -> dict:
        with self.db.session() as conn:
            nodes = conn.execute("""SELECT e.id,e.name,e.entity_type,ce.community_id,
                (SELECT COUNT(*) FROM relations r WHERE r.source_entity_id=e.id OR r.target_entity_id=e.id) AS degree
                FROM entities e LEFT JOIN community_entities ce ON ce.entity_id=e.id
                ORDER BY degree DESC,e.name LIMIT ?""", (limit,)).fetchall()
            edges = conn.execute("""SELECT r.id,r.source_entity_id,r.target_entity_id,r.relation_type,r.description,r.document_id
                FROM relations r ORDER BY r.confidence DESC LIMIT ?""", (limit*2,)).fetchall()
            communities = conn.execute("SELECT id,label,summary,entity_count FROM communities ORDER BY entity_count DESC LIMIT 50").fetchall()
            return {"nodes":[dict(x) for x in nodes],"edges":[{"id":e["id"],"source":e["source_entity_id"],"target":e["target_entity_id"],"relation_type":e["relation_type"],"description":e["description"],"source_document_id":e["document_id"]} for e in edges],"communities":[dict(x) for x in communities]}
