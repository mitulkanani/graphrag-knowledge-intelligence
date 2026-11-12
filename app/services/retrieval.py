import math
import re
import json
from collections import Counter, defaultdict, deque
from app.db import Database
from app.services.embedding import cosine_similarity

STOP = {"the","a","an","and","or","to","of","in","on","for","with","what","who","how","does","do","is","are","was","were","by","from","about","between","which","that","this","their","they","it","as","at","be","has","have"}

def terms(text: str) -> list[str]:
    return [x for x in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(x)>1 and x not in STOP]

class GraphRetriever:
    def __init__(self, db: Database, embedding_provider=None):
        self.db = db
        self.embedding_provider = embedding_provider

    def retrieve(self, question: str, mode: str="auto", top_k: int=8, max_hops: int=2) -> dict:
        query_terms = terms(question)
        with self.db.session() as conn:
            chunks = conn.execute("""SELECT c.id,c.document_id,c.content,d.title,d.source,ce.vector_json
                FROM chunks c JOIN documents d ON d.id=c.document_id
                LEFT JOIN chunk_embeddings ce ON ce.chunk_id=c.id""").fetchall()
            entities = conn.execute("SELECT id,name,normalized_name FROM entities").fetchall()
            relations = conn.execute("SELECT r.*,se.name AS source_name,te.name AS target_name,d.title FROM relations r JOIN entities se ON se.id=r.source_entity_id JOIN entities te ON te.id=r.target_entity_id JOIN documents d ON d.id=r.document_id").fetchall()
            communities = conn.execute("SELECT c.id,c.label,c.summary,c.entity_count FROM communities c").fetchall()
            community_entities = conn.execute("SELECT community_id,entity_id FROM community_entities").fetchall()
        if not chunks:
            return {"chunks":[],"paths":[],"communities":[],"debug":{"reason":"No documents have been indexed.","mode":"empty"}}

        # BM25-inspired lexical ranking (compact implementation for the demo).
        document_tokens = {row["id"]: terms(row["content"]) for row in chunks}
        df = Counter(token for token in query_terms if any(token in tokens for tokens in document_tokens.values()))
        lexical_scores = {}
        for row in chunks:
            tokens = document_tokens[row["id"]]
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                tf = counts[term]
                if tf:
                    idf = math.log(1 + (len(chunks) - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * (tf * 2.2) / (tf + 1.2)
            lexical_scores[row["id"]] = score

        # Optional dense vector retrieval complements lexical search.
        semantic_scores = {}
        if self.embedding_provider:
            query_vector = self.embedding_provider.embed([question])[0]
            raw = {}
            for row in chunks:
                if row["vector_json"]:
                    raw[row["id"]] = max(0.0, cosine_similarity(query_vector, json.loads(row["vector_json"])))
            maximum = max(raw.values(), default=0.0) or 1.0
            semantic_scores = {key:value/maximum for key,value in raw.items()}

        # Resolve seed entities via normalized name/token overlap.
        seeds = []
        qnorm = " ".join(query_terms)
        for entity in entities:
            ename = entity["normalized_name"]
            name_tokens = terms(ename)
            if ename and (ename in qnorm or (name_tokens and all(t in query_terms for t in name_tokens))):
                seeds.append(entity["id"])
        if mode == "auto":
            selected_mode = "local" if seeds else "global"
        else:
            selected_mode = mode

        graph_chunk_scores = defaultdict(float)
        paths = []
        community_ids = set()
        if selected_mode == "local" and seeds:
            adjacency = defaultdict(list)
            for rel in relations:
                adjacency[rel["source_entity_id"]].append((rel["target_entity_id"], rel))
                adjacency[rel["target_entity_id"]].append((rel["source_entity_id"], rel))
            entity_by_id = {e["id"]:e for e in entities}
            queue = deque((seed, [seed], []) for seed in seeds)
            visited_depth = {(seed,0) for seed in seeds}
            while queue:
                node, path_nodes, path_rels = queue.popleft()
                depth = len(path_rels)
                if depth >= max_hops:
                    continue
                for neighbor, rel in adjacency[node]:
                    if neighbor in path_nodes:
                        continue
                    new_nodes = path_nodes + [neighbor]
                    new_rels = path_rels + [rel]
                    edge_score = 1.0 / (depth + 1)
                    graph_chunk_scores[rel["source_chunk_id"]] += edge_score
                    paths.append({"entities":[entity_by_id[x]["name"] for x in new_nodes],"relationships":[r["relation_type"] if r["source_entity_id"] == path_nodes[i] else "REVERSE_" + r["relation_type"] for i,r in enumerate(new_rels)],"source_chunk_ids":[r["source_chunk_id"] for r in new_rels]})
                    state = (neighbor, depth+1)
                    if state not in visited_depth:
                        visited_depth.add(state); queue.append((neighbor,new_nodes,new_rels))
            entity_to_communities = defaultdict(set)
            for row in community_entities:
                entity_to_communities[row["entity_id"]].add(row["community_id"])
            for seed in seeds:
                community_ids.update(entity_to_communities[seed])
        elif selected_mode == "global":
            for community in communities:
                score = sum(1 for term in query_terms if term in terms(community["summary"] + " " + community["label"]))
                if score:
                    community_ids.add(community["id"])
            member_ids = {x["entity_id"] for x in community_entities if x["community_id"] in community_ids}
            for rel in relations:
                if rel["source_entity_id"] in member_ids or rel["target_entity_id"] in member_ids:
                    graph_chunk_scores[rel["source_chunk_id"]] += 0.65

        # Candidate score is normalized lexical + graph evidence; require at least one signal.
        max_lex = max(lexical_scores.values(), default=0.0) or 1.0
        max_graph = max(graph_chunk_scores.values(), default=0.0) or 1.0
        ranked = []
        chunk_by_id = {x["id"]:x for x in chunks}
        for chunk_id, lex in lexical_scores.items():
            graph_score = graph_chunk_scores[chunk_id]
            lex_norm = lex/max_lex
            graph_norm = graph_score/max_graph if graph_score else 0
            if semantic_scores:
                semantic_norm = semantic_scores.get(chunk_id, 0.0)
                score = 0.34*lex_norm + 0.33*semantic_norm + 0.33*graph_norm
            else:
                score = 0.58*lex_norm + 0.42*graph_norm
            if lex > 0 or graph_score > 0 or semantic_scores.get(chunk_id, 0.0) > 0:
                ranked.append((score, chunk_by_id[chunk_id]))
        ranked.sort(key=lambda item:item[0], reverse=True)
        chosen = []
        for score,row in ranked[:top_k]:
            chosen.append({"chunk_id":row["id"],"document_id":row["document_id"],"title":row["title"],"source":row["source"],"content":row["content"],"score":round(score,4)})
        community_results = [dict(x) for x in communities if x["id"] in community_ids]
        # Keep source chunks for the selected paths in the evidence set where possible.
        selected_chunk_ids = {x["chunk_id"] for x in chosen}
        path_chunk_ids = []
        for path in paths:
            for chunk_id in path.get("source_chunk_ids", []):
                if chunk_id not in selected_chunk_ids and chunk_id not in path_chunk_ids:
                    path_chunk_ids.append(chunk_id)
        for chunk_id in path_chunk_ids[:4]:
            row = chunk_by_id.get(chunk_id)
            if row:
                chosen.append({"chunk_id":row["id"],"document_id":row["document_id"],"title":row["title"],"source":row["source"],"content":row["content"],"score":0.0})
                selected_chunk_ids.add(chunk_id)
        citation_by_chunk = {item["chunk_id"]: f"S{idx+1}" for idx,item in enumerate(chosen)}
        for path in paths:
            path_chunks = path.pop("source_chunk_ids", [])
            path["source_id"] = next((citation_by_chunk[cid] for cid in path_chunks if cid in citation_by_chunk), None)
        # Keep a small set of high-value graph paths and remove duplicate sequences.
        seen_paths = set(); unique_paths=[]
        for path in sorted(paths,key=lambda p:len(p["relationships"])):
            key=(tuple(path["entities"]),tuple(path["relationships"]))
            if key not in seen_paths:
                seen_paths.add(key); unique_paths.append(path)
            if len(unique_paths)>=20: break
        debug={"mode":selected_mode,"seed_entities":[e["name"] for e in entities if e["id"] in seeds],"candidate_chunks":len(ranked),"graph_expansion_chunks":len(graph_chunk_scores),"semantic_chunks":len(semantic_scores),"retrieval_strategy":"BM25-inspired lexical + dense vector + graph traversal/community context" if semantic_scores else "BM25-inspired lexical ranking + graph traversal/community context","max_hops":max_hops}
        return {"chunks":chosen,"paths":unique_paths,"communities":community_results,"debug":debug}
