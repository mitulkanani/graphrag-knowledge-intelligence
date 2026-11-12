// Reference schema for migrating the demo SQLite graph to Neo4j.
// Run against a Neo4j database after adapting constraints to your environment.
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

// Example relationship shape:
// (:Entity {id, name, entity_type})-[:PARTNERED_WITH {description, confidence}]->(:Entity)
// (:Chunk)-[:MENTIONS]->(:Entity)
// (:Chunk)-[:FROM_DOCUMENT]->(:Document)
// (:Entity)-[:IN_COMMUNITY]->(:Community)
