import json
from neo4j import GraphDatabase

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "drugsafety123")
IN_PATH = "/workspace/data/processed/model_predicted_associations.jsonl"
BATCH_SIZE = 2000

CYPHER = """
UNWIND $rows AS row
MERGE (d:Drug {name: row.drug_name})
MERGE (dis:Disease {name: row.disease_name})
MERGE (d)-[r:MODEL_PREDICTED {relation: row.relation}]->(dis)
SET r.confidence = row.confidence, r.model = row.model
"""


def load_rows():
    with open(IN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def migrate():
    rows = load_rows()
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            session.run(CYPHER, rows=batch)
            print(f"  {min(i + BATCH_SIZE, len(rows))}/{len(rows)} loaded")
    driver.close()
    print("Done.")


if __name__ == "__main__":
    migrate()
