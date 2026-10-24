"""Loads model-predicted drug-disease relations into Neo4j as MODEL_PREDICTED edges."""
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

IN_PATH = cfg.PROCESSED_DIR / "model_predicted_associations.jsonl"
BATCH_SIZE = 2000
CYPHER = """
UNWIND $rows AS row
MERGE (d:Drug {name: row.drug_name})
MERGE (dis:Disease {name: row.disease_name})
MERGE (d)-[r:MODEL_PREDICTED {relation: row.relation}]->(dis)
SET r.confidence = row.confidence, r.mean_confidence = row.mean_confidence,
    r.support = row.support, r.model = row.model
"""

if __name__ == "__main__":
    with open(IN_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r.pop("evidence", None)

    driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=cfg.NEO4J_AUTH)
    with driver.session() as session:
        session.run("MATCH ()-[r:MODEL_PREDICTED]->() CALL { WITH r DELETE r } IN TRANSACTIONS OF 10000 ROWS").consume()
        for i in range(0, len(rows), BATCH_SIZE):
            session.run(CYPHER, rows=rows[i:i + BATCH_SIZE]).consume()
            print(f"  {min(i + BATCH_SIZE, len(rows)):,}/{len(rows):,} loaded")
    driver.close()
    print("Done.")
