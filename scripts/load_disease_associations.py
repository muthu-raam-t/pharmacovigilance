"""Loads gold drug-disease relations into Neo4j:
CAUSES -> CAUSES_CONDITION, TREATS -> TREATS_CONDITION, ASSOCIATED -> ASSOCIATED_WITH."""
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

IN_PATH = cfg.PROCESSED_DIR / "drug_disease_associations.jsonl"
BATCH_SIZE = 2000
EDGE_TYPES = {"CAUSES": "CAUSES_CONDITION", "TREATS": "TREATS_CONDITION", "ASSOCIATED": "ASSOCIATED_WITH"}

if __name__ == "__main__":
    with open(IN_PATH, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=cfg.NEO4J_AUTH)
    with driver.session() as session:
        session.run("CREATE CONSTRAINT disease_name_unique IF NOT EXISTS "
                    "FOR (dis:Disease) REQUIRE dis.name IS UNIQUE").consume()
        for relation, edge in EDGE_TYPES.items():
            rows = [p for p in pairs if p["relation"] == relation]
            cypher = f"""
            UNWIND $rows AS row
            MERGE (d:Drug {{name: row.drug_name}})
            MERGE (dis:Disease {{name: row.disease_name}})
            MERGE (d)-[r:{edge}]->(dis)
            SET r.support = row.support, r.sources = row.source_datasets
            """
            for i in range(0, len(rows), BATCH_SIZE):
                session.run(cypher, rows=rows[i:i + BATCH_SIZE]).consume()
            print(f"Loaded {len(rows):,} {edge} relationships")
    driver.close()
    print("Done.")
