"""Loads SIDER drug -> side-effect pairs (preferred terms) from PostgreSQL into Neo4j."""
import sys
from pathlib import Path

import psycopg2
from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

BATCH_SIZE = 5000
CYPHER = """
UNWIND $rows AS row
MERGE (d:Drug {name: row.drug_name})
MERGE (s:SideEffect {name: row.side_effect_name})
MERGE (d)-[r:CAUSES]->(s)
SET r.source = 'SIDER'
"""

if __name__ == "__main__":
    pg = psycopg2.connect(**cfg.POSTGRES)
    cur = pg.cursor()
    cur.execute("SELECT drug_name, side_effect_name FROM drug_side_effects_clean")
    rows = [{"drug_name": a, "side_effect_name": b} for a, b in cur.fetchall()]
    pg.close()

    print(f"Migrating {len(rows):,} drug-side-effect pairs ...")
    driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=cfg.NEO4J_AUTH)
    with driver.session() as session:
        for i in range(0, len(rows), BATCH_SIZE):
            session.run(CYPHER, rows=rows[i:i + BATCH_SIZE]).consume()
            print(f"  {min(i + BATCH_SIZE, len(rows)):,}/{len(rows):,}")
    driver.close()
    print("Migration complete.")
