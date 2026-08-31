import psycopg2
from neo4j import GraphDatabase

PG_CONFIG = dict(host="postgres", dbname="drug_safety_db", user="drug_user", password="drug_password")
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "drugsafety123")
BATCH_SIZE = 5000

CYPHER_MERGE = """
UNWIND $rows AS row
MERGE (d:Drug {name: row.drug_name})
MERGE (s:SideEffect {name: row.side_effect_name})
MERGE (d)-[:CAUSES]->(s)
"""

def fetch_rows(pg_conn):
    cur = pg_conn.cursor()
    cur.execute("SELECT drug_name, side_effect_name FROM drug_side_effects_clean;")
    return cur.fetchall()

def migrate():
    pg_conn = psycopg2.connect(**PG_CONFIG)
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    rows = fetch_rows(pg_conn)
    total = len(rows)
    print(f"Migrating {total} drug-side_effect pairs...")

    with driver.session() as session:
        for i in range(0, total, BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            batch_dicts = [{"drug_name": r[0], "side_effect_name": r[1]} for r in batch]
            session.run(CYPHER_MERGE, rows=batch_dicts)
            print(f"  {min(i + BATCH_SIZE, total)}/{total} migrated")

    pg_conn.close()
    driver.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
