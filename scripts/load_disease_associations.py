import json
from neo4j import GraphDatabase

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "drugsafety123")
IN_PATH = "/workspace/data/processed/drug_disease_associations.jsonl"
BATCH_SIZE = 2000

CYPHER_CAUSES = """
UNWIND $rows AS row
MERGE (d:Drug {name: row.drug_name})
MERGE (dis:Disease {name: row.disease_name})
MERGE (d)-[:CAUSES_CONDITION]->(dis)
"""

CYPHER_ASSOCIATED = """
UNWIND $rows AS row
MERGE (d:Drug {name: row.drug_name})
MERGE (dis:Disease {name: row.disease_name})
MERGE (d)-[:ASSOCIATED_WITH]->(dis)
"""


def load_pairs():
    with open(IN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def setup_disease_constraint(driver):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (dis:Disease) REQUIRE dis.name IS UNIQUE")
    print("Disease constraint ready.")


def migrate():
    pairs = load_pairs()
    causes_rows = [p for p in pairs if p["relation"] == "CAUSES"]
    associated_rows = [p for p in pairs if p["relation"] == "ASSOCIATED"]

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    setup_disease_constraint(driver)

    with driver.session() as session:
        print(f"Loading {len(causes_rows)} CAUSES_CONDITION relationships...")
        for i in range(0, len(causes_rows), BATCH_SIZE):
            batch = causes_rows[i:i + BATCH_SIZE]
            session.run(CYPHER_CAUSES, rows=batch)
        print("Done.")

        print(f"Loading {len(associated_rows)} ASSOCIATED_WITH relationships...")
        for i in range(0, len(associated_rows), BATCH_SIZE):
            batch = associated_rows[i:i + BATCH_SIZE]
            session.run(CYPHER_ASSOCIATED, rows=batch)
        print("Done.")

    driver.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
