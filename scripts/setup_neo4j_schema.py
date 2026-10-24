"""Creates the Neo4j uniqueness constraints. With --reset it first deletes the whole graph
(use this before rebuilding the graph from scratch)."""
import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

CONSTRAINTS = [
    "CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT sideeffect_name_unique IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (dis:Disease) REQUIRE dis.name IS UNIQUE",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete every node and relationship first")
    args = parser.parse_args()

    driver = GraphDatabase.driver(cfg.NEO4J_URI, auth=cfg.NEO4J_AUTH)
    with driver.session() as session:
        if args.reset:
            print("Deleting the existing graph ...")
            session.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
            left = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            print(f"Graph cleared ({left} nodes left).")
        for c in CONSTRAINTS:
            session.run(c).consume()
    driver.close()
    print("Constraints created.")
