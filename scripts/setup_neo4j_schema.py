from neo4j import GraphDatabase

URI = "bolt://neo4j:7687"
AUTH = ("neo4j", "drugsafety123")

def setup_constraints(driver):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT drug_name_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
        session.run("CREATE CONSTRAINT sideeffect_name_unique IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE")
        print("Constraints created.")

if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=AUTH)
    setup_constraints(driver)
    driver.close()
