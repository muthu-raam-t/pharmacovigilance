from neo4j import GraphDatabase

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "drugsafety123")


class KnowledgeGraphClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    def close(self):
        self.driver.close()

    def get_side_effects_for_drug(self, drug_name):
        drug_name = drug_name.strip().lower()
        query = """
        MATCH (d:Drug {name: $drug_name})-[:CAUSES]->(s:SideEffect)
        RETURN s.name AS side_effect
        """
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name)
            return [record["side_effect"] for record in result]

    def get_drugs_causing_side_effect(self, side_effect_name):
        side_effect_name = side_effect_name.strip().lower()
        query = """
        MATCH (d:Drug)-[:CAUSES]->(s:SideEffect {name: $side_effect_name})
        RETURN d.name AS drug
        """
        with self.driver.session() as session:
            result = session.run(query, side_effect_name=side_effect_name)
            return [record["drug"] for record in result]

    def drug_exists(self, drug_name):
        drug_name = drug_name.strip().lower()
        query = "MATCH (d:Drug {name: $drug_name}) RETURN count(d) AS cnt"
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name)
            return result.single()["cnt"] > 0


if __name__ == "__main__":
    client = KnowledgeGraphClient()

    test_drug = "aspirin"
    print(f"Testing lookup for: {test_drug}")
    effects = client.get_side_effects_for_drug(test_drug)
    print(f"Found {len(effects)} side effects")
    print(effects[:10])

    client.close()
