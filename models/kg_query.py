from neo4j import GraphDatabase

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_AUTH = ("neo4j", "drugsafety123")


class KnowledgeGraphClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    def get_model_prediction(self, drug_name, disease_name):
        drug_name = drug_name.strip().lower()
        disease_name = disease_name.strip().lower()
        query = """
        MATCH (d:Drug {name: $drug_name})-[r:MODEL_PREDICTED]->(dis:Disease {name: $disease_name})
        RETURN r.relation AS relation, r.confidence AS confidence, r.model AS model
        ORDER BY r.confidence DESC LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name, disease_name=disease_name)
            record = result.single()
            if record:
                return {"relation": record["relation"], "confidence": record["confidence"], "model": record["model"]}
            return None

    def get_drugs_model_associated_with_disease(self, disease_name, exclude_drug=None):
        disease_name = disease_name.strip().lower()
        query = """
        MATCH (d:Drug)-[r:MODEL_PREDICTED {relation: "ASSOCIATED"}]->(dis:Disease {name: $disease_name})
        RETURN DISTINCT d.name AS drug_name
        """
        with self.driver.session() as session:
            result = session.run(query, disease_name=disease_name)
            drugs = [record["drug_name"] for record in result]
        if exclude_drug:
            drugs = [d for d in drugs if d != exclude_drug.strip().lower()]
        return drugs


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

    def drug_causes_disease(self, drug_name, disease_name):
        drug_name = drug_name.strip().lower()
        disease_name = disease_name.strip().lower()
        query = """
        MATCH (d:Drug {name: $drug_name})-[:CAUSES_CONDITION]->(dis:Disease {name: $disease_name})
        RETURN count(*) AS cnt
        """
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name, disease_name=disease_name)
            return result.single()["cnt"] > 0

    def drug_associated_with_disease(self, drug_name, disease_name):
        drug_name = drug_name.strip().lower()
        disease_name = disease_name.strip().lower()
        query = """
        MATCH (d:Drug {name: $drug_name})-[:ASSOCIATED_WITH]->(dis:Disease {name: $disease_name})
        RETURN count(*) AS cnt
        """
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name, disease_name=disease_name)
            return result.single()["cnt"] > 0

    def get_drugs_associated_with_disease(self, disease_name, exclude_drug=None):
        disease_name = disease_name.strip().lower()
        query = """
        MATCH (d:Drug)-[:ASSOCIATED_WITH]->(dis:Disease {name: $disease_name})
        RETURN DISTINCT d.name AS drug_name
        """
        with self.driver.session() as session:
            result = session.run(query, disease_name=disease_name)
            drugs = [record["drug_name"] for record in result]

        if exclude_drug:
            drugs = [d for d in drugs if d != exclude_drug.strip().lower()]
        return drugs

    def get_drugs_that_cause_disease(self, disease_name):
        """Drugs known to cause/induce this exact condition - useful to exclude from suggestions."""
        disease_name = disease_name.strip().lower()
        query = """
        MATCH (d:Drug)-[:CAUSES_CONDITION]->(dis:Disease {name: $disease_name})
        RETURN DISTINCT d.name AS drug_name
        """
        with self.driver.session() as session:
            result = session.run(query, disease_name=disease_name)
            return [record["drug_name"] for record in result]

    def drug_exists(self, drug_name):
        drug_name = drug_name.strip().lower()
        query = "MATCH (d:Drug {name: $drug_name}) RETURN count(d) AS cnt"
        with self.driver.session() as session:
            result = session.run(query, drug_name=drug_name)
            return result.single()["cnt"] > 0

    def disease_exists(self, disease_name):
        disease_name = disease_name.strip().lower()
        query = "MATCH (dis:Disease {name: $disease_name}) RETURN count(dis) AS cnt"
        with self.driver.session() as session:
            result = session.run(query, disease_name=disease_name)
            return result.single()["cnt"] > 0
