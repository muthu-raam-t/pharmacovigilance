"""Neo4j knowledge-graph queries.

Graph schema
  (:Drug)-[:CAUSES]->(:SideEffect)                    SIDER
  (:Drug)-[:CAUSES_CONDITION]->(:Disease)             gold annotation (BC5CDR, ADE, BioRED positive correlation)
  (:Drug)-[:TREATS_CONDITION]->(:Disease)             gold annotation (BioRED negative correlation)
  (:Drug)-[:ASSOCIATED_WITH]->(:Disease)              gold annotation (other BioRED associations)
  (:Drug)-[:MODEL_PREDICTED {relation, confidence, support, model}]->(:Disease)
"""
from db import neo4j_query
from text_utils import normalize_name

GOLD_EDGES = {"CAUSES": "CAUSES_CONDITION", "TREATS": "TREATS_CONDITION", "ASSOCIATED": "ASSOCIATED_WITH"}


class KnowledgeGraphClient:
    """Thin wrapper kept for backwards compatibility; uses the shared Neo4j driver."""

    def close(self):
        pass  # the shared driver stays open for the lifetime of the process

    # ---------- existence ----------
    def drug_exists(self, drug_name):
        rows = neo4j_query("MATCH (d:Drug {name: $n}) RETURN count(d) AS c", n=normalize_name(drug_name))
        return rows[0]["c"] > 0

    def disease_exists(self, disease_name):
        rows = neo4j_query("MATCH (d:Disease {name: $n}) RETURN count(d) AS c", n=normalize_name(disease_name))
        return rows[0]["c"] > 0

    # ---------- side effects ----------
    def get_side_effects_for_drug(self, drug_name):
        rows = neo4j_query("MATCH (d:Drug {name: $n})-[:CAUSES]->(s:SideEffect) RETURN s.name AS name",
                           n=normalize_name(drug_name))
        return [r["name"] for r in rows]

    # ---------- drug-disease evidence ----------
    def get_pair_evidence(self, drug_name, disease_name):
        """All gold and model evidence connecting one drug and one disease."""
        rows = neo4j_query("""
            MATCH (d:Drug {name: $drug})-[r]->(dis:Disease {name: $disease})
            RETURN type(r) AS edge, r.relation AS relation, r.confidence AS confidence,
                   r.model AS model, r.support AS support
        """, drug=normalize_name(drug_name), disease=normalize_name(disease_name))
        gold = {rel: False for rel in GOLD_EDGES}
        gold_support = {rel: 0 for rel in GOLD_EDGES}
        model = []
        for r in rows:
            for rel, edge in GOLD_EDGES.items():
                if r["edge"] == edge:
                    gold[rel] = True
                    gold_support[rel] = r.get("support") or 1
            if r["edge"] == "MODEL_PREDICTED":
                model.append({"relation": r["relation"], "confidence": float(r["confidence"] or 0.0),
                              "model": r["model"], "support": r.get("support") or 1})
        model.sort(key=lambda m: m["confidence"], reverse=True)
        return {"gold": gold, "gold_support": gold_support, "model": model}

    def get_disease_candidates(self, disease_name):
        """Drugs linked to a disease, grouped by evidence type (used for alternative suggestions)."""
        rows = neo4j_query("""
            MATCH (d:Drug)-[r]->(dis:Disease {name: $disease})
            WHERE type(r) IN ['CAUSES_CONDITION', 'TREATS_CONDITION', 'MODEL_PREDICTED']
            RETURN d.name AS drug, type(r) AS edge, r.relation AS relation, r.confidence AS confidence
        """, disease=normalize_name(disease_name))
        out = {"gold_treats": set(), "gold_causes": set(), "model_treats": {}, "model_causes": {}}
        for r in rows:
            if r["edge"] == "TREATS_CONDITION":
                out["gold_treats"].add(r["drug"])
            elif r["edge"] == "CAUSES_CONDITION":
                out["gold_causes"].add(r["drug"])
            elif r["relation"] == "TREATS":
                out["model_treats"][r["drug"]] = max(out["model_treats"].get(r["drug"], 0.0), float(r["confidence"]))
            elif r["relation"] == "CAUSES":
                out["model_causes"][r["drug"]] = max(out["model_causes"].get(r["drug"], 0.0), float(r["confidence"]))
        return out
