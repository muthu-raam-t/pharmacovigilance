import sys
sys.path.append("/workspace/models")

from kg_query import KnowledgeGraphClient
from ranking_engine import get_ranked_side_effects


def check_suitability(drug_name, disease_name):
    kg = KnowledgeGraphClient()

    drug_known = kg.drug_exists(drug_name)
    disease_known = kg.disease_exists(disease_name)
    causes_condition = kg.drug_causes_disease(drug_name, disease_name)
    associated = kg.drug_associated_with_disease(drug_name, disease_name)

    kg.close()

    side_effects = get_ranked_side_effects(drug_name)
    common_effects = [e for e in side_effects if e["severity_bucket"] in ("common", "occasional")]

    if not drug_known:
        return {
            "verdict": "UNKNOWN_DRUG",
            "reason": f"'{drug_name}' was not found in the knowledge base. Cannot assess suitability.",
            "side_effect_count": None,
        }

    if not disease_known:
        return {
            "verdict": "UNKNOWN_DISEASE",
            "reason": f"'{disease_name}' was not found in the knowledge base. Cannot assess disease-specific suitability, "
                      f"but general side-effect information for '{drug_name}' is available.",
            "side_effect_count": len(side_effects),
        }

    if causes_condition:
        verdict = "NOT_RECOMMENDED"
        reason = (f"Literature evidence indicates '{drug_name}' has been associated with causing or worsening "
                  f"'{disease_name}'. This drug may not be appropriate for this condition.")
    elif associated:
        verdict = "LIKELY_SUITABLE"
        reason = (f"'{drug_name}' appears in biomedical literature in connection with '{disease_name}', "
                  f"suggesting relevant clinical use. It has {len(side_effects)} known side effects "
                  f"({len(common_effects)} common/occasional).")
    else:
        verdict = "NO_DIRECT_EVIDENCE"
        reason = (f"No direct evidence was found connecting '{drug_name}' to '{disease_name}' in the current "
                  f"knowledge sources. This does not necessarily mean it is unsuitable — only that this system "
                  f"cannot confirm a relationship. Consult a medical professional.")

    return {
        "verdict": verdict,
        "reason": reason,
        "side_effect_count": len(side_effects),
        "common_side_effect_count": len(common_effects),
    }


if __name__ == "__main__":
    result = check_suitability("aspirin", "headache")
    print(result["verdict"])
    print(result["reason"])
