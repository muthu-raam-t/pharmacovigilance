import sys
sys.path.append("/workspace/models")

from kg_query import KnowledgeGraphClient
from ranking_engine import get_ranked_side_effects


def suggest_alternatives_for_disease(disease_name, exclude_drug=None, top_n=5):
    kg = KnowledgeGraphClient()

    if not kg.disease_exists(disease_name):
        kg.close()
        return {"error": f"'{disease_name}' not found in knowledge base — cannot suggest alternatives."}

    candidates = kg.get_drugs_associated_with_disease(disease_name, exclude_drug=exclude_drug)
    known_to_cause_it = set(kg.get_drugs_that_cause_disease(disease_name))
    kg.close()

    candidates = [c for c in candidates if c not in known_to_cause_it]

    if not candidates:
        return {"error": f"No alternative drugs found associated with '{disease_name}' that don't also risk causing it."}

    scored = []
    for candidate in candidates:
        effects = get_ranked_side_effects(candidate)
        scored.append({
            "drug_name": candidate,
            "side_effect_count": len(effects),
        })

    scored.sort(key=lambda x: x["side_effect_count"])

    return {
        "disease": disease_name,
        "candidates_checked": len(candidates),
        "alternatives": scored[:top_n],
    }


if __name__ == "__main__":
    result = suggest_alternatives_for_disease("headache", exclude_drug="aspirin", top_n=5)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Alternatives for treating: {result['disease']}")
        print(f"Candidates checked: {result['candidates_checked']}\n")
        for alt in result["alternatives"]:
            print(f"  {alt['drug_name']:<30} side_effects={alt['side_effect_count']}")
