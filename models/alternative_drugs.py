import sys
sys.path.append("/workspace/models")

from common_indications import get_known_treating_drugs
from kg_query import KnowledgeGraphClient
from ranking_engine import get_ranked_side_effects
from kb_utils import is_known_drug_in_sider


def suggest_alternatives_for_disease(disease_name, exclude_drug=None, top_n=5):
    kg = KnowledgeGraphClient()

    if not kg.disease_exists(disease_name):
        kg.close()
        return {"error": f"'{disease_name}' not found in knowledge base — cannot suggest alternatives."}

    curated_candidates = set(get_known_treating_drugs(disease_name))
    if exclude_drug:
        curated_candidates.discard(exclude_drug.strip().lower())

    gold_candidates = set(kg.get_drugs_associated_with_disease(disease_name, exclude_drug=exclude_drug))
    model_candidates = set(kg.get_drugs_model_associated_with_disease(disease_name, exclude_drug=exclude_drug))
    known_to_cause_it = set(kg.get_drugs_that_cause_disease(disease_name))
    kg.close()

    all_candidates = (curated_candidates | gold_candidates | model_candidates) - known_to_cause_it

    # Only suggest candidates that are real, known drugs in SIDER — this
    # filters out non-drug entities (hormones, chemicals, etc.) that
    # merely co-occurred with the disease in adverse-event literature.
    known_drug_candidates = [c for c in all_candidates if is_known_drug_in_sider(c)]

    if not known_drug_candidates:
        return {"error": f"No verifiable drug alternatives found for '{disease_name}'."}

    scored = []
    for candidate in known_drug_candidates:
        effects = get_ranked_side_effects(candidate)
        if candidate in curated_candidates:
            source = "curated"
        elif candidate in gold_candidates:
            source = "gold"
        else:
            source = "model_prediction"
        scored.append({
            "drug_name": candidate,
            "side_effect_count": len(effects),
            "source": source,
        })

    scored.sort(key=lambda x: x["side_effect_count"])

    return {
        "disease": disease_name,
        "candidates_checked": len(known_drug_candidates),
        "alternatives": scored[:top_n],
    }


if __name__ == "__main__":
    result = suggest_alternatives_for_disease("hypotensive", exclude_drug="alpha-methyldopa", top_n=5)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Alternatives for treating: {result['disease']}")
        for alt in result["alternatives"]:
            print(f"  {alt['drug_name']:<30} side_effects={alt['side_effect_count']} source={alt['source']}")
