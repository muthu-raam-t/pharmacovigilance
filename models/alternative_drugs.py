"""Alternative-drug suggestion for a disease.

Candidates (evidence tier, strongest first):
  1. curated clinical reference          2. gold TREATS annotation          3. model-predicted TREATS
Excluded: the queried drug, drugs with gold CAUSES evidence for the disease, and drugs whose
strongest model prediction for the disease is CAUSES. Only drugs found in SIDER or OnSIDES are kept.
Ranking: evidence tier, then number of boxed-warning effects, then side-effect burden (lower is better).
"""
from common_indications import get_known_treating_drugs
from kg_query import KnowledgeGraphClient
from kb_utils import is_known_drug
from ranking_engine import get_ranked_side_effects, side_effect_burden
from text_utils import normalize_name

TIERS = {"curated": 1, "gold": 2, "model_prediction": 3}


def suggest_alternatives_for_disease(disease_name, exclude_drug=None, top_n=5):
    disease = normalize_name(disease_name)
    exclude = normalize_name(exclude_drug) if exclude_drug else None
    kg = KnowledgeGraphClient()

    curated = set(get_known_treating_drugs(disease))
    if not curated and not kg.disease_exists(disease):
        return {"error": f"'{disease_name}' was not found in the knowledge base, so no alternatives can be suggested."}

    c = kg.get_disease_candidates(disease)
    candidates = {}
    for drug in curated:
        candidates[drug] = ("curated", None)
    for drug in c["gold_treats"]:
        candidates.setdefault(drug, ("gold", None))
    for drug, conf in c["model_treats"].items():
        if conf >= c["model_causes"].get(drug, 0.0):
            candidates.setdefault(drug, ("model_prediction", conf))

    for drug in list(candidates):
        if drug == exclude:
            del candidates[drug]
        elif drug in c["gold_causes"] and candidates[drug][0] != "curated":
            del candidates[drug]

    verified = {d: v for d, v in candidates.items() if is_known_drug(d)}
    if not verified:
        return {"error": f"No verifiable alternative drugs were found for '{disease_name}'.",
                "candidates_checked": len(candidates)}

    scored = []
    for drug, (source, conf) in verified.items():
        effects = get_ranked_side_effects(drug)
        scored.append({
            "drug_name": drug,
            "source": source,
            "model_confidence": conf,
            "side_effect_count": len(effects),
            "common_side_effect_count": sum(e["severity_bucket"] in ("common", "occasional") for e in effects),
            "boxed_warning_count": sum(bool(e["boxed_warning"]) for e in effects),
            "burden_score": side_effect_burden(effects),
            "top_side_effects": [e["side_effect"] for e in effects[:3]],
        })
    scored.sort(key=lambda x: (TIERS[x["source"]], x["boxed_warning_count"], x["burden_score"]))

    return {"disease": disease_name, "candidates_checked": len(candidates), "alternatives": scored[:top_n]}


if __name__ == "__main__":
    res = suggest_alternatives_for_disease("hypertension", exclude_drug="alpha-methyldopa")
    if "error" in res:
        print("Error:", res["error"])
    else:
        for a in res["alternatives"]:
            print(f"  {a['drug_name']:<25} {a['source']:<17} burden={a['burden_score']:<7} "
                  f"boxed={a['boxed_warning_count']} effects={a['side_effect_count']}")
