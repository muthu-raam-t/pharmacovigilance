import sys
sys.path.append("/workspace/models")

from kg_query import KnowledgeGraphClient
from ranking_engine import get_ranked_side_effects
from kb_utils import is_known_drug_in_sider


def check_suitability(drug_name, disease_name):
    kg = KnowledgeGraphClient()

    drug_known = kg.drug_exists(drug_name)
    disease_known = kg.disease_exists(disease_name)
    causes_condition = kg.drug_causes_disease(drug_name, disease_name)
    associated = kg.drug_associated_with_disease(drug_name, disease_name)
    model_pred = kg.get_model_prediction(drug_name, disease_name)

    kg.close()

    side_effects = get_ranked_side_effects(drug_name)
    common_effects = [e for e in side_effects if e["severity_bucket"] in ("common", "occasional")]
    has_side_effect_data = is_known_drug_in_sider(drug_name)

    if not drug_known and not model_pred and not has_side_effect_data:
        return {
            "verdict": "UNKNOWN_DRUG",
            "reason": f"'{drug_name}' was not found in the knowledge base. Cannot assess suitability.",
            "side_effect_count": None,
            "has_side_effect_data": False,
            "source": "none",
        }

    if not disease_known and not model_pred:
        return {
            "verdict": "UNKNOWN_DISEASE",
            "reason": f"'{disease_name}' was not found in the knowledge base.",
            "side_effect_count": len(side_effects),
            "has_side_effect_data": has_side_effect_data,
            "source": "none",
        }

    if causes_condition:
        verdict, source = "NOT_RECOMMENDED", "gold_annotation"
        reason = (f"Literature evidence (gold-annotated) indicates '{drug_name}' has been associated with causing "
                  f"or worsening '{disease_name}'. This drug may not be appropriate for this condition.")
    elif model_pred and model_pred["relation"] == "CAUSES":
        verdict = "NOT_RECOMMENDED"
        source = f"model_prediction (SciBERT, {model_pred['confidence']*100:.0f}% confidence)"
        reason = (f"The SciBERT model predicts (confidence {model_pred['confidence']*100:.0f}%) that '{drug_name}' "
                  f"may cause or worsen '{disease_name}'. This drug may not be appropriate for this condition.")
    elif associated:
        verdict, source = "LIKELY_SUITABLE", "gold_annotation"
        reason = (f"'{drug_name}' appears in gold-annotated literature connected to '{disease_name}'. "
                  + (f"It has {len(side_effects)} known side effects ({len(common_effects)} common/occasional)."
                     if has_side_effect_data else
                     "No side-effect data is available for this drug in the current database."))
    elif model_pred and model_pred["relation"] == "ASSOCIATED":
        verdict = "LIKELY_SUITABLE"
        source = f"model_prediction (SciBERT, {model_pred['confidence']*100:.0f}% confidence)"
        reason = (f"The SciBERT model predicts (confidence {model_pred['confidence']*100:.0f}%) a clinical "
                  f"association between '{drug_name}' and '{disease_name}'.")
    else:
        verdict, source = "NO_DIRECT_EVIDENCE", "none"
        reason = (f"No direct evidence was found connecting '{drug_name}' to '{disease_name}' in gold annotations "
                  f"or model predictions. This does not necessarily mean it is unsuitable — consult a medical professional.")

    return {
        "verdict": verdict,
        "reason": reason,
        "side_effect_count": len(side_effects),
        "common_side_effect_count": len(common_effects),
        "has_side_effect_data": has_side_effect_data,
        "source": source,
    }


if __name__ == "__main__":
    result = check_suitability("alpha-methyldopa", "hypotensive")
    print(result["verdict"])
    print(result["reason"])
    print("Source:", result["source"])
