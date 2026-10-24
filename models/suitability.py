"""Evidence-fusion verdict for one drug-disease pair.

Priority of evidence (strongest first):
  1. Curated clinical reference (well-established indication)         -> LIKELY_SUITABLE
  2. Unknown drug / disease                                            -> UNKNOWN_DRUG / UNKNOWN_DISEASE
  3. Gold (human-annotated) literature evidence
       TREATS and CAUSES both present                                  -> MIXED_EVIDENCE
       TREATS                                                          -> LIKELY_SUITABLE
       CAUSES                                                          -> NOT_RECOMMENDED
  4. Model-predicted evidence (highest-confidence TREATS vs CAUSES)    -> LIKELY_SUITABLE / NOT_RECOMMENDED
  5. Only a non-directional association (gold or model)                -> ASSOCIATION_ONLY
  6. Nothing                                                           -> NO_DIRECT_EVIDENCE
Human annotations always outrank model predictions.
"""
from common_indications import is_known_indication, get_known_treating_drugs
from kg_query import KnowledgeGraphClient
from kb_utils import is_known_drug, side_effect_sources_for
from ranking_engine import get_ranked_side_effects


def _side_effect_summary(side_effects):
    return {
        "side_effect_count": len(side_effects),
        "common_side_effect_count": sum(e["severity_bucket"] in ("common", "occasional") for e in side_effects),
        "boxed_warning_count": sum(bool(e["boxed_warning"]) for e in side_effects),
    }


def check_suitability(drug_name, disease_name, side_effects=None):
    if side_effects is None:
        side_effects = get_ranked_side_effects(drug_name)
    kg = KnowledgeGraphClient()
    evidence = kg.get_pair_evidence(drug_name, disease_name)
    gold, model = evidence["gold"], evidence["model"]
    se_sources = side_effect_sources_for(drug_name)

    base = {
        **_side_effect_summary(side_effects),
        "has_side_effect_data": bool(se_sources),
        "side_effect_sources": se_sources,
        "gold_evidence": gold,
        "gold_support": evidence["gold_support"],
        "model_predictions": model,
        "evidence_type": "none",
        "model_name": None,
        "model_confidence": None,
    }

    def result(verdict, reason, source, evidence_type, model_pred=None):
        out = dict(base, verdict=verdict, reason=reason, source=source, evidence_type=evidence_type)
        if model_pred:
            out["model_name"] = model_pred["model"]
            out["model_confidence"] = model_pred["confidence"]
        return out

    d, dis = drug_name.strip(), disease_name.strip()

    # 1. curated clinical reference
    if is_known_indication(d, dis):
        return result("LIKELY_SUITABLE",
                      f"'{d.title()}' is a well-established treatment for '{dis}' according to a standard "
                      f"clinical reference. Side effects listed below are known adverse effects of the drug, "
                      f"not a contraindication.",
                      "curated_clinical_reference", "curated")

    # 2. unknown inputs
    drug_known = is_known_drug(d) or kg.drug_exists(d) or bool(model)
    if not drug_known:
        return result("UNKNOWN_DRUG", f"'{d}' was not found in the knowledge base, so suitability cannot be assessed.",
                      "none", "none")
    if not (kg.disease_exists(dis) or get_known_treating_drugs(dis) or model):
        return result("UNKNOWN_DISEASE", f"'{dis}' was not found in the knowledge base.", "none", "none")

    # 3. gold annotations
    if gold["TREATS"] and gold["CAUSES"]:
        return result("MIXED_EVIDENCE",
                      f"Human-annotated literature reports '{d}' both as a treatment for '{dis}' and as a cause "
                      f"or aggravating factor of it. The balance depends on dose, patient group and context - "
                      f"consult a medical professional.",
                      "gold_annotation", "gold")
    if gold["TREATS"]:
        return result("LIKELY_SUITABLE",
                      f"Human-annotated literature reports that '{d}' treats or reduces '{dis}'.",
                      "gold_annotation", "gold")
    if gold["CAUSES"]:
        return result("NOT_RECOMMENDED",
                      f"Human-annotated literature reports '{d}' as a cause or aggravating factor of '{dis}'. "
                      f"This drug may not be appropriate for this condition.",
                      "gold_annotation", "gold")

    # 4. model predictions (directional)
    directional = [m for m in model if m["relation"] in ("TREATS", "CAUSES")]
    if directional:
        best = directional[0]
        pct = f"{best['confidence'] * 100:.0f}%"
        src = f"model_prediction ({best['model']}, {pct} confidence)"
        if best["relation"] == "TREATS":
            return result("LIKELY_SUITABLE",
                          f"The {best['model']} relation-extraction model predicts (confidence {pct}) that '{d}' "
                          f"treats or reduces '{dis}'. This is an automated prediction, not a human-verified fact.",
                          src, "model", best)
        return result("NOT_RECOMMENDED",
                      f"The {best['model']} relation-extraction model predicts (confidence {pct}) that '{d}' may "
                      f"cause or worsen '{dis}'. This is an automated prediction, not a human-verified fact.",
                      src, "model", best)

    # 5. association only
    assoc_model = [m for m in model if m["relation"] == "ASSOCIATED"]
    if gold["ASSOCIATED"] or assoc_model:
        who = "Human-annotated literature" if gold["ASSOCIATED"] else f"The {assoc_model[0]['model']} model"
        return result("ASSOCIATION_ONLY",
                      f"{who} links '{d}' and '{dis}', but does not say whether the drug treats or causes it.",
                      "gold_annotation" if gold["ASSOCIATED"] else
                      f"model_prediction ({assoc_model[0]['model']}, {assoc_model[0]['confidence'] * 100:.0f}% confidence)",
                      "gold" if gold["ASSOCIATED"] else "model",
                      None if gold["ASSOCIATED"] else assoc_model[0])

    # 6. nothing
    return result("NO_DIRECT_EVIDENCE",
                  f"No evidence connecting '{d}' and '{dis}' was found in the annotated literature or the model "
                  f"predictions. This does not mean the drug is unsuitable - consult a medical professional.",
                  "none", "none")


if __name__ == "__main__":
    r = check_suitability("alpha-methyldopa", "hypotensive")
    print(r["verdict"], "|", r["source"])
    print(r["reason"])
