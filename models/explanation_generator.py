"""Plain-language explanation of a verdict, built from the verdict, its factors and its evidence."""

VERDICT_SENTENCES = {
    "LIKELY_SUITABLE": "{drug} appears to be a reasonable option for {disease}.",
    "NOT_RECOMMENDED": "{drug} is likely NOT recommended for {disease}.",
    "MIXED_EVIDENCE": "The evidence on {drug} for {disease} is mixed.",
    "ASSOCIATION_ONLY": "{drug} is linked to {disease} in the literature, but the direction of the effect is unclear.",
    "NO_DIRECT_EVIDENCE": "No direct evidence was found linking {drug} to {disease}.",
    "UNKNOWN_DRUG": "{drug} was not found in the knowledge base.",
    "UNKNOWN_DISEASE": "{disease} was not found in the knowledge base.",
}


def generate_explanation(drug_name, disease_name, suitability, breakdown, evidence_sentences, side_effects=None):
    drug, disease = drug_name.strip().title(), disease_name.strip()
    lines = [VERDICT_SENTENCES.get(suitability["verdict"], suitability["reason"]).format(drug=drug, disease=disease), ""]

    lines.append("Why this conclusion:")
    for f in breakdown:
        lines.append(f"  • {f['factor']} — {f['detail']}")
    lines.append("")

    if side_effects:
        top = [e["side_effect"] for e in side_effects[:5]]
        lines.append(f"Most important known side effects: {', '.join(top)}.")
        lines.append("")

    if evidence_sentences:
        lines.append(f"Supporting evidence ({len(evidence_sentences)} source sentence(s)):")
        for e in evidence_sentences:
            origin = "annotated" if e.get("origin") == "gold" else "model-extracted"
            lines.append(f'  [{e.get("source_dataset")}, {origin}] "{e["sentence"]}"')
    else:
        lines.append("No source sentences were found for this specific drug-disease pair.")

    lines.append("")
    lines.append("This is an automated research output, not medical advice.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_explanation("aspirin", "headache", {"verdict": "NO_DIRECT_EVIDENCE", "reason": ""},
                               [{"factor": "No direct evidence", "detail": "None found."}], []))
