def generate_explanation(drug_name, disease_name, suitability, breakdown, evidence_sentences):
    lines = []

    verdict_phrases = {
        "NOT_RECOMMENDED": f"{drug_name.title()} is likely NOT recommended for {disease_name}.",
        "LIKELY_SUITABLE": f"{drug_name.title()} appears to be a reasonable option for {disease_name}.",
        "NO_DIRECT_EVIDENCE": f"No direct evidence was found linking {drug_name.title()} to {disease_name}.",
        "UNKNOWN_DRUG": f"{drug_name.title()} was not found in the knowledge base.",
        "UNKNOWN_DISEASE": f"{disease_name.title()} was not found in the knowledge base.",
    }
    lines.append(verdict_phrases.get(suitability["verdict"], suitability["reason"]))
    lines.append("")

    lines.append("Why this conclusion:")
    for f in breakdown:
        lines.append(f"  • {f['factor']} — {f['detail']}")
    lines.append("")

    if evidence_sentences:
        lines.append(f"Supporting evidence ({len(evidence_sentences)} source sentence(s)):")
        for e in evidence_sentences:
            lines.append(f'  [{e["source_dataset"]}] "{e["sentence"]}"')
    else:
        lines.append("No direct source sentences were found for this specific drug-disease pair.")

    return "\n".join(lines)


if __name__ == "__main__":
    fake_suitability = {"verdict": "NO_DIRECT_EVIDENCE", "reason": "test"}
    fake_breakdown = [{"factor": "No direct evidence found", "detail": "None found in corpora."}]
    text = generate_explanation("aspirin", "headache", fake_suitability, fake_breakdown, [])
    print(text)
