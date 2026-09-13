import { useState } from "react";

const VERDICT_STYLES = {
  NOT_RECOMMENDED: { color: "#c0392b", bg: "#fdecea", label: "Not Recommended" },
  LIKELY_SUITABLE: { color: "#1e8449", bg: "#eafaf1", label: "Likely Suitable" },
  NO_DIRECT_EVIDENCE: { color: "#7f8c8d", bg: "#f4f4f4", label: "No Direct Evidence" },
  UNKNOWN_DRUG: { color: "#7f8c8d", bg: "#f4f4f4", label: "Unknown Drug" },
  UNKNOWN_DISEASE: { color: "#7f8c8d", bg: "#f4f4f4", label: "Unknown Disease" },
};

function SourceBadge({ source }) {
  const isModel = source && source.startsWith("model_prediction");
  return (
    <span
      style={{
        display: "inline-block",
        fontSize: "0.75rem",
        fontWeight: 600,
        padding: "0.2rem 0.6rem",
        borderRadius: "999px",
        backgroundColor: isModel ? "#e8f0fe" : "#fff4e0",
        color: isModel ? "#1a56db" : "#8a5a00",
        border: `1px solid ${isModel ? "#a9c6f7" : "#f0d090"}`,
      }}
    >
      {isModel ? source.replace("model_prediction", "SciBERT Prediction") : "Human-Verified (Gold Annotation)"}
    </span>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginTop: "1.75rem" }}>
      <h3 style={{ fontSize: "1rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#555", marginBottom: "0.6rem" }}>
        {title}
      </h3>
      {children}
    </div>
  );
}

function App() {
  const [drug, setDrug] = useState("");
  const [disease, setDisease] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!drug.trim() || !disease.trim()) {
      setError("Please enter both a drug and a disease.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drug, disease }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleAnalyze();
  };

  const verdict = result ? (VERDICT_STYLES[result.suitability.verdict] || VERDICT_STYLES.NO_DIRECT_EVIDENCE) : null;

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#fafafa", fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      <div style={{ maxWidth: "800px", margin: "0 auto", padding: "3rem 1.5rem" }}>

        <header style={{ textAlign: "center", marginBottom: "2.5rem" }}>
          <h1 style={{ fontSize: "1.9rem", marginBottom: "0.3rem", color: "#1a1a1a" }}>
            Pharmacovigilance Advisory
          </h1>
          <p style={{ color: "#777", fontSize: "0.95rem" }}>
            Check drug-disease suitability, side effects, and alternatives — backed by literature and model evidence.
          </p>
        </header>

        <div
          style={{
            backgroundColor: "#fff",
            borderRadius: "12px",
            boxShadow: "0 2px 10px rgba(0,0,0,0.06)",
            padding: "1.75rem",
          }}
        >
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: "200px" }}>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.4rem", color: "#444" }}>
                Drug Name
              </label>
              <input
                value={drug}
                onChange={(e) => setDrug(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. aspirin"
                style={{
                  width: "100%", padding: "0.65rem 0.8rem", borderRadius: "8px",
                  border: "1px solid #ddd", fontSize: "0.95rem", boxSizing: "border-box",
                }}
              />
            </div>
            <div style={{ flex: 1, minWidth: "200px" }}>
              <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.4rem", color: "#444" }}>
                Disease Name
              </label>
              <input
                value={disease}
                onChange={(e) => setDisease(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. hypotensive"
                style={{
                  width: "100%", padding: "0.65rem 0.8rem", borderRadius: "8px",
                  border: "1px solid #ddd", fontSize: "0.95rem", boxSizing: "border-box",
                }}
              />
            </div>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              marginTop: "1.25rem", width: "100%", padding: "0.75rem", borderRadius: "8px",
              border: "none", backgroundColor: loading ? "#a0a0a0" : "#1a56db", color: "#fff",
              fontSize: "1rem", fontWeight: 600, cursor: loading ? "default" : "pointer",
              transition: "background-color 0.2s",
            }}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>

          {error && <p style={{ color: "#c0392b", marginTop: "1rem", fontSize: "0.9rem" }}>{error}</p>}
        </div>

        {result && (
          <div
            style={{
              marginTop: "1.5rem", backgroundColor: "#fff", borderRadius: "12px",
              boxShadow: "0 2px 10px rgba(0,0,0,0.06)", padding: "1.75rem",
            }}
          >
            <div
              style={{
                backgroundColor: verdict.bg, borderRadius: "10px", padding: "1.25rem",
                borderLeft: `5px solid ${verdict.color}`,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                <h2 style={{ margin: 0, color: verdict.color, fontSize: "1.3rem" }}>{verdict.label}</h2>
                <SourceBadge source={result.suitability.source} />
              </div>
              <p style={{ margin: "0.6rem 0 0", color: "#333", fontSize: "0.95rem", lineHeight: 1.5 }}>
                {result.suitability.reason}
              </p>
            </div>

            <Section title="Side Effects">
              {result.suitability.has_side_effect_data === false ? (
                <p style={{ color: "#888", fontStyle: "italic" }}>No side-effect data available for this drug.</p>
              ) : (
                <>
                  <p style={{ color: "#555", marginBottom: "0.5rem" }}>{result.total_side_effects} known side effects</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                    {result.side_effects.slice(0, 10).map((e, i) => (
                      <span
                        key={i}
                        style={{
                          fontSize: "0.85rem", padding: "0.3rem 0.7rem", borderRadius: "999px",
                          backgroundColor: "#f1f1f1", color: "#444",
                        }}
                      >
                        {e.side_effect}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </Section>

            <Section title="Suggested Alternatives">
              {result.alternatives ? (
                result.alternatives.error ? (
                  <p style={{ color: "#888", fontStyle: "italic" }}>{result.alternatives.error}</p>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {result.alternatives.alternatives.map((a, i) => (
                      <div
                        key={i}
                        style={{
                          display: "flex", justifyContent: "space-between", padding: "0.6rem 0.9rem",
                          backgroundColor: "#f8f8f8", borderRadius: "8px", fontSize: "0.9rem",
                        }}
                      >
                        <span style={{ fontWeight: 600 }}>{a.drug_name}</span>
                        <span style={{ color: "#777" }}>{a.side_effect_count} side effects · {a.source}</span>
                      </div>
                    ))}
                  </div>
                )
              ) : (
                <p style={{ color: "#888", fontStyle: "italic" }}>No alternatives were requested for this verdict.</p>
              )}
            </Section>

            <Section title="Supporting Evidence">
              {result.evidence_sentences.length > 0 ? (
                result.evidence_sentences.map((e, i) => (
                  <div key={i} style={{ marginBottom: "0.75rem", padding: "0.8rem", backgroundColor: "#f8f8f8", borderRadius: "8px" }}>
                    <div style={{ fontSize: "0.75rem", color: "#888", marginBottom: "0.3rem" }}>
                      {e.source_dataset} · relation: {e.relation_type}
                    </div>
                    <p style={{ fontStyle: "italic", fontSize: "0.88rem", color: "#444", margin: 0, lineHeight: 1.5 }}>
                      "{e.sentence}"
                    </p>
                  </div>
                ))
              ) : (
                <p style={{ color: "#888", fontStyle: "italic" }}>No direct source sentences were found for this pair.</p>
              )}
            </Section>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
