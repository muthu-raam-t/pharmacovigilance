import { useState } from "react";
import "./App.css";

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
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const verdictColor = {
    NOT_RECOMMENDED: "#c0392b",
    LIKELY_SUITABLE: "#27ae60",
    NO_DIRECT_EVIDENCE: "#7f8c8d",
    UNKNOWN_DRUG: "#7f8c8d",
    UNKNOWN_DISEASE: "#7f8c8d",
  };

  return (
    <div className="App" style={{ maxWidth: "750px", margin: "0 auto", padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Pharmacovigilance Advisory</h1>

      <div style={{ marginBottom: "1rem" }}>
        <label>Drug name: </label>
        <input
          value={drug}
          onChange={(e) => setDrug(e.target.value)}
          placeholder="e.g. aspirin"
          style={{ marginLeft: "0.5rem", padding: "0.4rem", width: "250px" }}
        />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <label>Disease name: </label>
        <input
          value={disease}
          onChange={(e) => setDisease(e.target.value)}
          placeholder="e.g. headache"
          style={{ marginLeft: "0.5rem", padding: "0.4rem", width: "250px" }}
        />
      </div>

      <button onClick={handleAnalyze} disabled={loading} style={{ padding: "0.5rem 1.2rem", cursor: "pointer" }}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p style={{ color: "red", marginTop: "1rem" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: "2rem", textAlign: "left" }}>

          <div
            style={{
              padding: "1rem",
              borderRadius: "8px",
              backgroundColor: "#f5f5f5",
              borderLeft: `6px solid ${verdictColor[result.suitability.verdict] || "#7f8c8d"}`,
              marginBottom: "1.5rem",
            }}
          >
            <h2 style={{ margin: 0, color: verdictColor[result.suitability.verdict] || "#333" }}>
              {result.suitability.verdict.replace(/_/g, " ")}
            </h2>
            <p style={{ marginBottom: "0.3rem" }}>{result.suitability.reason}</p>
            <p style={{ fontSize: "0.85rem", color: "#666", margin: 0 }}>
              Source: {result.suitability.source}
            </p>
          </div>

          <h3>Side Effects</h3>
          {result.suitability.has_side_effect_data === false ? (
            <p><em>No side-effect data available for this drug in the current database.</em></p>
          ) : (
            <>
              <p>{result.total_side_effects} known side effects</p>
              <ul>
                {result.side_effects.slice(0, 10).map((e, i) => (
                  <li key={i}>{e.side_effect} ({e.severity_bucket})</li>
                ))}
              </ul>
            </>
          )}

          <h3>Suggested Alternatives</h3>
          {result.alternatives ? (
            result.alternatives.error ? (
              <p><em>{result.alternatives.error}</em></p>
            ) : (
              <ul>
                {result.alternatives.alternatives.map((a, i) => (
                  <li key={i}>
                    {a.drug_name} — {a.side_effect_count} known side effects ({a.source})
                  </li>
                ))}
              </ul>
            )
          ) : (
            <p><em>No alternatives were requested for this verdict.</em></p>
          )}

          <h3>Supporting Evidence</h3>
          {result.evidence_sentences.length > 0 ? (
            result.evidence_sentences.map((e, i) => (
              <p key={i} style={{ fontStyle: "italic", fontSize: "0.9rem", color: "#444" }}>
                [{e.source_dataset}, relation: {e.relation_type}] "{e.sentence}"
              </p>
            ))
          ) : (
            <p><em>No direct source sentences were found for this specific drug-disease pair.</em></p>
          )}

        </div>
      )}
    </div>
  );
}

export default App;
