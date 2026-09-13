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

  return (
    <div className="App" style={{ maxWidth: "700px", margin: "0 auto", padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>Pharmacovigilance Advisory</h1>

      <div style={{ marginBottom: "1rem" }}>
        <label>Drug name: </label>
        <input value={drug} onChange={(e) => setDrug(e.target.value)} placeholder="e.g. aspirin" style={{ marginLeft: "0.5rem", padding: "0.4rem" }} />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <label>Disease name: </label>
        <input value={disease} onChange={(e) => setDisease(e.target.value)} placeholder="e.g. headache" style={{ marginLeft: "0.5rem", padding: "0.4rem" }} />
      </div>

      <button onClick={handleAnalyze} disabled={loading} style={{ padding: "0.5rem 1rem" }}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
        <div style={{ marginTop: "2rem", textAlign: "left" }}>
          <h2>Verdict: {result.suitability.verdict}</h2>
          <p>{result.suitability.reason}</p>
          <p><small>Source: {result.suitability.source}</small></p>

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

          {result.alternatives && result.alternatives.alternatives && (
            <>
              <h3>Suggested Alternatives</h3>
              <ul>
                {result.alternatives.alternatives.map((a, i) => (
                  <li key={i}>{a.drug_name} — {a.side_effect_count} known side effects ({a.source})</li>
                ))}

              </ul>
            </>
          )}

          {result.evidence_sentences.length > 0 && (
            <>
              <h3>Supporting Evidence</h3>
              {result.evidence_sentences.map((e, i) => (
                <p key={i} style={{ fontStyle: "italic", fontSize: "0.9rem" }}>
                  [{e.source_dataset}] "{e.sentence}"
                </p>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
