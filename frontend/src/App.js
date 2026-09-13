import { useState } from "react";

const VERDICT_STYLES = {
  NOT_RECOMMENDED: { color: "#c0392b", bg: "#fdecea", icon: "⚠️", label: "Not Recommended" },
  LIKELY_SUITABLE: { color: "#1e8449", bg: "#eafaf1", icon: "✅", label: "Likely Suitable" },
  NO_DIRECT_EVIDENCE: { color: "#7f8c8d", bg: "#f4f4f4", icon: "❔", label: "No Direct Evidence" },
  UNKNOWN_DRUG: { color: "#7f8c8d", bg: "#f4f4f4", icon: "❓", label: "Unknown Drug" },
  UNKNOWN_DISEASE: { color: "#7f8c8d", bg: "#f4f4f4", icon: "❓", label: "Unknown Disease" },
};

const EXAMPLES = [
  { drug: "alpha-methyldopa", disease: "hypotensive" },
  { drug: "aspirin", disease: "headache" },
  { drug: "clonidine", disease: "hypertensive" },
];

function SourceBadge({ source }) {
  const isModel = source && source.startsWith("model_prediction");
  const confidenceMatch = isModel && source.match(/(\d+)%/);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
      <span
        style={{
          fontSize: "0.75rem", fontWeight: 600, padding: "0.25rem 0.7rem", borderRadius: "999px",
          backgroundColor: isModel ? "#e8f0fe" : "#fff4e0",
          color: isModel ? "#1a56db" : "#8a5a00",
          border: `1px solid ${isModel ? "#a9c6f7" : "#f0d090"}`,
          whiteSpace: "nowrap",
        }}
      >
        {isModel ? "🤖 SciBERT Prediction" : "📖 Human-Verified"}
      </span>
      {confidenceMatch && (
        <div style={{ width: "70px", height: "6px", backgroundColor: "#dbe7fb", borderRadius: "3px", overflow: "hidden" }}>
          <div style={{ width: `${confidenceMatch[1]}%`, height: "100%", backgroundColor: "#1a56db" }} />
        </div>
      )}
    </div>
  );
}

function Section({ title, icon, children }) {
  return (
    <div style={{ marginTop: "1.75rem" }}>
      <h3 style={{ fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#555", marginBottom: "0.6rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
        <span>{icon}</span> {title}
      </h3>
      {children}
    </div>
  );
}

function EmptyNote({ children }) {
  return (
    <div style={{ backgroundColor: "#fafafa", border: "1px dashed #ddd", borderRadius: "8px", padding: "0.9rem 1rem" }}>
      <p style={{ margin: 0, color: "#777", fontSize: "0.88rem", lineHeight: 1.5 }}>{children}</p>
    </div>
  );
}

function HowItWorks() {
  const [open, setOpen] = useState(false);
  const steps = [
    { icon: "📚", label: "5 transformer models compared (BERT, RoBERTa, SciBERT, PubMedBERT, BioBERT) for joint entity + relation extraction" },
    { icon: "🗄️", label: "Structured safety data from SIDER & OnSIDES (28.9M+ facts) loaded into PostgreSQL" },
    { icon: "🕸️", label: "Drug, Disease, and SideEffect entities linked in a Neo4j knowledge graph (158K+ relationships)" },
    { icon: "🧠", label: "SciBERT (top performer) extracts additional drug-disease relations beyond human-annotated data" },
    { icon: "⚖️", label: "Evidence fusion engine combines literature evidence + severity data into a suitability verdict" },
  ];
  return (
    <div style={{ marginTop: "1.5rem" }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: "none", border: "none", color: "#1a56db", fontSize: "0.85rem",
          fontWeight: 600, cursor: "pointer", padding: 0,
        }}
      >
        {open ? "▾" : "▸"} How does this system work?
      </button>
      {open && (
        <div style={{ marginTop: "0.8rem", display: "flex", flexDirection: "column", gap: "0.6rem" }}>
          {steps.map((s, i) => (
            <div key={i} style={{ display: "flex", gap: "0.6rem", alignItems: "flex-start" }}>
              <span>{s.icon}</span>
              <span style={{ fontSize: "0.85rem", color: "#555", lineHeight: 1.4 }}>{s.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function App() {
  const [drug, setDrug] = useState("");
  const [disease, setDisease] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runAnalysis = async (d, dis) => {
    if (!d.trim() || !dis.trim()) {
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
        body: JSON.stringify({ drug: d, disease: dis }),
      });
      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = () => runAnalysis(drug, disease);
  const handleExample = (ex) => {
    setDrug(ex.drug);
    setDisease(ex.disease);
    runAnalysis(ex.drug, ex.disease);
  };
  const handleKeyDown = (e) => { if (e.key === "Enter") handleAnalyze(); };

  const verdict = result ? (VERDICT_STYLES[result.suitability.verdict] || VERDICT_STYLES.NO_DIRECT_EVIDENCE) : null;

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#fafafa", fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      <div style={{ maxWidth: "800px", margin: "0 auto", padding: "3rem 1.5rem" }}>

        <header style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "#1a56db", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
            AI-POWERED DRUG SAFETY ANALYSIS
          </div>
          <h1 style={{ fontSize: "1.9rem", marginBottom: "0.3rem", color: "#1a1a1a" }}>
            Pharmacovigilance Advisory
          </h1>
          <p style={{ color: "#777", fontSize: "0.95rem", maxWidth: "550px", margin: "0 auto" }}>
            Checks drug-disease suitability, side effects, and safer alternatives using a
            biomedical knowledge graph and SciBERT-based relation extraction.
          </p>
        </header>

        <div style={{ backgroundColor: "#fff", borderRadius: "12px", boxShadow: "0 2px 10px rgba(0,0,0,0.06)", padding: "1.75rem" }}>
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
                style={{ width: "100%", padding: "0.65rem 0.8rem", borderRadius: "8px", border: "1px solid #ddd", fontSize: "0.95rem", boxSizing: "border-box" }}
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
                style={{ width: "100%", padding: "0.65rem 0.8rem", borderRadius: "8px", border: "1px solid #ddd", fontSize: "0.95rem", boxSizing: "border-box" }}
              />
            </div>
          </div>

          <div style={{ marginTop: "0.8rem", display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: "0.8rem", color: "#999" }}>Try:</span>
            {EXAMPLES.map((ex, i) => (
              <button
                key={i}
                onClick={() => handleExample(ex)}
                style={{
                  fontSize: "0.78rem", padding: "0.3rem 0.7rem", borderRadius: "999px",
                  border: "1px solid #ddd", backgroundColor: "#f7f7f7", color: "#555", cursor: "pointer",
                }}
              >
                {ex.drug} + {ex.disease}
              </button>
            ))}
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading}
            style={{
              marginTop: "1.25rem", width: "100%", padding: "0.75rem", borderRadius: "8px", border: "none",
              backgroundColor: loading ? "#a0a0a0" : "#1a56db", color: "#fff", fontSize: "1rem", fontWeight: 600,
              cursor: loading ? "default" : "pointer",
            }}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>

          {error && <p style={{ color: "#c0392b", marginTop: "1rem", fontSize: "0.9rem" }}>{error}</p>}

          <HowItWorks />
        </div>

        {result && (
          <div style={{ marginTop: "1.5rem", backgroundColor: "#fff", borderRadius: "12px", boxShadow: "0 2px 10px rgba(0,0,0,0.06)", padding: "1.75rem" }}>

            <div style={{ backgroundColor: verdict.bg, borderRadius: "10px", padding: "1.25rem", borderLeft: `5px solid ${verdict.color}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                <h2 style={{ margin: 0, color: verdict.color, fontSize: "1.3rem" }}>
                  {verdict.icon} {verdict.label}
                </h2>
                <SourceBadge source={result.suitability.source} />
              </div>
              <p style={{ margin: "0.6rem 0 0", color: "#333", fontSize: "0.95rem", lineHeight: 1.5 }}>
                {result.suitability.reason}
              </p>
              <p style={{ margin: "0.5rem 0 0", fontSize: "0.78rem", color: "#888" }}>
                Query: <strong>{result.drug}</strong> for <strong>{result.disease}</strong>
              </p>
            </div>

            <Section title="Side Effects" icon="💊">
              {result.suitability.has_side_effect_data === false ? (
                <EmptyNote>
                  No side-effect data is available for "<strong>{result.drug}</strong>" in the SIDER database used
                  by this project. SIDER covers roughly 1,400 drugs — this drug either uses a different naming
                  convention or falls outside that coverage.
                </EmptyNote>
              ) : (
                <>
                  <p style={{ color: "#555", marginBottom: "0.5rem" }}>{result.total_side_effects} known side effects</p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                    {result.side_effects.slice(0, 10).map((e, i) => (
                      <span key={i} style={{ fontSize: "0.85rem", padding: "0.3rem 0.7rem", borderRadius: "999px", backgroundColor: "#f1f1f1", color: "#444" }}>
                        {e.side_effect}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </Section>

            <Section title="Suggested Alternatives" icon="🔄">
              {result.alternatives ? (
                result.alternatives.error ? (
                  <EmptyNote>
                    No alternative was suggested because no candidate drug connected to "<strong>{result.disease}</strong>"
                    could be verified as a real, known drug in the SIDER database. This system only recommends
                    drugs it can independently confirm exist — rather than suggesting any chemical name that
                    happened to co-occur in literature.
                  </EmptyNote>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {result.alternatives.alternatives.map((a, i) => (
                      <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "0.6rem 0.9rem", backgroundColor: "#f8f8f8", borderRadius: "8px", fontSize: "0.9rem" }}>
                        <span style={{ fontWeight: 600 }}>{a.drug_name}</span>
                        <span style={{ color: "#777" }}>{a.side_effect_count} side effects · {a.source}</span>
                      </div>
                    ))}
                  </div>
                )
              ) : (
                <EmptyNote>Alternatives are only suggested when a drug is flagged as not recommended or unconfirmed.</EmptyNote>
              )}
            </Section>

            <Section title="Supporting Evidence" icon="🔍">
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
                <EmptyNote>No annotated source sentence directly connects this drug and disease pair.</EmptyNote>
              )}
            </Section>
          </div>
        )}

        <footer style={{ textAlign: "center", marginTop: "2.5rem", fontSize: "0.78rem", color: "#aaa" }}>
          Built on a 5-model transformer comparison (BERT, RoBERTa, SciBERT, PubMedBERT, BioBERT) ·
          Knowledge graph: 158K+ drug-side-effect relationships · Data: SIDER, OnSIDES, BC5CDR, BioRED, ADE Corpus
        </footer>

      </div>
    </div>
  );
}

export default App;
