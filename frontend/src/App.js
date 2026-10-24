import { useState } from "react";

const API_URL = (process.env.REACT_APP_API_URL || "http://localhost:8000").replace(/\/$/, "");

const VERDICT_STYLES = {
  LIKELY_SUITABLE: { color: "#1e8449", bg: "#eafaf1", icon: "✅", label: "Likely Suitable" },
  NOT_RECOMMENDED: { color: "#c0392b", bg: "#fdecea", icon: "⚠️", label: "Not Recommended" },
  MIXED_EVIDENCE: { color: "#b9770e", bg: "#fef5e7", icon: "⚖️", label: "Mixed Evidence" },
  ASSOCIATION_ONLY: { color: "#2e5c8a", bg: "#eaf1f8", icon: "🔗", label: "Association Only" },
  NO_DIRECT_EVIDENCE: { color: "#7f8c8d", bg: "#f4f4f4", icon: "❔", label: "No Direct Evidence" },
  UNKNOWN_DRUG: { color: "#7f8c8d", bg: "#f4f4f4", icon: "❓", label: "Unknown Drug" },
  UNKNOWN_DISEASE: { color: "#7f8c8d", bg: "#f4f4f4", icon: "❓", label: "Unknown Disease" },
};

const IMPACT_STYLES = {
  positive: { color: "#1e8449", bg: "#eafaf1", symbol: "+" },
  negative: { color: "#c0392b", bg: "#fdecea", symbol: "−" },
  neutral: { color: "#7f8c8d", bg: "#f4f4f4", symbol: "•" },
  informational: { color: "#1a56db", bg: "#e8f0fe", symbol: "i" },
};

const SEVERITY_STYLES = {
  common: { color: "#c0392b", bg: "#fdecea", label: "Common ≥10%" },
  occasional: { color: "#b9770e", bg: "#fef5e7", label: "Occasional 1–10%" },
  rare: { color: "#1e8449", bg: "#eafaf1", label: "Rare <1%" },
  unknown: { color: "#7f8c8d", bg: "#f1f1f1", label: "Frequency unknown" },
};

const ALT_SOURCE_LABELS = {
  curated: "Clinical reference",
  gold: "Human-annotated",
  model_prediction: "Model prediction",
};

const EXAMPLES = [
  { drug: "metformin", disease: "type 2 diabetes" },
  { drug: "albuterol", disease: "asthma" },
  { drug: "alpha-methyldopa", disease: "hypotensive" },
];

const card = { backgroundColor: "#fff", borderRadius: "12px", boxShadow: "0 2px 10px rgba(0,0,0,0.06)", padding: "1.75rem" };
const pill = (bg, color, border) => ({
  fontSize: "0.72rem", fontWeight: 600, padding: "0.2rem 0.6rem", borderRadius: "999px",
  backgroundColor: bg, color, border: border ? `1px solid ${border}` : "none", whiteSpace: "nowrap",
});

function SourceBadge({ suitability }) {
  const { evidence_type: type, model_name: model, model_confidence: conf } = suitability;
  const style = {
    curated: { bg: "#eafaf1", color: "#1e8449", border: "#a3d9b1", label: "📋 Clinical Reference" },
    gold: { bg: "#fff4e0", color: "#8a5a00", border: "#f0d090", label: "📖 Human-Annotated" },
    model: { bg: "#e8f0fe", color: "#1a56db", border: "#a9c6f7", label: `🤖 ${model || "Model"} Prediction` },
  }[type];
  if (!style) return null;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
      <span style={pill(style.bg, style.color, style.border)}>{style.label}</span>
      {type === "model" && conf != null && (
        <div title={`${Math.round(conf * 100)}% confidence`} style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
          <div style={{ width: "70px", height: "6px", backgroundColor: "#dbe7fb", borderRadius: "3px", overflow: "hidden" }}>
            <div style={{ width: `${Math.round(conf * 100)}%`, height: "100%", backgroundColor: "#1a56db" }} />
          </div>
          <span style={{ fontSize: "0.72rem", color: "#1a56db" }}>{Math.round(conf * 100)}%</span>
        </div>
      )}
    </div>
  );
}

function Section({ title, icon, children, aside }) {
  return (
    <div style={{ marginTop: "1.75rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "0.5rem", flexWrap: "wrap" }}>
        <h3 style={{ fontSize: "0.95rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "#555", margin: "0 0 0.6rem", display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <span>{icon}</span> {title}
        </h3>
        {aside}
      </div>
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

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function Highlighted({ text, terms }) {
  const clean = terms.filter(Boolean).map((t) => t.trim()).filter((t) => t.length > 1);
  if (!clean.length) return text;
  const re = new RegExp(`(${clean.map(escapeRegExp).join("|")})`, "gi");
  const lower = clean.map((t) => t.toLowerCase());
  return text.split(re).map((part, i) =>
    lower.includes(part.toLowerCase())
      ? <mark key={i} style={{ backgroundColor: "#fff3bf", padding: "0 2px", borderRadius: "3px", fontStyle: "normal", fontWeight: 600 }}>{part}</mark>
      : <span key={i}>{part}</span>
  );
}

function ScoreBreakdown({ factors }) {
  if (!factors || !factors.length) return <EmptyNote>No factors were computed for this query.</EmptyNote>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      {factors.map((f, i) => {
        const s = IMPACT_STYLES[f.impact] || IMPACT_STYLES.neutral;
        return (
          <div key={i} style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start", padding: "0.7rem 0.9rem", backgroundColor: "#f8f8f8", borderRadius: "8px" }}>
            <span style={{ minWidth: "1.6rem", height: "1.6rem", borderRadius: "50%", backgroundColor: s.bg, color: s.color, fontWeight: 700, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.9rem" }}>
              {s.symbol}
            </span>
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", flexWrap: "wrap" }}>
                <span style={{ fontWeight: 600, fontSize: "0.9rem", color: "#333" }}>{f.factor}</span>
                <span style={{ fontSize: "0.75rem", color: "#888" }}>{f.impact} · weight {Number(f.weight).toFixed(2)}</span>
              </div>
              <p style={{ margin: "0.2rem 0 0", fontSize: "0.84rem", color: "#666", lineHeight: 1.45 }}>{f.detail}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function SideEffects({ result }) {
  const [showAll, setShowAll] = useState(false);
  if (result.suitability.has_side_effect_data === false) {
    return (
      <EmptyNote>
        No side-effect data is available for "<strong>{result.drug}</strong>" in SIDER or OnSIDES. The drug may use a
        different name in these databases or fall outside their coverage.
      </EmptyNote>
    );
  }
  const shown = showAll ? result.side_effects : result.side_effects.slice(0, 8);
  return (
    <>
      <p style={{ color: "#555", margin: "0 0 0.6rem", fontSize: "0.9rem" }}>
        {result.total_side_effects} known side effects · sources: {(result.suitability.side_effect_sources || []).join(" + ")}
        {result.suitability.boxed_warning_count > 0 && (
          <span style={{ ...pill("#fdecea", "#c0392b", "#f5b7b1"), marginLeft: "0.5rem" }}>
            {result.suitability.boxed_warning_count} in FDA boxed warning
          </span>
        )}
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
        {shown.map((e, i) => {
          const s = SEVERITY_STYLES[e.severity_bucket] || SEVERITY_STYLES.unknown;
          return (
            <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.5rem", flexWrap: "wrap", padding: "0.45rem 0.8rem", backgroundColor: "#f8f8f8", borderRadius: "8px", fontSize: "0.88rem" }}>
              <span style={{ fontWeight: 600, color: "#333", textTransform: "capitalize" }}>{e.side_effect}</span>
              <span style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap" }}>
                {e.boxed_warning && <span style={pill("#c0392b", "#fff")}>Boxed warning</span>}
                <span style={pill(s.bg, s.color)}>{e.frequency != null ? `${(e.frequency * 100).toFixed(1)}% · ` : ""}{s.label}</span>
                <span style={pill("#eef1f5", "#555")}>{(e.sources || []).join(" + ")}</span>
              </span>
            </div>
          );
        })}
      </div>
      {result.side_effects.length > 8 && (
        <button onClick={() => setShowAll(!showAll)} style={{ marginTop: "0.5rem", background: "none", border: "none", color: "#1a56db", fontSize: "0.82rem", fontWeight: 600, cursor: "pointer", padding: 0 }}>
          {showAll ? "Show fewer" : `Show top ${result.side_effects.length}`}
        </button>
      )}
    </>
  );
}

function Alternatives({ result }) {
  if (!result.alternatives) {
    return <EmptyNote>Alternatives are suggested only when the drug is not recommended or the evidence is inconclusive.</EmptyNote>;
  }
  if (result.alternatives.error) {
    return (
      <EmptyNote>
        {result.alternatives.error} The system only suggests drugs that have treatment evidence for "
        <strong>{result.disease}</strong>" and can be verified in SIDER or OnSIDES.
      </EmptyNote>
    );
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      {result.alternatives.alternatives.map((a, i) => (
        <div key={i} style={{ padding: "0.7rem 0.9rem", backgroundColor: "#f8f8f8", borderRadius: "8px", fontSize: "0.88rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", flexWrap: "wrap" }}>
            <span style={{ fontWeight: 700, textTransform: "capitalize" }}>{i + 1}. {a.drug_name}</span>
            <span style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap" }}>
              <span style={pill("#e8f0fe", "#1a56db")}>
                {ALT_SOURCE_LABELS[a.source] || a.source}{a.model_confidence != null ? ` · ${Math.round(a.model_confidence * 100)}%` : ""}
              </span>
              {a.boxed_warning_count > 0 && <span style={pill("#fdecea", "#c0392b")}>{a.boxed_warning_count} boxed warning{a.boxed_warning_count > 1 ? "s" : ""}</span>}
              <span style={pill("#eef1f5", "#555")}>burden {a.burden_score}</span>
            </span>
          </div>
          <div style={{ color: "#777", marginTop: "0.25rem", fontSize: "0.82rem" }}>
            {a.side_effect_count} known side effects ({a.common_side_effect_count} common/occasional)
            {a.top_side_effects && a.top_side_effects.length > 0 && <> · e.g. {a.top_side_effects.join(", ")}</>}
          </div>
        </div>
      ))}
    </div>
  );
}

function Evidence({ result }) {
  if (!result.evidence_sentences.length) {
    return <EmptyNote>No annotated or model-extracted source sentence connects this drug and disease.</EmptyNote>;
  }
  return result.evidence_sentences.map((e, i) => (
    <div key={i} style={{ marginBottom: "0.75rem", padding: "0.8rem", backgroundColor: "#f8f8f8", borderRadius: "8px" }}>
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", fontSize: "0.75rem", color: "#888", marginBottom: "0.35rem" }}>
        <span style={e.origin === "gold" ? pill("#fff4e0", "#8a5a00") : pill("#e8f0fe", "#1a56db")}>
          {e.origin === "gold" ? "Human-annotated" : `Model-extracted${e.confidence != null ? ` · ${Math.round(e.confidence * 100)}%` : ""}`}
        </span>
        <span>{e.source_dataset} · relation: {e.relation_type}</span>
      </div>
      <p style={{ fontStyle: "italic", fontSize: "0.88rem", color: "#444", margin: 0, lineHeight: 1.55 }}>
        "<Highlighted text={e.sentence} terms={[e.drug_mention, e.disease_mention]} />"
      </p>
    </div>
  ));
}

function HowItWorks() {
  const [open, setOpen] = useState(false);
  const steps = [
    { icon: "📚", label: "Five transformer backbones (BERT, RoBERTa, SciBERT, BioBERT, PubMedBERT) compared for joint entity + relation extraction on BC5CDR, BioRED and ADE" },
    { icon: "🧠", label: "The best model extracts TREATS / CAUSES relations between drugs and diseases from the literature" },
    { icon: "🗄️", label: "SIDER (frequencies) and OnSIDES (FDA labels, boxed warnings) provide two independent side-effect sources in PostgreSQL" },
    { icon: "🕸️", label: "Drugs, diseases and side effects are linked in a Neo4j knowledge graph with human-annotated and model-predicted edges" },
    { icon: "⚖️", label: "Evidence fusion ranks the evidence (clinical reference > human annotation > model prediction) into a verdict, with every factor shown" },
  ];
  return (
    <div style={{ marginTop: "1.5rem" }}>
      <button onClick={() => setOpen(!open)} style={{ background: "none", border: "none", color: "#1a56db", fontSize: "0.85rem", fontWeight: 600, cursor: "pointer", padding: 0 }}>
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

function PlainSummary({ text }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: "1.25rem" }}>
      <button onClick={() => setOpen(!open)} style={{ background: "none", border: "none", color: "#1a56db", fontSize: "0.85rem", fontWeight: 600, cursor: "pointer", padding: 0 }}>
        {open ? "▾" : "▸"} Plain-language explanation
      </button>
      {open && (
        <pre style={{ whiteSpace: "pre-wrap", fontFamily: "inherit", fontSize: "0.85rem", color: "#444", backgroundColor: "#fafafa", border: "1px solid #eee", borderRadius: "8px", padding: "0.9rem", marginTop: "0.6rem", lineHeight: 1.55 }}>
          {text}
        </pre>
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
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drug: d, disease: dis }),
      });
      if (!response.ok) {
        let detail = "";
        try { detail = (await response.json()).detail; } catch (_) { /* ignore */ }
        throw new Error(`Server error ${response.status}${detail ? `: ${typeof detail === "string" ? detail : JSON.stringify(detail)}` : ""}`);
      }
      setResult(await response.json());
    } catch (err) {
      setError(err.message === "Failed to fetch" ? `Cannot reach the API at ${API_URL}. Is the backend running?` : err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = () => runAnalysis(drug, disease);
  const handleExample = (ex) => { setDrug(ex.drug); setDisease(ex.disease); runAnalysis(ex.drug, ex.disease); };
  const handleKeyDown = (e) => { if (e.key === "Enter") handleAnalyze(); };

  const verdict = result ? (VERDICT_STYLES[result.suitability.verdict] || VERDICT_STYLES.NO_DIRECT_EVIDENCE) : null;
  const inputStyle = { width: "100%", padding: "0.65rem 0.8rem", borderRadius: "8px", border: "1px solid #ddd", fontSize: "0.95rem", boxSizing: "border-box" };
  const labelStyle = { display: "block", fontSize: "0.85rem", fontWeight: 600, marginBottom: "0.4rem", color: "#444" };

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#fafafa", fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      <div style={{ maxWidth: "820px", margin: "0 auto", padding: "3rem 1.25rem" }}>
        <header style={{ textAlign: "center", marginBottom: "1.5rem" }}>
          <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "#1a56db", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
            AI-POWERED DRUG SAFETY ANALYSIS
          </div>
          <h1 style={{ fontSize: "1.9rem", margin: "0 0 0.3rem", color: "#1a1a1a" }}>Pharmacovigilance Advisory</h1>
          <p style={{ color: "#777", fontSize: "0.95rem", maxWidth: "560px", margin: "0 auto" }}>
            Checks drug–disease suitability, side effects and safer alternatives using transformer-based relation
            extraction, a biomedical knowledge graph and two drug-safety databases.
          </p>
        </header>

        <div style={card}>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: "200px" }}>
              <label htmlFor="drug-input" style={labelStyle}>Drug Name</label>
              <input id="drug-input" value={drug} onChange={(e) => setDrug(e.target.value)} onKeyDown={handleKeyDown} placeholder="e.g. aspirin" style={inputStyle} />
            </div>
            <div style={{ flex: 1, minWidth: "200px" }}>
              <label htmlFor="disease-input" style={labelStyle}>Disease Name</label>
              <input id="disease-input" value={disease} onChange={(e) => setDisease(e.target.value)} onKeyDown={handleKeyDown} placeholder="e.g. hypertension" style={inputStyle} />
            </div>
          </div>

          <div style={{ marginTop: "0.8rem", display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: "0.8rem", color: "#999" }}>Try:</span>
            {EXAMPLES.map((ex, i) => (
              <button key={i} onClick={() => handleExample(ex)} style={{ fontSize: "0.78rem", padding: "0.3rem 0.7rem", borderRadius: "999px", border: "1px solid #ddd", backgroundColor: "#f7f7f7", color: "#555", cursor: "pointer" }}>
                {ex.drug} + {ex.disease}
              </button>
            ))}
          </div>

          <button onClick={handleAnalyze} disabled={loading} style={{ marginTop: "1.25rem", width: "100%", padding: "0.75rem", borderRadius: "8px", border: "none", backgroundColor: loading ? "#a0a0a0" : "#1a56db", color: "#fff", fontSize: "1rem", fontWeight: 600, cursor: loading ? "default" : "pointer" }}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>

          {error && <p role="alert" style={{ color: "#c0392b", marginTop: "1rem", fontSize: "0.9rem" }}>{error}</p>}
          <HowItWorks />
        </div>

        {result && (
          <div style={{ ...card, marginTop: "1.5rem" }}>
            <div style={{ backgroundColor: verdict.bg, borderRadius: "10px", padding: "1.25rem", borderLeft: `5px solid ${verdict.color}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "0.5rem" }}>
                <h2 style={{ margin: 0, color: verdict.color, fontSize: "1.3rem" }}>{verdict.icon} {verdict.label}</h2>
                <SourceBadge suitability={result.suitability} />
              </div>
              <p style={{ margin: "0.6rem 0 0", color: "#333", fontSize: "0.95rem", lineHeight: 1.5 }}>{result.suitability.reason}</p>
              <p style={{ margin: "0.5rem 0 0", fontSize: "0.78rem", color: "#888" }}>
                Query: <strong>{result.drug}</strong> for <strong>{result.disease}</strong>
              </p>
            </div>

            <Section title="Why this verdict" icon="🧮"><ScoreBreakdown factors={result.score_breakdown} /></Section>
            <Section title="Side Effects" icon="💊"><SideEffects result={result} /></Section>
            <Section title="Suggested Alternatives" icon="🔄"><Alternatives result={result} /></Section>
            <Section title="Supporting Evidence" icon="🔍"><Evidence result={result} /></Section>
            <PlainSummary text={result.explanation} />
          </div>
        )}

        <footer style={{ textAlign: "center", marginTop: "2.5rem", fontSize: "0.78rem", color: "#aaa", lineHeight: 1.6 }}>
          Research prototype — not medical advice. Data: BC5CDR, BioRED, ADE Corpus, SIDER, OnSIDES.
        </footer>
      </div>
    </div>
  );
}

export default App;
