import { useEffect, useState } from "react";

const EXAMPLES = [
  "Aaj weather mast aahe",
  "आज हवामान खूप छान आहे",
  "tu kon lavadya",
  "तू कोण लवड्या",
  "Tu assignment submit kela ka?",
  "Tum log bahut ganda bolte ho",
  "fuck you",
  "Thanks for helping me",
];

export default function App() {
  const [text, setText] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [train, setTrain] = useState(null);

  useEffect(() => {
    let stop = false;
    async function poll() {
      try {
        const t = await (await fetch("/api/training")).json();
        if (!stop) setTrain(t);
      } catch {}
    }
    poll();
    const id = setInterval(poll, 4000);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, []);

  async function check(value = text) {
    if (!value.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          texts: value.split("\n").map((l) => l.replace(/^\s*\d+[.)]\s*/, "").trim()).filter(Boolean),
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Cannot reach the model server. Is it running?");
      setResults(data.results);
    } catch (e) {
      setResults([]);
      setError(e.message === "Failed to fetch" ? "Cannot reach the model server." : e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="card">
      {train?.active && train.percent > 0 && (
        <div className="train">
          <div className="train-head">
            <span>{train.status}</span>
            <span>{train.percent.toFixed(1)}%</span>
          </div>
          <div className="bar train-bar">
            <div style={{ width: `${train.percent}%` }} />
          </div>
          {train.active && (
            <div className="meta">
              <span>Step {train.step} / {train.total}</span>
              <span>Elapsed {train.elapsed} · ETA {train.eta}</span>
            </div>
          )}
        </div>
      )}

      <h1>Prompt Checker</h1>
      <p className="sub">Detects hate speech in Marathi, Hindi, Hinglish and English</p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => (e.metaKey || e.ctrlKey) && e.key === "Enter" && check()}
        placeholder="Type a prompt (one per line to check several)…"
        rows={5}
      />

      <div className="chips">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="chip"
            onClick={() => {
              setText(ex);
              check(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      <button className="primary" onClick={() => check()} disabled={loading || !text.trim()}>
        {loading ? "Checking…" : "Check prompt"}
      </button>

      {error && <div className="error">{error}</div>}

      {results.map((r, i) => {
        const hate = r.label === "Hate";
        return (
          <section key={i} className={`result ${hate ? "bad" : "good"}`}>
            {results.length > 1 && <div className="quote">{r.text}</div>}
            <div className="verdict">{hate ? "✗ Not good — hate speech detected" : "✓ Good — no hate speech"}</div>
            <div className="bar">
              <div style={{ width: `${r.scores.Hate * 100}%` }} />
            </div>
            <div className="meta">
              <span>Hate probability: {(r.scores.Hate * 100).toFixed(1)}%</span>
              <span>Confidence: {(r.confidence * 100).toFixed(1)}%</span>
            </div>
          </section>
        );
      })}
    </main>
  );
}
