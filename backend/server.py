import re
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = next((d for d in (ROOT / "model_v3", ROOT / "model_v2") if (d / "model.safetensors").exists()), ROOT)
MAX_LEN = 64
ID2LABEL = {0: "None", 1: "Hate"}

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

app = Flask(__name__)
CORS(app)


def clean(t):
    t = re.sub(r"http\S+|www\.\S+", " ", str(t))
    t = re.sub(r"@\w+", "@user", t)
    return re.sub(r"\s+", " ", t).strip()


@app.get("/api/health")
def health():
    return jsonify(status="ok", device=device)


TRAIN_LOG = Path("/tmp/train.log")


@app.get("/api/training")
def training():
    if not TRAIN_LOG.exists():
        return jsonify(active=False)
    log = TRAIN_LOG.read_bytes()[-200_000:].decode("utf-8", errors="ignore")
    steps = re.findall(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+)", log)
    done = "SAVED" in log
    if not steps:
        return jsonify(active=not done, percent=0, status="Starting…")
    step, total, elapsed, eta = steps[-1]
    return jsonify(
        active=not done,
        percent=100.0 if done else int(step) / int(total) * 100,
        step=int(step), total=int(total), elapsed=elapsed, eta=eta,
        status="Training finished" if done else "Training new model…",
    )


@app.post("/api/predict")
def predict():
    body = request.get_json(silent=True) or {}
    texts = body.get("texts") or [body.get("text", "")]
    texts = [clean(t) for t in texts if clean(t)]
    if not texts:
        return jsonify(error="Empty prompt"), 400
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().tolist()
    results = []
    for t, p in zip(texts, probs):
        idx = max(range(len(p)), key=p.__getitem__)
        results.append(dict(text=t, label=ID2LABEL[idx], confidence=p[idx],
                            scores={ID2LABEL[i]: v for i, v in enumerate(p)}))
    return jsonify(results=results)


if __name__ == "__main__":
    app.run(port=8765)
