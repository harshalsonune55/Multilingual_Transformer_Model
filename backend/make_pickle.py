import pickle, torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "model_v3"
model = AutoModelForSequenceClassification.from_pretrained(SRC).cpu().eval()
tokenizer = AutoTokenizer.from_pretrained(SRC)

with open(ROOT / "hate_speech_model.pkl", "wb") as f:
    pickle.dump({"model": model, "tokenizer": tokenizer, "max_len": 64,
                 "id2label": {0: "None", 1: "Hate"}}, f, protocol=pickle.HIGHEST_PROTOCOL)
print("done")
