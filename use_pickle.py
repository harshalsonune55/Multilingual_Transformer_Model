"""Load hate_speech_model.pkl and classify text.
Needs: pip install torch transformers
Usage: python use_pickle.py "your text here"
"""
import pickle, re, sys, torch

with open("hate_speech_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model, tok, max_len, id2label = bundle["model"], bundle["tokenizer"], bundle["max_len"], bundle["id2label"]

def clean(t):
    t = re.sub(r"http\S+|www\.\S+", " ", str(t))
    t = re.sub(r"@\w+", "@user", t)
    return re.sub(r"\s+", " ", t).strip()

def predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    enc = tok([clean(t) for t in texts], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)
    return [{"text": t, "label": id2label[int(p.argmax())], "hate_probability": float(p[1])}
            for t, p in zip(texts, probs)]

if __name__ == "__main__":
    for r in predict(sys.argv[1:] or ["hello", "fuck you"]):
        print(r)
