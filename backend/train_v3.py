import re, unicodedata, numpy as np, pandas as pd
from datasets import Dataset
from indic_transliteration import sanscript
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding)

ROOT = "/Users/harshalsonune/Documents/aayuumodel"
SRC, OUT = f"{ROOT}/model_v2", f"{ROOT}/model_v3"
MAX_LEN, BATCH, EPOCHS, LR = 64, 32, 2, 1.5e-5
rng = np.random.RandomState(42)

def clean(t):
    t = re.sub(r"http\S+|www\.\S+", " ", str(t))
    t = re.sub(r"@\w+", "@user", t)
    return re.sub(r"\s+", " ", t).strip()

DEV = re.compile(r"[ऀ-ॿ]+")
def romanize(t):
    """Devanagari -> plain lowercase Roman, the way people type Marathi/Hindi in chat."""
    def f(m):
        s = sanscript.transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.IAST)
        s = unicodedata.normalize("NFD", s)
        return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return DEV.sub(f, t)

def prism(s):
    d = pd.read_csv(f"{ROOT}/data/prism_{s}.csv").dropna(subset=["text", "label"])
    d["text"] = d["text"].map(clean); d["label"] = d["label"].astype(int)
    return d[d.text.str.len() > 0][["text", "label"]]

def maha(s):
    d = pd.read_excel(f"{ROOT}/data/maha_{s}.xlsx").dropna()
    d["text"] = d["text"].map(clean); d["label"] = (d["label"] == "HOF").astype(int)
    return d[d.text.str.len() > 0][["text", "label"]]

mine = pd.read_csv(f"{ROOT}/marathi_english_code_mixed_sample.csv", encoding="utf-8-sig")[["text", "label"]]

def codemix(names):
    d = pd.concat([pd.read_csv(f"{ROOT}/{n}.csv") for n in names])
    d = d.dropna(subset=["tweet", "hate"]).rename(columns={"tweet": "text", "hate": "label"})
    d["text"] = d["text"].map(clean); d["label"] = d["label"].astype(int)
    d = d[d.text.str.len() > 0][["text", "label"]]
    return d.drop_duplicates("text")

cm_test = codemix(["codemix-hate-small-test"])
cm_all = codemix(["codemix-hate-small-train", "codemix-sent-train", "codemix-sent-test"])
cm_all = cm_all[~cm_all.text.isin(set(cm_test.text))]          # no test leakage
cm_all = cm_all.reset_index(drop=True)
cm_val = pd.concat([cm_all[cm_all.label == l].sample(frac=0.08, random_state=5) for l in (0, 1)])
cm_train = cm_all.drop(cm_val.index)
cm_train = pd.concat([cm_train, cm_train[cm_train.label == 1]])  # hate is rarer here: upweight x2
print("codemix train", len(cm_train), "val", len(cm_val), "test", len(cm_test), flush=True)

mt, pt = maha("train"), prism("train")
mt = mt.sample(frac=1, random_state=1)
roman = mt.iloc[:3000].copy(); roman["text"] = roman["text"].map(romanize)   # Roman-script Marathi
train = pd.concat([cm_train, mt.iloc[3000:13000], roman, pt.sample(6000, random_state=2)] + [mine] * 5)
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

mv, pv = maha("valid"), prism("val")
val = pd.concat([cm_val, mv.sample(500, random_state=3), pv.sample(500, random_state=3)]).reset_index(drop=True)
mte, pte = maha("test"), prism("test")
for n, d in [("train", train), ("val", val), ("cm_test", cm_test), ("maha_test", mte), ("prism_test", pte)]:
    assert d.label.isin([0, 1]).all() and d.text.str.len().min() > 0, f"bad data in {n}"
    print(n, len(d), "hate share %.2f" % d.label.mean(), flush=True)
print("train", len(train), "val", len(val), flush=True)

tok = AutoTokenizer.from_pretrained(SRC)
def ds(d):
    return Dataset.from_pandas(d.reset_index(drop=True)).map(
        lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])
model = AutoModelForSequenceClassification.from_pretrained(SRC)

def metrics(p):
    pr = p.predictions.argmax(1)
    a, b, f, _ = precision_recall_fscore_support(p.label_ids, pr, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(p.label_ids, pr), precision=a, recall=b, f1=f)

args = TrainingArguments(
    output_dir=f"{ROOT}/results_v3", num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=64, weight_decay=0.01,
    warmup_steps=80, eval_strategy="epoch", save_strategy="no", logging_steps=50,
    seed=42, report_to="none", dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, train_dataset=ds(train), eval_dataset=ds(val),
                  processing_class=tok, data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
trainer.train()

for name, d in [("MARATHI-ENGLISH CODE-MIXED (your CSV, held out)", cm_test), ("MARATHI (Devanagari)", mte), ("HINDI/HINGLISH/ENGLISH (PRISM)", pte)]:
    o = trainer.predict(ds(d))
    print("=====", name, flush=True)
    print(classification_report(o.label_ids, o.predictions.argmax(1), target_names=["None", "Hate"], digits=3), flush=True)
trainer.save_model(OUT); tok.save_pretrained(OUT)
print("SAVED", OUT, flush=True)
