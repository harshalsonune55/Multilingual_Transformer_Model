import re, numpy as np, pandas as pd, torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding, EarlyStoppingCallback)

ROOT = "/Users/harshalsonune/Documents/aayuumodel"
OUT = f"{ROOT}/model_v2"
BASE, MAX_LEN, BATCH, EPOCHS, LR = "xlm-roberta-base", 64, 32, 3, 2e-5

def clean(t):
    t = re.sub(r"http\S+|www\.\S+", " ", str(t))
    t = re.sub(r"@\w+", "@user", t)
    return re.sub(r"\s+", " ", t).strip()

def load(p):
    d = pd.read_csv(p).dropna(subset=["text", "label"])
    d["text"] = d["text"].map(clean)
    return d[d.text.str.len() > 0][["text", "label"]].astype({"label": int})

train, val, test = (load(f"{ROOT}/data/prism_{s}.csv") for s in ("train", "val", "test"))
mine = load(f"{ROOT}/marathi_english_code_mixed_sample.csv")
train = pd.concat([train] + [mine] * 5).sample(frac=1, random_state=42).reset_index(drop=True)
print(len(train), len(val), len(test), flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
def ds(d):
    return Dataset.from_pandas(d.reset_index(drop=True)).map(
        lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])
tr, va, te = ds(train), ds(val), ds(test)

id2label = {0: "None", 1: "Hate"}
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=2, id2label=id2label, label2id={v: k for k, v in id2label.items()})

def metrics(p):
    pr = p.predictions.argmax(1)
    a, b, f, _ = precision_recall_fscore_support(p.label_ids, pr, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(p.label_ids, pr), precision=a, recall=b, f1=f)

args = TrainingArguments(
    output_dir=f"{ROOT}/results_v2", num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=64, weight_decay=0.01,
    warmup_steps=167, eval_strategy="epoch", save_strategy="no", logging_steps=50,
    seed=42, report_to="none", dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                  processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=metrics)
trainer.train()

out = trainer.predict(te)
print(classification_report(out.label_ids, out.predictions.argmax(1), target_names=["None", "Hate"]), flush=True)
trainer.save_model(OUT); tok.save_pretrained(OUT)
print("SAVED", OUT, flush=True)
