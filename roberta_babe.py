"""
Fine-tune RoBERTa on BABE (media bias, Biased vs Non-biased) -- Google Colab version.

Same data and same 80/20 test split as the earlier Logistic Regression (73.2%) and
CNN (73.1%) runs, so the numbers are directly comparable.
Run on a GPU runtime. Takes roughly 5-10 minutes.
"""
import io, json, random, urllib.request
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL = "roberta-base"
URL = ("https://raw.githubusercontent.com/Media-Bias-Group/Neural-Media-Bias-Detection-"
       "Using-Distant-Supervision-With-BABE/main/data/final_labels_SG2.xlsx")
EPOCHS, BATCH, LR, MAXLEN = 3, 16, 2e-5, 96
SEED = 2   # <-- change this number for each run (42, 1, 2 ...)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cpu":
    print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU, then run again.")

# ---- 1. Data: download BABE and make the SAME split as before ----
raw = urllib.request.urlopen(URL).read()
df = pd.read_excel(io.BytesIO(raw))
df = df[df["label_bias"].isin(["Biased", "Non-biased"])].copy()
df["label"] = (df["label_bias"] == "Biased").astype(int)
X_all, X_te, y_all, y_te = train_test_split(
    df["text"].tolist(), df["label"].values, test_size=0.2, random_state=42, stratify=df["label"])
X_tr, X_va, y_tr, y_va = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
print(f"train {len(X_tr)}  val {len(X_va)}  test {len(X_te)}")

# ---- 2. Load pretrained RoBERTa (downloads from Hugging Face) ----
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)

def encode(texts):
    return tok(texts, truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt")

enc = {k: encode(v) for k, v in dict(tr=X_tr, va=X_va, te=X_te).items()}
lab = {"tr": torch.tensor(y_tr), "va": torch.tensor(y_va), "te": torch.tensor(y_te)}

def predict(split):
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(lab[split]), 64):
            b = {k: v[i:i+64].to(device) for k, v in enc[split].items()}
            preds.append(model(**b).logits.argmax(1).cpu())
    return torch.cat(preds).numpy()

# ---- 3. Fine-tune ----
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(y_tr) + BATCH - 1) // BATCH)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
best_va, best_state = -1, None
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(y_tr))
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        batch = {k: v[idx].to(device) for k, v in enc["tr"].items()}
        out = model(**batch, labels=lab["tr"][idx].to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
    va = accuracy_score(y_va, predict("va"))
    print(f"epoch {ep+1}/{EPOCHS}  validation accuracy {va:.3f}", flush=True)
    if va > best_va:
        best_va = va
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

# ---- 4. Final test (only once, using the best epoch picked on validation) ----
model.load_state_dict(best_state)
p = predict("te")
acc = accuracy_score(y_te, p)
prec, rec, f1, _ = precision_recall_fscore_support(y_te, p, average="binary")
cm = confusion_matrix(y_te, p)

# ---- 5. Our own geopolitical sentences ----
ours = [
    "Pakistan's navy was lawfully carrying out a routine, pre-planned exercise.",
    "The Pakistani vessel tailed the Indian ship, ignored repeated warning signals, and recklessly caused the crash.",
]
model.eval()
with torch.no_grad():
    b = {k: v.to(device) for k, v in encode(ours).items()}
    probs = torch.softmax(model(**b).logits, dim=1).cpu().numpy()

print("\n" + "=" * 60)
print("COPY EVERYTHING BELOW THIS LINE AND SEND IT TO CLAUDE")
print("=" * 60)
print(f"Model: {MODEL}   epochs: {EPOCHS}   test sentences: {len(y_te)}   SEED USED: {SEED}")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")
for s, pr in zip(ours, probs):
    print(f"\n\"{s}\"\n  P(non-biased)={pr[0]:.2f}  P(biased)={pr[1]:.2f}")
json.dump(dict(model=MODEL, acc=acc, prec=prec, rec=rec, f1=f1, cm=cm.tolist()),
          open("roberta_results.json", "w"), indent=2)
