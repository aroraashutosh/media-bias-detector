"""
TextCNN (Kim 2014) vs Logistic Regression on BABE media-bias sentences.
Same 80/20 split as the earlier baseline (random_state=42, stratified).
A validation set is carved out of the TRAIN part only, for early stopping,
so the test set is evaluated exactly once per model at the end.
"""
import sys, json, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

DATA = sys.argv[1]
OUT = sys.argv[2]
torch.set_num_threads(2)

df = pd.read_excel(DATA)
df = df[df["label_bias"].isin(["Biased", "Non-biased"])].copy()
df["label"] = (df["label_bias"] == "Biased").astype(int)
X_tr_all, X_te, y_tr_all, y_te = train_test_split(
    df["text"].tolist(), df["label"].values, test_size=0.2, random_state=42, stratify=df["label"])
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr_all, y_tr_all, test_size=0.1, random_state=42, stratify=y_tr_all)
print(f"train {len(X_tr)}  val {len(X_va)}  test {len(X_te)}")

def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="binary")
    return dict(acc=accuracy_score(y, p), prec=pr, rec=rc, f1=f1)

results = {}

# ---- Baseline: TF-IDF + Logistic Regression (trained on full 80%, as before) ----
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vec.fit_transform(X_tr_all), y_tr_all)
results["LogReg (TF-IDF)"] = [metrics(y_te, lr.predict(vec.transform(X_te)))]
print("LogReg", results["LogReg (TF-IDF)"][0])

# ---- Tokenise + vocab + pretrained vectors ----
nlp = spacy.load("en_core_web_md")
def tok(t): return [w.text.lower() for w in nlp.make_doc(t) if not w.is_space]
toks = {k: [tok(t) for t in v] for k, v in dict(tr=X_tr, va=X_va, te=X_te).items()}
vocab = {"<pad>": 0, "<unk>": 1}
for s in toks["tr"]:
    for w in s: vocab.setdefault(w, len(vocab))
MAXLEN = 60
def enc(sents):
    out = np.zeros((len(sents), MAXLEN), dtype=np.int64)
    for i, s in enumerate(sents):
        ids = [vocab.get(w, 1) for w in s][:MAXLEN]
        out[i, :len(ids)] = ids
    return torch.tensor(out)
E = {k: enc(v) for k, v in toks.items()}
Y = {"tr": torch.tensor(y_tr), "va": torch.tensor(y_va), "te": torch.tensor(y_te)}

emb_matrix = np.random.RandomState(0).normal(0, 0.1, (len(vocab), 300)).astype("float32")
emb_matrix[0] = 0
hit = 0
for w, i in vocab.items():
    lex = nlp.vocab[w]
    if lex.has_vector:
        emb_matrix[i] = lex.vector; hit += 1
print(f"vocab {len(vocab)}, pretrained vector coverage {hit/len(vocab):.1%}")

class TextCNN(nn.Module):
    def __init__(self, pretrained, static):
        super().__init__()
        if pretrained:
            self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix), freeze=static, padding_idx=0)
        else:
            self.emb = nn.Embedding(len(vocab), 300, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, k) for k in (2, 3, 4, 5)])
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(400, 2)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = torch.cat([F.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)
        return self.fc(self.drop(h))

def run(pretrained, static, seed, epochs=25):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = TextCNN(pretrained, static)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    best_va, best_state = -1, None
    n = len(Y["tr"])
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i+64]
            opt.zero_grad()
            F.cross_entropy(m(E["tr"][idx]), Y["tr"][idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            va = (m(E["va"]).argmax(1) == Y["va"]).float().mean().item()
        if va > best_va:
            best_va = va
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        p = m(E["te"]).argmax(1).numpy()
    return metrics(y_te, p), best_va, p

configs = {
    "CNN (random embeddings)": (False, False),
    "CNN (pretrained, static)": (True, True),
    "CNN (pretrained, fine-tuned)": (True, False),
}
SEEDS = [0, 1, 2, 3, 4]
last_pred = {}
for name, (pre, static) in configs.items():
    runs = []
    for s in SEEDS:
        met, va, p = run(pre, static, s)
        runs.append(met); last_pred[name] = p
        print(f"{name} seed {s}: val {va:.3f}  test acc {met['acc']:.3f}", flush=True)
    results[name] = runs

summary = {}
for name, runs in results.items():
    a = np.array([[r["acc"], r["prec"], r["rec"], r["f1"]] for r in runs])
    summary[name] = dict(acc_mean=a[:, 0].mean(), acc_std=a[:, 0].std(), prec=a[:, 1].mean(),
                         rec=a[:, 2].mean(), f1=a[:, 3].mean(), n_runs=len(runs))
json.dump(summary, open(OUT, "w"), indent=2)
print("\n=== SUMMARY (test set, mean over seeds) ===")
for k, v in summary.items():
    print(f"{k:32s} acc {v['acc_mean']:.3f} ± {v['acc_std']:.3f}  prec {v['prec']:.3f}  rec {v['rec']:.3f}  f1 {v['f1']:.3f}  (n={v['n_runs']})")
