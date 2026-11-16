import os, re, warnings
from collections import defaultdict
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEAT, DS, OUT = "features/ffpp", "ffpp", "results/ffpp_per_method_v2.csv"
RS, K, BATCH = 42, 4, 4096
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

def method_of(v):
    m = re.search(r"fake/([^/]+)/", str(v))
    return m.group(1) if m else ("real" if "/real/" in str(v) else "?")

meta = np.load(os.path.join(FEAT, f"{DS}_hog_meta.npz"), allow_pickle=True)
X = np.memmap(os.path.join(FEAT, f"{DS}_hog.dat"), dtype=np.float32, mode="r",
              shape=tuple(int(x) for x in meta["shape"]))
split, label, vid = meta["split"], meta["label"], meta["video_id"]
tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]

# same subsampling as evaluate_balanced.py (k=4, seed 42)
rng = np.random.RandomState(RS); by = defaultdict(list)
for i in tr: by[vid[i]].append(i)
sel = []
for v, idx in by.items():
    sel.extend(idx if len(idx) <= K else rng.choice(idx, K, replace=False).tolist())
sel = np.array(sorted(sel), dtype=np.int64)
ytr = label[sel]
print(f"[train] rows={len(sel)} fake={int((ytr==0).sum())} real={int((ytr==1).sum())}", flush=True)

sc = StandardScaler(copy=False)
Xtr = sc.fit_transform(np.asarray(X[sel], dtype=np.float32))
clf = LinearSVC(C=1.0, random_state=RS, max_iter=10000).fit(Xtr, ytr)
print(f"[train] training accuracy = {clf.score(Xtr, ytr):.4f}", flush=True)
del Xtr

s = np.empty(len(te), dtype=np.float32)
for b in range(0, len(te), BATCH):
    c = te[b:b+BATCH]
    s[b:b+len(c)] = clf.decision_function(sc.transform(np.asarray(X[c], dtype=np.float32)))

bv = defaultdict(list); yl = {}
for v, y, x in zip(vid[te], label[te], s): bv[v].append(x); yl[v] = int(y)
vids = list(bv); vs = np.array([np.mean(bv[v]) for v in vids])
vy = np.array([yl[v] for v in vids]); vm = np.array([method_of(v) for v in vids])

pooled = roc_auc_score(vy, vs)
print(f"\n[check] pooled AUC = {pooled:.4f} (paper 0.7513) -> "
      f"{'REPRODUCED' if abs(pooled-0.7513) <= 0.002 else 'NOT REPRODUCED'}")
real = vm == "real"
rows = ["method,n_fake_videos,HOG_AUC"]
for m in METHODS:
    f = vm == m
    a = roc_auc_score(np.r_[vy[real], vy[f]], np.r_[vs[real], vs[f]])
    print(f"  {m:15s} n={int(f.sum())}  HOG_AUC={a:.4f}")
    rows.append(f"{m},{int(f.sum())},{a:.4f}")
open(OUT, "w").write("\n".join(rows) + "\n")
print(f"[done] -> {OUT}")
