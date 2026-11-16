"""
Evaluate handcrafted descriptors on the Celeb-DF IDENTITY-DISJOINT split,
WITHOUT re-extracting features. Now includes SVM, RF, AND XGBoost (matching
the other datasets' three-classifier set).

HOG/etc. per-frame features are identical regardless of split; only the
train/test assignment changes. So we:
  1. load existing memmap features + per-row video_id (from meta)
  2. read the id-disjoint manifest -> map video_id -> {train,val,test}
  3. re-label each feature row's split accordingly
  4. train on train rows (subsampled 4/video), evaluate on test, video-level

Matches the main pipeline protocol: majority vote (ties->real) for label,
mean frame score for AUC, macro metrics, class-weighted / scale_pos_weight
when --balanced.

Run:
  python eval_idsplit_handcrafted.py \
     --feat-dir ../features/celebdf2 --name celebdf2 --descriptor hog \
     --idsplit-manifest ../manifests/celebdf2_idsplit_frames.csv \
     --out ../results/celebdf2_idsplit_handcrafted.csv \
     --train-frames-per-video 4 --balanced
"""
import argparse, csv, os
from collections import defaultdict
import warnings
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RS = 42
BATCH = 4096


def load_meta(feat_dir, name, desc):
    meta = np.load(os.path.join(feat_dir, f"{name}_{desc}_meta.npz"), allow_pickle=True)
    shape = tuple(int(x) for x in meta["shape"])
    X = np.memmap(os.path.join(feat_dir, f"{name}_{desc}.dat"), dtype=np.float32,
                  mode="r", shape=shape)
    return X, meta["video_id"], meta["label"]


def read_split_map(manifest):
    m = {}
    for r in csv.DictReader(open(manifest)):
        m[r["video_id"]] = r["split"]
    return m


def subsample(rows_idx, vids, k, seed=42):
    rng = np.random.RandomState(seed); by = defaultdict(list)
    for i in rows_idx:
        by[vids[i]].append(i)
    keep = []
    for v, idx in by.items():
        keep.extend(idx if len(idx) <= k else rng.choice(idx, k, replace=False).tolist())
    return np.array(sorted(keep))


def video_agg(vids, y, s):
    """majority vote of frame preds for label (ties->real); mean score for AUC."""
    by = defaultdict(lambda: {"y": None, "pred": [], "score": []})
    for v, yy, pp, ss in zip(vids, y[0], y[1], s):
        by[v]["y"] = int(yy); by[v]["pred"].append(int(pp)); by[v]["score"].append(float(ss))
    order = list(by.keys())
    yt = np.array([by[v]["y"] for v in order])
    yp = np.array([1 if sum(1 for x in by[v]["pred"] if x == 1) >=
                   sum(1 for x in by[v]["pred"] if x == 0) else 0 for v in order])
    sc = np.array([np.mean(by[v]["score"]) for v in order])
    return yt, yp, sc


def metrics(yt, yp, sc):
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    try: auc = roc_auc_score(yt, sc)
    except ValueError: auc = float("nan")
    return dict(P=p, R=r, F1=f1, Acc=accuracy_score(yt, yp), AUC=auc)


def make_models(balanced, spw):
    cw = "balanced" if balanced else None
    return {
        "SVM": LinearSVC(C=1.0, random_state=RS, max_iter=10000, class_weight=cw),
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RS,
                                     class_weight=cw),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     n_jobs=-1, random_state=RS, eval_metric="logloss",
                                     scale_pos_weight=(spw if balanced else 1.0)),
    }


def score_stream(model, name, X, test_idx, scaler):
    preds = np.empty(len(test_idx), dtype=np.int64)
    scores = np.empty(len(test_idx), dtype=np.float32)
    for start in range(0, len(test_idx), BATCH):
        chunk = test_idx[start:start + BATCH]
        Xb = np.asarray(X[chunk], dtype=np.float32)
        if scaler is not None:
            Xb = scaler.transform(Xb)
        preds[start:start+len(chunk)] = model.predict(Xb)
        if name == "SVM":
            d = model.decision_function(Xb)
            scores[start:start+len(chunk)] = 1/(1+np.exp(-d))
        else:
            proba = model.predict_proba(Xb)
            scores[start:start+len(chunk)] = proba[:, list(model.classes_).index(1)]
    return preds, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True); ap.add_argument("--name", default="celebdf2")
    ap.add_argument("--descriptor", required=True)
    ap.add_argument("--idsplit-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="SVM,RF,XGBoost")
    ap.add_argument("--train-frames-per-video", type=int, default=4)
    ap.add_argument("--balanced", action="store_true")
    args = ap.parse_args()

    X, vids, labels = load_meta(args.feat_dir, args.name, args.descriptor)
    smap = read_split_map(args.idsplit_manifest)
    row_split = np.array([smap.get(v, "?") for v in vids])
    tr_idx = np.where(row_split == "train")[0]
    te_idx = np.where(row_split == "test")[0]
    print(f"[remap] train_rows={len(tr_idx)} test_rows={len(te_idx)}", flush=True)

    keep = subsample(tr_idx, vids, args.train_frames_per_video, RS)
    Xtr = np.asarray(X[keep], dtype=np.float32); ytr = labels[keep].astype(int)
    n0, n1 = int((ytr == 0).sum()), int((ytr == 1).sum())
    spw = n0 / max(1, n1)
    print(f"[train] Xtr={Xtr.shape} ({Xtr.nbytes/1e9:.2f} GB) spw={spw:.3f}", flush=True)

    sc = StandardScaler(copy=False); Xtr = sc.fit_transform(Xtr)
    models = [m.strip() for m in args.models.split(",")]

    header = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["dataset","descriptor","classifier","P","R","F1","Acc","AUC","mode","training"])
        for name in models:
            model = make_models(args.balanced, spw)[name]
            model.fit(Xtr, ytr)
            preds, scores = score_stream(model, name, X, te_idx, sc)
            yt, yp, s = video_agg(vids[te_idx], (labels[te_idx], preds), scores)
            m = metrics(yt, yp, s)
            print(f"[result] {args.descriptor}/{name} F1={m['F1']:.3f} AUC={m['AUC']:.3f}", flush=True)
            w.writerow([args.name+"_idsplit", args.descriptor, name,
                        f"{m['P']:.4f}", f"{m['R']:.4f}", f"{m['F1']:.4f}",
                        f"{m['Acc']:.4f}", f"{m['AUC']:.4f}", "idsplit",
                        "balanced" if args.balanced else "default"])
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
