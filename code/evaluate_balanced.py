"""
Memory-safe, single-descriptor, append-mode evaluation (handcrafted).

REDESIGN for low-RAM machines (e.g. 31GB) vs huge HOG (208928 x 34596 ~29GB):
  * process ONE descriptor per invocation (--descriptors hog) and APPEND to
    a shared CSV. Peak RAM = one descriptor, freed fully between runs.
  * training matrix is subsampled to k frames/video BEFORE materializing,
    and we materialize only the (small) train rows we actually use.
  * test set is scored in BATCHES streamed from the memmap -> the full test
    matrix is never copied into RAM.
  * StandardScaler fit on the small train matrix; applied to test batches.
  * flush=True everywhere; CSV written incrementally per classifier.

Run ONE descriptor at a time (recommended for celebdf2):
  python evaluate_balanced.py --feat-dir ../features/celebdf2 --dataset celebdf2 \
     --descriptor hog --out ../results_balanced/celebdf2_balanced.csv \
     --train-frames-per-video 8 --balanced
  ... repeat with --descriptor sift / orb / lbp (appends to same CSV).

Header is written once; subsequent runs append rows.
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score)
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RANDOM_STATE = 42
BATCH = 4096   # test-scoring batch size (rows streamed from memmap)


def log(*a):
    print(*a, flush=True)


def load_feature(feat_dir, dataset, desc):
    meta = np.load(os.path.join(feat_dir, f"{dataset}_{desc}_meta.npz"),
                   allow_pickle=True)
    shape = tuple(int(x) for x in meta["shape"])
    X = np.memmap(os.path.join(feat_dir, f"{dataset}_{desc}.dat"),
                  dtype=np.float32, mode="r", shape=shape)
    return X, meta


def make_models(balanced, spw=1.0):
    cw = "balanced" if balanced else None
    return {
        "SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=10000,
                         class_weight=cw),
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                     random_state=RANDOM_STATE,
                                     class_weight=("balanced" if balanced else None)),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                     learning_rate=0.1, n_jobs=-1,
                                     random_state=RANDOM_STATE,
                                     eval_metric="logloss",
                                     scale_pos_weight=(spw if balanced else 1.0)),
    }


def subsample_train_indices(train_idx, video_ids_all, k_per_video, seed=42):
    """Return a SMALL array of selected row indices (<=k per video)."""
    if k_per_video is None or k_per_video <= 0:
        return train_idx
    rng = np.random.RandomState(seed)
    by_vid = defaultdict(list)
    for i in train_idx:
        by_vid[video_ids_all[i]].append(i)
    sel = []
    for v, idxs in by_vid.items():
        if len(idxs) <= k_per_video:
            sel.extend(idxs)
        else:
            sel.extend(rng.choice(idxs, size=k_per_video, replace=False).tolist())
    return np.array(sorted(sel), dtype=np.int64)


def materialize(X, idx):
    """Copy ONLY the selected rows out of the memmap (small train set)."""
    return np.asarray(X[idx], dtype=np.float32)


def score_test_streaming(model, name, X, test_idx, scaler):
    """Stream test rows from memmap in batches; never copy full test matrix.
    Returns per-frame predicted label and score (P(real))."""
    preds = np.empty(len(test_idx), dtype=np.int64)
    scores = np.empty(len(test_idx), dtype=np.float32)
    for start in range(0, len(test_idx), BATCH):
        chunk = test_idx[start:start + BATCH]
        Xb = np.asarray(X[chunk], dtype=np.float32)
        if scaler is not None:
            Xb = scaler.transform(Xb)
        preds[start:start + len(chunk)] = model.predict(Xb)
        if name == "SVM":
            scores[start:start + len(chunk)] = model.decision_function(Xb)
        else:
            proba = model.predict_proba(Xb)
            ridx = list(model.classes_).index(1)
            scores[start:start + len(chunk)] = proba[:, ridx]
        del Xb
    return preds, scores


def aggregate(video_ids, y_true, preds, scores):
    by = defaultdict(lambda: {"yt": None, "pred": [], "score": []})
    for v, yt, p, s in zip(video_ids, y_true, preds, scores):
        by[v]["yt"] = yt; by[v]["pred"].append(p); by[v]["score"].append(s)
    vids = list(by.keys())
    yt = np.array([by[v]["yt"] for v in vids])
    yp = np.array([1 if sum(1 for x in by[v]["pred"] if x == 1) >=
                   sum(1 for x in by[v]["pred"] if x == 0) else 0 for v in vids])
    sc = np.array([np.mean(by[v]["score"]) for v in vids])
    return yt, yp, sc


def metrics(yt, yp, sc):
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro",
                                                  zero_division=0)
    acc = accuracy_score(yt, yp)
    try:
        auc = roc_auc_score(yt, sc)
    except ValueError:
        auc = float("nan")
    return dict(P=p, R=r, F1=f1, Acc=acc, AUC=auc)


def append_row(out_csv, row, write_header):
    mode = "w" if write_header else "a"
    with open(out_csv, mode) as f:
        if write_header:
            f.write("dataset,descriptor,classifier,P,R,F1,Acc,AUC,mode,training\n")
        f.write(row + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--descriptor", required=True, help="ONE of hog/sift/orb/lbp")
    ap.add_argument("--models", default="SVM,RF,XGBoost")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--train-frames-per-video", type=int, default=0)
    args = ap.parse_args()

    desc = args.descriptor.strip().lower()
    models = [m.strip() for m in args.models.split(",")]
    scale = not args.no_scale
    bal = args.balanced
    tag = "balanced" if bal else "default"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_header = not os.path.exists(args.out)

    log(f"[load] {args.dataset}/{desc} ...")
    X, meta = load_feature(args.feat_dir, args.dataset, desc)
    split = meta["split"]; label = meta["label"]; vid = meta["video_id"]
    is_cv = (split[0] == "cv")
    mode = "5foldCV" if is_cv else "split"
    log(f"[load] X={X.shape} mode={mode} training={tag}")

    if is_cv:
        log("[error] this redesigned script handles SPLIT mode (celebdf2/ffpp) "
            "only. Use the normal evaluate for UADFV (small, no memory issue).")
        sys.exit(1)

    train_idx = np.where(split == "train")[0]
    test_idx = np.where(split == "test")[0]
    log(f"[split] train_frames={len(train_idx)} test_frames={len(test_idx)}")

    sub_idx = subsample_train_indices(train_idx, vid,
                                      args.train_frames_per_video, RANDOM_STATE)
    log(f"[subsample] train rows used={len(sub_idx)} "
        f"(k/video={args.train_frames_per_video})")

    ytr = label[sub_idx]
    n0, n1 = int(np.sum(ytr == 0)), int(np.sum(ytr == 1))
    spw = (n0 / max(1, n1)) if bal else 1.0
    log(f"[balance] train class counts: fake(0)={n0} real(1)={n1} "
        f"scale_pos_weight={spw:.3f} balanced={bal}")

    log("[materialize] copying subsampled TRAIN rows from memmap ...")
    Xtr = materialize(X, sub_idx)
    log(f"[materialize] Xtr={Xtr.shape} ({Xtr.nbytes/1e9:.2f} GB)")

    scaler = None
    if scale:
        log("[scale] fitting StandardScaler on train (copy=False) ...")
        scaler = StandardScaler(copy=False)
        Xtr = scaler.fit_transform(Xtr)

    yte = label[test_idx]; vte = vid[test_idx]

    for name in models:
        log(f"[train] {desc}/{name} start ...")
        spw_eff = spw if bal else 1.0
        model = make_models(bal, spw_eff)[name]
        model.fit(Xtr, ytr)
        log(f"[train] {desc}/{name} done; scoring test (streaming) ...")
        preds, scores = score_test_streaming(model, name, X, test_idx, scaler)
        yt, yp, sc = aggregate(vte, yte, preds, scores)
        m = metrics(yt, yp, sc)
        log(f"[result] {desc}/{name} F1={m['F1']:.3f} AUC={m['AUC']:.3f} "
            f"Acc={m['Acc']:.3f} P={m['P']:.3f} R={m['R']:.3f}")
        row = (f"{args.dataset},{desc},{name},{m['P']:.4f},{m['R']:.4f},"
               f"{m['F1']:.4f},{m['Acc']:.4f},{m['AUC']:.4f},{mode},{tag}")
        append_row(args.out, row, write_header)
        write_header = False
        log(f"[csv] appended {desc}/{name} -> {args.out}")
        del model

    del Xtr
    log(f"[done] {desc} complete.")


if __name__ == "__main__":
    main()
