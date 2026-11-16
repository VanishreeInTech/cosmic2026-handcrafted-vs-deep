"""
Classifier training + video-level evaluation (handcrafted descriptors).

Loads memmapped descriptor features, trains SVM(linear)/RF/XGBoost, then
aggregates frame predictions to VIDEO level:
  - label  : majority vote over frames (ties -> REAL=1), matches prior work
  - AUC    : mean per-frame score over frames (continuous), video-level

Metrics (video-level): macro-Precision/Recall/F1, Accuracy, ROC-AUC.

Modes (auto-detected from metadata):
  - SPLIT mode  (Celeb-DF, FF++): uses split column train/test.
  - CV mode     (UADFV): 5-fold CV over the fold column; reports mean +/- std.

Loading is memory-mapped: the big X (.dat) is never fully copied into RAM
for tree models we subsample frames for training if too large (configurable).

Run:
    python evaluate.py --feat-dir .../features/celebdf2 --dataset celebdf2 \
        --descriptors hog,sift,orb,lbp --out .../results/celebdf2_results.csv
    python evaluate.py --feat-dir .../features/uadfv --dataset uadfv \
        --descriptors hog,sift,orb,lbp --out .../results/uadfv_results.csv

Dependencies: scikit-learn, xgboost, numpy
    pip install scikit-learn xgboost numpy --break-system-packages
"""

import argparse
import os
import warnings
from collections import defaultdict

from sklearn.exceptions import ConvergenceWarning
# ORB's 16000-D padded features are ill-conditioned for a linear margin and
# do not fully converge at any reasonable max_iter; the result is stable, so
# we silence the cosmetic warning (documented as a finding in the paper).
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_fscore_support,
                             accuracy_score, roc_auc_score)
import xgboost as xgb

RANDOM_STATE = 42


def load_feature(feat_dir, dataset, desc):
    meta = np.load(os.path.join(feat_dir, f"{dataset}_{desc}_meta.npz"),
                   allow_pickle=True)
    shape = tuple(int(x) for x in meta["shape"])
    X = np.memmap(os.path.join(feat_dir, f"{dataset}_{desc}.dat"),
                  dtype=np.float32, mode="r", shape=shape)
    return X, meta


def make_models():
    return {
        "SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=10000),
        "RF": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                     random_state=RANDOM_STATE,
                                     class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                     learning_rate=0.1, n_jobs=-1,
                                     random_state=RANDOM_STATE,
                                     eval_metric="logloss"),
    }


def frame_scores(model, name, Xte):
    """Return per-frame continuous score for class=1 (real)."""
    if name == "SVM":
        s = model.decision_function(Xte)
        return s  # higher = more 'real'
    proba = model.predict_proba(Xte)
    # class index for label 1
    idx = list(model.classes_).index(1)
    return proba[:, idx]


def aggregate(video_ids, y_true_frame, frame_pred, frame_score):
    """Video-level: majority vote label + mean score; return per-video
    label-truth, predicted-label, mean-score."""
    by_vid = defaultdict(lambda: {"yt": None, "pred": [], "score": []})
    for vid, yt, p, s in zip(video_ids, y_true_frame, frame_pred, frame_score):
        by_vid[vid]["yt"] = yt
        by_vid[vid]["pred"].append(p)
        by_vid[vid]["score"].append(s)
    vids = list(by_vid.keys())
    yt = np.array([by_vid[v]["yt"] for v in vids])
    # majority vote; tie -> REAL=1
    yp = []
    for v in vids:
        preds = by_vid[v]["pred"]
        n1 = sum(1 for x in preds if x == 1)
        n0 = len(preds) - n1
        yp.append(1 if n1 >= n0 else 0)
    yp = np.array(yp)
    sc = np.array([np.mean(by_vid[v]["score"]) for v in vids])
    return yt, yp, sc


def metrics(yt, yp, sc):
    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0)
    acc = accuracy_score(yt, yp)
    try:
        auc = roc_auc_score(yt, sc)
    except ValueError:
        auc = float("nan")  # single-class edge
    return dict(P=p, R=r, F1=f1, Acc=acc, AUC=auc)


def train_eval(Xtr, ytr, vtr, Xte, yte, vte, name, scale):
    model = make_models()[name]
    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
    model.fit(Xtr, ytr)
    fp = model.predict(Xte)
    fs = frame_scores(model, name, Xte)
    yt, yp, sc = aggregate(vte, yte, fp, fs)
    return metrics(yt, yp, sc)


def subsample_train(mask, vid, fold_or_split_arr, video_ids_all, k_per_video, seed=42):
    """Return a boolean mask selecting at most k_per_video TRUE frames per
    video within `mask`. Reduces the train matrix size (memory) with
    negligible accuracy impact (frames per video are highly redundant).
    k_per_video<=0 disables subsampling (use all)."""
    if k_per_video is None or k_per_video <= 0:
        return mask
    rng = np.random.RandomState(seed)
    new_mask = np.zeros_like(mask)
    idx_by_vid = defaultdict(list)
    for i in np.where(mask)[0]:
        idx_by_vid[video_ids_all[i]].append(i)
    for v, idxs in idx_by_vid.items():
        if len(idxs) <= k_per_video:
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=k_per_video, replace=False)
        new_mask[list(chosen)] = True
    return new_mask


def run_split(X, meta, desc, models, scale, k_train=0):
    split = meta["split"]; label = meta["label"]; vid = meta["video_id"]
    tr = split == "train"; te = split == "test"
    tr = subsample_train(tr, vid, split, vid, k_train)
    Xtr = np.asarray(X[tr]); Xte = np.asarray(X[te])
    out = {}
    for name in models:
        out[name] = train_eval(Xtr, label[tr], vid[tr],
                               Xte, label[te], vid[te], name, scale)
    return out


def run_cv(X, meta, desc, models, scale, n_folds=5, k_train=0):
    fold = meta["fold"]; label = meta["label"]; vid = meta["video_id"]
    out = {name: [] for name in models}
    for k in range(n_folds):
        te = fold == k; tr = ~te
        tr = subsample_train(tr, vid, fold, vid, k_train)
        Xtr = np.asarray(X[tr]); Xte = np.asarray(X[te])
        for name in models:
            out[name].append(train_eval(Xtr, label[tr], vid[tr],
                                        Xte, label[te], vid[te], name, scale))
    # mean +/- std across folds
    agg = {}
    for name in models:
        agg[name] = {}
        for m in ["P", "R", "F1", "Acc", "AUC"]:
            vals = [d[m] for d in out[name]]
            agg[name][m] = (np.nanmean(vals), np.nanstd(vals))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--descriptors", default="hog,sift,orb,lbp")
    ap.add_argument("--models", default="SVM,RF,XGBoost")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-scale", action="store_true",
                    help="disable feature standardization (default: scale)")
    ap.add_argument("--train-frames-per-video", type=int, default=0,
                    help="subsample N train frames/video to save RAM "
                         "(0=use all; try 8 for big descriptors on low-RAM machines)")
    args = ap.parse_args()

    descs = [d.strip().lower() for d in args.descriptors.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    scale = not args.no_scale

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lines = ["dataset,descriptor,classifier,P,R,F1,Acc,AUC,mode"]

    for desc in descs:
        X, meta = load_feature(args.feat_dir, args.dataset, desc)
        is_cv = (meta["split"][0] == "cv")
        mode = "5foldCV" if is_cv else "split"
        print(f"\n=== {args.dataset} / {desc} ({mode}) X={X.shape} ===")

        if is_cv:
            res = run_cv(X, meta, desc, models, scale,
                         k_train=args.train_frames_per_video)
            for name in models:
                r = res[name]
                print(f"  {name:8s} F1={r['F1'][0]:.3f}+/-{r['F1'][1]:.3f} "
                      f"AUC={r['AUC'][0]:.3f} Acc={r['Acc'][0]:.3f}")
                lines.append(f"{args.dataset},{desc},{name},"
                             f"{r['P'][0]:.4f},{r['R'][0]:.4f},{r['F1'][0]:.4f},"
                             f"{r['Acc'][0]:.4f},{r['AUC'][0]:.4f},{mode}")
        else:
            res = run_split(X, meta, desc, models, scale,
                            k_train=args.train_frames_per_video)
            for name in models:
                r = res[name]
                print(f"  {name:8s} F1={r['F1']:.3f} AUC={r['AUC']:.3f} "
                      f"Acc={r['Acc']:.3f} P={r['P']:.3f} R={r['R']:.3f}")
                lines.append(f"{args.dataset},{desc},{name},"
                             f"{r['P']:.4f},{r['R']:.4f},{r['F1']:.4f},"
                             f"{r['Acc']:.4f},{r['AUC']:.4f},{mode}")
        del X

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[done] results -> {args.out}")


if __name__ == "__main__":
    main()
