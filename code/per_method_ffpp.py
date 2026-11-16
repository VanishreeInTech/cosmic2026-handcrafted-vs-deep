"""
Per-manipulation-method FF++ breakdown (within-dataset test).

Trains once on the full FF++ train set (all 4 methods), then evaluates
SEPARATELY on each method's fake videos (vs the real test videos). Shows
WHICH manipulations handcrafted HOG catches vs which defeat it, and the
same for the CNN. A large-effect, informative analysis.

Method is parsed from video_id: 'ffpp/fake/<Method>/...'; real: 'ffpp/real/...'.

For the CNN we reuse saved test frame_predictions if available; here we
recompute HOG-based per-method AUC (CNN per-method uses its test scores).

This script does the HANDCRAFTED (HOG) per-method breakdown. For CNN
per-method, we read cnn/ffpp/predictions/frame_predictions.csv.

Run:
  python per_method_ffpp.py \
     --hog-dir ../features/ffpp --hog-name ffpp \
     --cnn-frame-preds ../cnn/ffpp/predictions/frame_predictions.csv \
     --out ../results/ffpp_per_method.csv --k 4
"""
import argparse, csv, os, re
from collections import defaultdict
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

RS = 42
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def method_of(vid):
    m = re.search(r"fake/([^/]+)/", vid)
    return m.group(1) if m else ("real" if "/real/" in vid else "?")


def load_hog(feat_dir, name):
    meta = np.load(os.path.join(feat_dir, f"{name}_hog_meta.npz"), allow_pickle=True)
    shape = tuple(int(x) for x in meta["shape"])
    X = np.memmap(os.path.join(feat_dir, f"{name}_hog.dat"), dtype=np.float32, mode="r", shape=shape)
    return X, meta["split"], meta["video_id"], meta["label"]


def vmean(vids, feats, labs):
    acc = defaultdict(list); lab = {}
    for v, f, y in zip(vids, feats, labs):
        acc[v].append(f); lab[v] = int(y)
    order = list(acc.keys())
    return order, np.vstack([np.mean(acc[v], 0) for v in order]).astype(np.float32), \
           np.array([lab[v] for v in order])


def subsample(vids, k, seed=42):
    rng = np.random.RandomState(seed); by = defaultdict(list)
    for i, v in enumerate(vids): by[v].append(i)
    keep = []
    for v, idx in by.items():
        keep.extend(idx if len(idx) <= k else rng.choice(idx, k, replace=False).tolist())
    return np.array(sorted(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hog-dir", required=True); ap.add_argument("--hog-name", default="ffpp")
    ap.add_argument("--cnn-frame-preds", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    X, split, vid, lab = load_hog(args.hog_dir, args.hog_name)
    tr = np.where(split == "train")[0]; te = np.where(split == "test")[0]

    # train HOG classifier on all methods
    ktr = subsample(vid[tr], args.k); seltr = tr[ktr]
    _, Xtr, ytr = vmean(vid[seltr], np.asarray(X[seltr]), lab[seltr])
    sc = StandardScaler(copy=False); Xtr = sc.fit_transform(Xtr)
    clf = CalibratedClassifierCV(
        LinearSVC(C=1.0, random_state=RS, max_iter=10000, class_weight="balanced"), cv=3)
    clf.fit(Xtr, ytr)
    ridx = list(clf.classes_).index(1)

    # test videos: aggregate to video-level
    tord, Xte, yte = vmean(vid[te], np.asarray(X[te]), lab[te])
    Xte_s = sc.transform(Xte)
    hog_score = clf.predict_proba(Xte_s)[:, ridx]
    meth = np.array([method_of(v) for v in tord])

    # optional CNN per-method (from saved frame preds -> video mean)
    cnn_score_by_vid = {}
    if args.cnn_frame_preds and os.path.exists(args.cnn_frame_preds):
        acc = defaultdict(list)
        for r in csv.DictReader(open(args.cnn_frame_preds)):
            acc[r["video_id"]].append(float(r["score_real"]))
        cnn_score_by_vid = {v: np.mean(s) for v, s in acc.items()}

    real_mask = meth == "real"
    y_real = yte[real_mask]; hog_real = hog_score[real_mask]
    rows = [["method", "n_fake_videos", "HOG_AUC", "CNN_AUC"]]
    print("\n=== FF++ per-method AUC (fake method vs real) ===")
    for M in METHODS:
        fm = meth == M
        if fm.sum() == 0:
            continue
        y = np.concatenate([y_real, yte[fm]])
        hs = np.concatenate([hog_real, hog_score[fm]])
        try: hog_auc = roc_auc_score(y, hs)
        except ValueError: hog_auc = float("nan")
        cnn_auc = float("nan")
        if cnn_score_by_vid:
            cs_real = [cnn_score_by_vid[v] for v, mm in zip(tord, real_mask) if mm and v in cnn_score_by_vid]
            cs_fake = [cnn_score_by_vid[v] for v, mm in zip(tord, fm) if mm and v in cnn_score_by_vid]
            yy = np.array([1]*len(cs_real) + [0]*len(cs_fake))
            ss = np.array(cs_real + cs_fake)
            if len(np.unique(yy)) == 2:
                cnn_auc = roc_auc_score(yy, ss)
        print(f"  {M:15s} n={int(fm.sum()):4d}  HOG_AUC={hog_auc:.3f}  CNN_AUC={cnn_auc:.3f}")
        rows.append([M, int(fm.sum()), f"{hog_auc:.4f}", f"{cnn_auc:.4f}"])

    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n[done] -> {args.out}")
    print("Interpretation: methods with high HOG_AUC have strong structural/edge")
    print("artifacts HOG catches; low HOG_AUC methods defeat handcrafted cues.")


if __name__ == "__main__":
    main()
