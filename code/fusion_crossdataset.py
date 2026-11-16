"""
Cross-dataset (FF++ -> Celeb-DF) three-way comparison:
  1. CNN alone   : FF++ model's P(real) on Celeb-DF test  -> video AUC
  2. HOG alone   : classifier trained on FF++ HOG, tested on Celeb-DF HOG
  3. Fused       : classifier on [CNN emb (+) HOG], FF++-train -> Celeb-DF test

All video-level (mean frame score -> AUC; majority vote -> F1). The question:
does fusion beat CNN-alone on cross-dataset generalization?

Inputs:
  FF++ train:
    - CNN embeddings: cnn/ffpp/embeddings/emb_train.npz (X, video_id, label)
    - HOG features:   features/ffpp/ffpp_hog.dat + _meta.npz  (split==train rows)
  Celeb-DF test:
    - FF++-model embeddings: crossdataset/celebdf_test_ffppmodel_emb.npz
    - FF++-model scores:     crossdataset/celebdf_test_ffppmodel_scores.npz
    - HOG features:          features/celebdf2/celebdf2_hog.dat + _meta.npz (split==test)

Frame alignment: CNN emb and HOG for the SAME dataset come from the SAME
frame manifest order, so row i corresponds to the same frame. We verify by
video_id + within-video frame index. To keep it robust we aggregate each
modality to VIDEO level first (mean over frames), then fuse at video level.
This avoids frame-alignment pitfalls and is a valid late-fusion design.

Run:
  python fusion_crossdataset.py \
     --ffpp-emb ../cnn/ffpp/embeddings/emb_train.npz \
     --ffpp-hog-dir ../features/ffpp --ffpp-hog-name ffpp \
     --celeb-emb ../crossdataset/celebdf_test_ffppmodel_emb.npz \
     --celeb-scores ../crossdataset/celebdf_test_ffppmodel_scores.npz \
     --celeb-hog-dir ../features/celebdf2 --celeb-hog-name celebdf2 \
     --out ../results/crossdataset_fusion.csv \
     --train-frames-per-video 4
"""
import argparse, os
from collections import defaultdict
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV

RS = 42


def load_hog_split(feat_dir, name, want_split):
    meta = np.load(os.path.join(feat_dir, f"{name}_hog_meta.npz"), allow_pickle=True)
    shape = tuple(int(x) for x in meta["shape"])
    X = np.memmap(os.path.join(feat_dir, f"{name}_hog.dat"), dtype=np.float32,
                  mode="r", shape=shape)
    split = meta["split"]; label = meta["label"]; vid = meta["video_id"]
    idx = np.where(split == want_split)[0]
    return X, idx, vid, label


def video_mean(vids, feats, labels):
    """Aggregate frame features to per-video mean vectors."""
    acc = defaultdict(list); lab = {}
    for v, f, y in zip(vids, feats, labels):
        acc[v].append(f); lab[v] = int(y)
    order = list(acc.keys())
    Xv = np.vstack([np.mean(acc[v], axis=0) for v in order]).astype(np.float32)
    yv = np.array([lab[v] for v in order], dtype=np.int64)
    return order, Xv, yv


def subsample_by_video(vids, k, seed=42):
    rng = np.random.RandomState(seed)
    by = defaultdict(list)
    for i, v in enumerate(vids):
        by[v].append(i)
    keep = []
    for v, idxs in by.items():
        keep.extend(idxs if len(idxs) <= k else
                    rng.choice(idxs, size=k, replace=False).tolist())
    return np.array(sorted(keep))


def metrics(yt, yp, sc):
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro",
                                                  zero_division=0)
    acc = accuracy_score(yt, yp)
    try: auc = roc_auc_score(yt, sc)
    except ValueError: auc = float("nan")
    return dict(P=p, R=r, F1=f1, Acc=acc, AUC=auc)


def fit_eval(Xtr, ytr, Xte, yte, scale=True):
    """LinearSVC with calibrated probabilities for AUC."""
    if scale:
        sc = StandardScaler(copy=False)
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    base = LinearSVC(C=1.0, random_state=RS, max_iter=10000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
    yp = (proba >= 0.5).astype(int)
    return metrics(yte, yp, proba)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffpp-emb", required=True)
    ap.add_argument("--ffpp-hog-dir", required=True)
    ap.add_argument("--ffpp-hog-name", default="ffpp")
    ap.add_argument("--celeb-emb", required=True)
    ap.add_argument("--celeb-scores", required=True)
    ap.add_argument("--celeb-hog-dir", required=True)
    ap.add_argument("--celeb-hog-name", default="celebdf2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-frames-per-video", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows_out = ["experiment,P,R,F1,Acc,AUC"]

    # ---------- 1. CNN ALONE (cross-dataset) ----------
    cs = np.load(args.celeb_scores, allow_pickle=True)
    _, _, _ = None, None, None
    order, sc_by_vid, lab_by_vid = [], {}, {}
    acc = defaultdict(list); lab = {}
    for v, y, s in zip(cs["video_id"], cs["label"], cs["frame_score"]):
        acc[v].append(float(s)); lab[v] = int(y)
    vids = list(acc.keys())
    yt = np.array([lab[v] for v in vids])
    sc = np.array([np.mean(acc[v]) for v in vids])
    yp = (sc >= 0.5).astype(int)
    m_cnn = metrics(yt, yp, sc)
    print(f"[CNN alone]   AUC={m_cnn['AUC']:.3f} F1={m_cnn['F1']:.3f}", flush=True)
    rows_out.append(f"CNN_alone,{m_cnn['P']:.4f},{m_cnn['R']:.4f},{m_cnn['F1']:.4f},"
                    f"{m_cnn['Acc']:.4f},{m_cnn['AUC']:.4f}")

    # ---------- prep FF++ train (emb + hog) at VIDEO level ----------
    fe = np.load(args.ffpp_emb, allow_pickle=True)
    # subsample frames per video for tractability, then video-mean
    keep = subsample_by_video(fe["video_id"], args.train_frames_per_video, RS)
    ford, fEmb, fy = video_mean(fe["video_id"][keep], fe["X"][keep], fe["label"][keep])

    Xh, hidx, hvid, hlab = load_hog_split(args.ffpp_hog_dir, args.ffpp_hog_name, "train")
    keep_h = subsample_by_video(hvid[hidx], args.train_frames_per_video, RS)
    sel = hidx[keep_h]
    hord, fHog, fyh = video_mean(hvid[sel], np.asarray(Xh[sel]), hlab[sel])

    # align FF++ emb-videos and hog-videos (intersection, same order)
    common = [v for v in ford if v in set(hord)]
    emb_map = {v: fEmb[i] for i, v in enumerate(ford)}
    hog_map = {v: fHog[i] for i, v in enumerate(hord)}
    lab_map = {v: fy[i] for i, v in enumerate(ford)}
    fEmbA = np.vstack([emb_map[v] for v in common])
    fHogA = np.vstack([hog_map[v] for v in common])
    fyA = np.array([lab_map[v] for v in common])
    print(f"[ffpp train] videos aligned: {len(common)} "
          f"emb={fEmbA.shape} hog={fHogA.shape}", flush=True)

    # ---------- prep Celeb-DF test (emb + hog) at VIDEO level ----------
    ce = np.load(args.celeb_emb, allow_pickle=True)
    cord, cEmb, cy = video_mean(ce["video_id"], ce["X"], ce["label"])
    Xch, chidx, chvid, chlab = load_hog_split(args.celeb_hog_dir, args.celeb_hog_name, "test")
    chord, cHog, cyh = video_mean(chvid[chidx], np.asarray(Xch[chidx]), chlab[chidx])
    commonc = [v for v in cord if v in set(chord)]
    cemb_map = {v: cEmb[i] for i, v in enumerate(cord)}
    chog_map = {v: cHog[i] for i, v in enumerate(chord)}
    clab_map = {v: cy[i] for i, v in enumerate(cord)}
    cEmbA = np.vstack([cemb_map[v] for v in commonc])
    cHogA = np.vstack([chog_map[v] for v in commonc])
    cyA = np.array([clab_map[v] for v in commonc])
    print(f"[celeb test] videos aligned: {len(commonc)} "
          f"emb={cEmbA.shape} hog={cHogA.shape}", flush=True)

    # ---------- 2. HOG ALONE (cross-dataset) ----------
    m_hog = fit_eval(fHogA, fyA, cHogA, cyA)
    print(f"[HOG alone]   AUC={m_hog['AUC']:.3f} F1={m_hog['F1']:.3f}", flush=True)
    rows_out.append(f"HOG_alone,{m_hog['P']:.4f},{m_hog['R']:.4f},{m_hog['F1']:.4f},"
                    f"{m_hog['Acc']:.4f},{m_hog['AUC']:.4f}")

    # ---------- 3. FUSED (emb (+) hog) ----------
    fFus = np.hstack([fEmbA, fHogA]); cFus = np.hstack([cEmbA, cHogA])
    m_fus = fit_eval(fFus, fyA, cFus, cyA)
    print(f"[FUSED]       AUC={m_fus['AUC']:.3f} F1={m_fus['F1']:.3f}", flush=True)
    rows_out.append(f"Fused_CNN+HOG,{m_fus['P']:.4f},{m_fus['R']:.4f},{m_fus['F1']:.4f},"
                    f"{m_fus['Acc']:.4f},{m_fus['AUC']:.4f}")

    with open(args.out, "w") as f:
        f.write("\n".join(rows_out) + "\n")
    print(f"\n[done] -> {args.out}", flush=True)
    print("\n=== SUMMARY (cross-dataset FF++ -> Celeb-DF) ===")
    print(f"  CNN alone : AUC={m_cnn['AUC']:.3f}")
    print(f"  HOG alone : AUC={m_hog['AUC']:.3f}")
    print(f"  Fused     : AUC={m_fus['AUC']:.3f}")


if __name__ == "__main__":
    main()
