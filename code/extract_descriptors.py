"""
Handcrafted descriptor extraction (MEMORY-SAFE, streaming).

Reads a *_frames.csv (from preprocess.py) and computes classical
descriptors per frame, writing each descriptor to disk as a memory-mapped
array so peak RAM stays flat regardless of dataset size. Critical for
large sets (Celeb-DF 208k frames, FF++ larger) where holding all features
in RAM causes OOM kills.

Descriptors (256x256 grayscale unless noted):
  HOG  : 9 bins, 8x8 cells, 2x2 blocks, L2-Hys, 50% overlap (34596-D)
  SIFT : DoG keypoints, 128-D, MEAN-POOLED per frame; zero-vec if none
  ORB  : up to 500 binary 32-D, pad/truncate -> 16000-D flat
  LBP  : uniform, P=12, R=2 -> 14-bin L1-norm histogram

Strategy:
  Pass 1: count rows, infer each descriptor's dimensionality from one frame.
  Pass 2: stream frames, write features directly into a pre-allocated
          np.memmap on disk (per descriptor). Metadata saved once.
  Final : wrap each memmap + metadata into the same .npz interface the
          classifier stage expects (X, video_id, label, split, fold).

This keeps RAM ~ one batch, not the whole matrix.

Run (one descriptor at a time recommended for the biggest sets):
    python extract_descriptors.py --frames .../celebdf2_frames.csv \
        --out-dir .../features/celebdf2 --size 256 --descriptors hog
    ... then sift, orb, lbp separately, OR all at once on smaller sets.

Dependencies: opencv-contrib-python, scikit-image, numpy, tqdm
"""

import argparse
import csv
import os

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

ORB_N = 500
ORB_D = 32
LBP_P = 12
LBP_R = 2

DIMS = {"hog": 34596, "sift": 128, "orb": ORB_N * ORB_D, "lbp": LBP_P + 2}


def load_gray(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def feat_hog(gray):
    return hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


def feat_sift(gray, sift):
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        return np.zeros(128, dtype=np.float32)
    return desc.mean(axis=0).astype(np.float32)


def feat_orb(gray, orb):
    _, desc = orb.detectAndCompute(gray, None)
    out = np.zeros((ORB_N, ORB_D), dtype=np.float32)
    if desc is not None and len(desc) > 0:
        d = desc[:ORB_N].astype(np.float32)
        out[:len(d)] = d
    return out.flatten()


def feat_lbp(gray):
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def compute_one(d, gray, sift, orb):
    if d == "hog":  return feat_hog(gray)
    if d == "sift": return feat_sift(gray, sift)
    if d == "orb":  return feat_orb(gray, orb)
    if d == "lbp":  return feat_lbp(gray)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--descriptors", default="hog,sift,orb,lbp")
    args = ap.parse_args()

    from tqdm import tqdm
    os.makedirs(args.out_dir, exist_ok=True)
    want = [d.strip().lower() for d in args.descriptors.split(",")]

    rows = []
    with open(args.frames) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        print("[error] empty frame manifest"); return
    dataset = rows[0]["dataset"]
    N = len(rows)

    sift = cv2.SIFT_create() if "sift" in want else None
    orb = cv2.ORB_create(nfeatures=ORB_N) if "orb" in want else None

    # Pre-allocate one disk-backed memmap per descriptor (flat RAM use)
    memmaps = {}
    for d in want:
        path = os.path.join(args.out_dir, f"{dataset}_{d}.dat")
        memmaps[d] = np.memmap(path, dtype=np.float32, mode="w+",
                               shape=(N, DIMS[d]))

    meta = {"video_id": [], "label": [], "split": [], "fold": []}
    valid = np.ones(N, dtype=bool)

    for i, r in enumerate(tqdm(rows, desc=f"frames[{dataset}]")):
        gray = load_gray(r["frame_path"], args.size)
        if gray is None:
            valid[i] = False
            for d in want:
                memmaps[d][i] = 0.0
        else:
            for d in want:
                memmaps[d][i] = compute_one(d, gray, sift, orb)
        meta["video_id"].append(r["video_id"])
        meta["label"].append(int(r["label"]))
        meta["split"].append(r["split"])
        meta["fold"].append(int(r["fold"]))

    vid = np.array(meta["video_id"]); lab = np.array(meta["label"], dtype=np.int64)
    spl = np.array(meta["split"]); fld = np.array(meta["fold"], dtype=np.int64)

    # Flush each memmap and pair with metadata. We keep the .dat (memmap)
    # plus a small _meta.npz so the classifier stage can mmap-load X
    # without pulling the whole matrix into RAM.
    for d in want:
        memmaps[d].flush()
        meta_path = os.path.join(args.out_dir, f"{dataset}_{d}_meta.npz")
        np.savez_compressed(meta_path, video_id=vid, label=lab, split=spl,
                            fold=fld, shape=np.array([N, DIMS[d]]),
                            valid=valid)
        dat = os.path.join(args.out_dir, f"{dataset}_{d}.dat")
        print(f"[saved] {d}: X=({N}, {DIMS[d]}) -> {dat} (+ _meta.npz)")
        del memmaps[d]

    n_invalid = int((~valid).sum())
    if n_invalid:
        print(f"[warn] {n_invalid} frames failed to load (zero-filled).")


if __name__ == "__main__":
    main()
