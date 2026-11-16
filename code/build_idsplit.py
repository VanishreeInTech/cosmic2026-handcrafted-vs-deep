"""
Build an IDENTITY-DISJOINT Celeb-DF v2 split (custom, by target identity).

Why: the official List_of_testing_videos.txt shares ~54 of 59 target
identities between train and test (Celeb-DF has only 59 subjects). A strong
CNN then memorizes identities -> AUC ~1.000 from epoch 1. This split assigns
each of the 59 target identities ENTIRELY to one of {train,val,test}, so no
person appears in two splits. Numbers become honest (val is a real
generalization check). NOTE: not comparable to official-split literature;
use alongside the official results, not as a replacement.

Assignment:
  - identity = TARGET id (first idN in filename). Synthesis id{T}_id{S}_*
    is assigned by T. YouTube-real numeric names -> their own pseudo-id.
  - 70/15/15 of the 59 identities -> train/val/test (seeded, reproducible).
  - every video routed by its target identity's group.

Output: a video manifest in the SAME shared format (manifest.py), so all
downstream code (preprocess already done; we reuse existing crops) works.
Since crops already exist under crops/celebdf2/<video_id>/, we just emit a
new FRAME manifest by RE-LABELING splits on the existing frames manifest.

Run (reuse existing frames, just relabel splits):
  python build_idsplit.py \
     --frames ../manifests/celebdf2_frames.csv \
     --out ../manifests/celebdf2_idsplit_frames.csv

This avoids re-preprocessing: it reads the existing frame manifest and
rewrites the 'split' column by identity group. fold stays -1.
"""
import argparse, csv, re
import numpy as np
from collections import defaultdict


def target_identity(video_id):
    base = video_id.split("/")[-1]
    folder = video_id.split("/")[-2] if "/" in video_id else ""
    ids = re.findall(r"id(\d+)", base)
    if ids:
        return f"id{ids[0]}"        # target = first id (works for real & synth)
    return f"yt_{base}"             # YouTube-real numeric -> pseudo identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="existing celebdf2_frames.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.frames)))
    fields = rows[0].keys()

    # collect identities and which are real-capable vs fake
    ident_videos = defaultdict(set)
    for r in rows:
        ident_videos[target_identity(r["video_id"])].add(r["video_id"])
    idents = sorted(ident_videos.keys())
    print(f"[info] total target identities: {len(idents)}")

    rng = np.random.RandomState(args.seed)
    rng.shuffle(idents)
    n = len(idents)
    n_tr = int(n * args.train)
    n_va = int(n * args.val)
    train_ids = set(idents[:n_tr])
    val_ids = set(idents[n_tr:n_tr + n_va])
    test_ids = set(idents[n_tr + n_va:])
    print(f"[info] identities -> train={len(train_ids)} val={len(val_ids)} "
          f"test={len(test_ids)}")

    def grp(vid):
        i = target_identity(vid)
        if i in train_ids: return "train"
        if i in val_ids: return "val"
        return "test"

    # rewrite split column
    out_rows = []
    cls = defaultdict(lambda: defaultdict(int))
    vids_per_split = defaultdict(set)
    for r in rows:
        r2 = dict(r)
        r2["split"] = grp(r["video_id"])
        out_rows.append(r2)
        cls[r2["split"]][int(r["label"])] += 1
        vids_per_split[r2["split"]].add(r["video_id"])

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader(); w.writerows(out_rows)

    print(f"\n[manifest] wrote {len(out_rows)} frame rows -> {args.out}")
    print("\n=== split summary (videos / frames) ===")
    for sp in ["train", "val", "test"]:
        rl = cls[sp].get(1, 0); fk = cls[sp].get(0, 0)
        print(f"  {sp:5s}: videos={len(vids_per_split[sp]):4d}  "
              f"real_frames={rl:6d}  fake_frames={fk:6d}")

    # verify disjointness
    def ids_in(sp):
        return {target_identity(v) for v in vids_per_split[sp]}
    ti, vi, tei = ids_in("train"), ids_in("val"), ids_in("test")
    print("\n=== disjointness check (should all be 0) ===")
    print(f"  train\u2229val={len(ti & vi)}  train\u2229test={len(ti & tei)}  "
          f"val\u2229test={len(vi & tei)}")


if __name__ == "__main__":
    main()
