"""
Leakage diagnostic for the Celeb-DF CNN result (AUC=1.000 sanity check).

Checks:
  1. VIDEO overlap: any video_id in both train and test? (should be 0)
  2. IDENTITY overlap: Celeb-DF videos are named idX_... / idX_idY_...
     If the SAME identity (idX) appears in both train and test, the model can
     memorize a person's appearance -> inflated AUC. The OFFICIAL split is
     supposed to be identity-disjoint; this verifies it.
  3. Frame-path overlap: any identical frame file in both splits.
  4. Class balance per split.

Run:
  python check_leakage.py --frames ../manifests/celebdf2_frames.csv
"""
import argparse, csv, re
from collections import defaultdict


def identity_of(video_id):
    """Extract Celeb-DF identity id(s) from a video_id like
    'celebdf2/Celeb-synthesis/id0_id16_0009' or '.../Celeb-real/id0_0000'
    or '.../YouTube-real/00170'. Returns a tuple of ids found."""
    base = video_id.split("/")[-1]
    ids = re.findall(r"id\d+", base)
    if ids:
        return tuple(sorted(set(ids)))
    return (base,)  # YouTube-real numeric names: treat whole name as identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.frames)))
    train_vids, test_vids = set(), set()
    train_paths, test_paths = set(), set()
    train_ids, test_ids = set(), set()
    cls = defaultdict(lambda: defaultdict(int))

    for r in rows:
        v = r["video_id"]; sp = r["split"]
        ids = identity_of(v)
        if sp == "train":
            train_vids.add(v); train_paths.add(r["frame_path"]); train_ids.update(ids)
        elif sp == "test":
            test_vids.add(v); test_paths.add(r["frame_path"]); test_ids.update(ids)
        cls[sp][int(r["label"])] += 1

    print("=== SPLIT SIZES (videos) ===")
    print(f"  train videos: {len(train_vids)}   test videos: {len(test_vids)}")

    print("\n=== 1. VIDEO overlap (should be 0) ===")
    vo = train_vids & test_vids
    print(f"  overlapping videos: {len(vo)}")
    if vo: print("   ", list(vo)[:10])

    print("\n=== 2. IDENTITY overlap (Celeb-DF should be identity-disjoint) ===")
    io = train_ids & test_ids
    print(f"  train identities: {len(train_ids)}  test identities: {len(test_ids)}")
    print(f"  overlapping identities: {len(io)}")
    if io: print("   examples:", list(io)[:20])

    print("\n=== 3. FRAME-PATH overlap (should be 0) ===")
    po = train_paths & test_paths
    print(f"  overlapping frame files: {len(po)}")

    print("\n=== 4. CLASS BALANCE (frames) ===")
    for sp in ["train", "test"]:
        r = cls[sp].get(1, 0); f = cls[sp].get(0, 0)
        print(f"  {sp}: real(1)={r}  fake(0)={f}")

    print("\n=== VERDICT ===")
    if vo or po:
        print("  *** LEAKAGE: direct video/frame overlap. Result invalid. ***")
    elif io:
        print(f"  *** IDENTITY OVERLAP: {len(io)} identities in both splits. ***")
        print("  This CAN inflate AUC (model memorizes a person). For Celeb-DF the")
        print("  OFFICIAL split is identity-disjoint by design -- if overlap is")
        print("  large, the manifest/split build may be wrong. If overlap is small")
        print("  or zero, the high AUC is more likely genuine (CNNs do very well on")
        print("  Celeb-DF v2 synthesis artifacts).")
    else:
        print("  No video/frame/identity overlap detected. AUC=1.000 is then")
        print("  plausibly genuine: EfficientNet-B0 finds Celeb-synthesis artifacts")
        print("  easily. Still report carefully and cross-check vs literature.")


if __name__ == "__main__":
    main()
