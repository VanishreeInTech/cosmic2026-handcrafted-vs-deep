"""
Refined Celeb-DF identity-leakage check.

Celeb-DF naming:
  Celeb-real:      id{P}_{vid}            -> identity = idP
  Celeb-synthesis: id{T}_id{S}_{vid}      -> TARGET = idT (face being faked),
                                             SOURCE = idS (face donor)
  YouTube-real:    {NNNNN}                -> no celeb id

The standard 'identity-disjoint' definition concerns the TARGET identity
(the person the video purports to show). Counting SOURCE ids too can create
false 'overlap'. This script reports BOTH so we know which is real.

Run: python check_leakage2.py --frames ../manifests/celebdf2_frames.csv
"""
import argparse, csv, re
from collections import defaultdict


def parse_ids(video_id):
    """Return (target_id, all_ids, kind)."""
    base = video_id.split("/")[-1]
    folder = video_id.split("/")[-2] if "/" in video_id else ""
    ids = re.findall(r"id(\d+)", base)
    if "synthesis" in folder.lower():
        # id{T}_id{S}_... -> target is FIRST id
        target = f"id{ids[0]}" if ids else None
        return target, [f"id{i}" for i in ids], "fake"
    elif "real" in folder.lower() and ids:
        return f"id{ids[0]}", [f"id{ids[0]}"], "real"
    else:
        return base, [base], "youtube"  # numeric, no celeb id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.frames)))

    # dedup to video level
    seen = {}
    for r in rows:
        seen[r["video_id"]] = r["split"]

    tgt = {"train": set(), "test": set()}
    allids = {"train": set(), "test": set()}
    for vid, sp in seen.items():
        if sp not in ("train", "test"):
            continue
        t, a, kind = parse_ids(vid)
        if t: tgt[sp].add(t)
        allids[sp].update(a)

    print("=== TARGET-identity overlap (the standard definition) ===")
    to = tgt["train"] & tgt["test"]
    print(f"  train target-ids={len(tgt['train'])} test target-ids={len(tgt['test'])}")
    print(f"  OVERLAP (target): {len(to)}")
    if to: print("   ", sorted(to)[:20])

    print("\n=== ALL-identity overlap (target+source, stricter) ===")
    ao = allids["train"] & allids["test"]
    print(f"  OVERLAP (any id): {len(ao)}")

    print("\n=== INTERPRETATION ===")
    if len(to) == 0:
        print("  Target identities ARE disjoint -> official split is identity-")
        print("  disjoint by the standard definition. Earlier overlap was from")
        print("  counting SOURCE ids in synthesis videos. Your split is CORRECT.")
    else:
        print(f"  {len(to)} target identities shared -> genuine identity overlap.")
        print("  This would need addressing.")


if __name__ == "__main__":
    main()
