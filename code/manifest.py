"""
Shared manifest schema for all datasets (UADFV, Celeb-DF v2, FF++).

The whole point of this file: every dataset's split-builder emits the SAME
CSV manifest format, so all downstream code (descriptor extraction, CNN
training, evaluation) is dataset-agnostic and reads manifests only.

Manifest columns (one row per VIDEO):
    video_id   : unique string id, stable across runs (used for aggregation)
    video_path : absolute path to the source .mp4
    label      : int, 1 = REAL, 0 = FAKE   (consistent across ALL datasets)
    split      : one of {train, val, test}
    dataset    : one of {uadfv, celebdf2, ffpp}
    fold       : int, CV fold index, or -1 if not used (UADFV uses 0..4)

A second manifest is emitted per video AFTER preprocessing (frames):
    frame manifest columns:
    video_id, frame_path, label, split, dataset, fold

LABEL CONVENTION (READ THIS):
    1 = REAL, 0 = FAKE  --- everywhere, all datasets.
    Celeb-DF's own List_of_testing_videos.txt also uses 1=real, 0=fake,
    so it maps directly. FF++ real=original, fake=manipulated. UADFV
    real=real, fake=synthesized. We normalize all to 1=real/0=fake here.
"""

import csv
import os
from dataclasses import dataclass, asdict, fields

REAL = 1
FAKE = 0

VIDEO_FIELDS = ["video_id", "video_path", "label", "split", "dataset", "fold"]
FRAME_FIELDS = ["video_id", "frame_path", "label", "split", "dataset", "fold"]


@dataclass
class VideoEntry:
    video_id: str
    video_path: str
    label: int
    split: str          # train | val | test
    dataset: str        # uadfv | celebdf2 | ffpp
    fold: int = -1


def write_video_manifest(entries, out_csv):
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_FIELDS)
        w.writeheader()
        for e in entries:
            w.writerow(asdict(e))
    print(f"[manifest] wrote {len(entries)} videos -> {out_csv}")


def read_video_manifest(in_csv):
    out = []
    with open(in_csv) as f:
        for row in csv.DictReader(f):
            out.append(VideoEntry(
                video_id=row["video_id"],
                video_path=row["video_path"],
                label=int(row["label"]),
                split=row["split"],
                dataset=row["dataset"],
                fold=int(row["fold"]),
            ))
    return out


def summarize(entries, name=""):
    """Print a sanity table: counts per split x label. Catches label flips."""
    from collections import Counter
    c = Counter((e.split, "REAL" if e.label == REAL else "FAKE") for e in entries)
    print(f"\n=== manifest summary: {name} ===")
    splits = sorted({e.split for e in entries})
    for s in splits:
        r = c.get((s, "REAL"), 0)
        k = c.get((s, "FAKE"), 0)
        print(f"  {s:6s}  real={r:5d}  fake={k:5d}  total={r+k:5d}")
    print(f"  TOTAL videos: {len(entries)}")
    print("=" * 40)
