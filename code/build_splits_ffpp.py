"""
FF++ split builder adapted to the user's actual folder layout:
  <root>/original/*.mp4                      (real, e.g. 033.mp4)
  <root>/Deepfakes/*.mp4                     (fake, e.g. 033_097.mp4)
  <root>/Face2Face/*.mp4
  <root>/FaceSwap/*.mp4
  <root>/NeuralTextures/*.mp4
  <root>/splits/{train,val,test}.json        (official pairs)

Standard 4 manipulation methods (FaceShifter + DeepFakeDetection excluded).

Split assignment by official json:
  - each json is a list of pairs [["033","097"], ...]; every id in a pair
    belongs to that split.
  - real 033.mp4 -> split of id 033
  - fake 033_097.mp4 -> TARGET id is 033 -> split of id 033
    (FF++ convention: <target>_<source>.mp4)

Emits shared-format video manifest (label 1=real, 0=fake).

Run:
  python build_splits_ffpp.py \
     --root ../Dataset/FaceForensics++_C23 \
     --out ../manifests/ffpp_videos.csv
"""
import argparse, csv, glob, json, os
from dataclasses import dataclass, asdict

REAL, FAKE = 1, 0
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
VIDEO_FIELDS = ["video_id", "video_path", "label", "split", "dataset", "fold"]


@dataclass
class VideoEntry:
    video_id: str
    video_path: str
    label: int
    split: str
    dataset: str
    fold: int = -1


def load_id_split(splits_dir):
    id2split = {}
    for sp in ["train", "val", "test"]:
        path = os.path.join(splits_dir, f"{sp}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}")
        for pair in json.load(open(path)):
            for vid in pair:
                id2split[vid] = sp
    return id2split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits-dir", default=None,
                    help="defaults to <root>/splits")
    args = ap.parse_args()

    splits_dir = args.splits_dir or os.path.join(args.root, "splits")
    id2split = load_id_split(splits_dir)
    print(f"[info] loaded id->split for {len(id2split)} identities")

    entries = []
    miss = 0

    # REAL: original/<id>.mp4
    real_dir = os.path.join(args.root, "original")
    for vp in sorted(glob.glob(os.path.join(real_dir, "*.mp4"))):
        vid = os.path.splitext(os.path.basename(vp))[0]   # e.g. 033
        if vid not in id2split:
            miss += 1; continue
        entries.append(VideoEntry(
            video_id=f"ffpp/real/{vid}",
            video_path=os.path.abspath(vp),
            label=REAL, split=id2split[vid], dataset="ffpp"))

    # FAKE: <method>/<target>_<source>.mp4
    for method in METHODS:
        mdir = os.path.join(args.root, method)
        if not os.path.isdir(mdir):
            print(f"[warn] missing method folder {mdir}")
            continue
        for vp in sorted(glob.glob(os.path.join(mdir, "*.mp4"))):
            base = os.path.splitext(os.path.basename(vp))[0]   # 033_097
            target = base.split("_")[0]
            if target not in id2split:
                miss += 1; continue
            entries.append(VideoEntry(
                video_id=f"ffpp/fake/{method}/{base}",
                video_path=os.path.abspath(vp),
                label=FAKE, split=id2split[target], dataset="ffpp"))

    # summary
    from collections import Counter
    c = Counter((e.split, "real" if e.label == REAL else "fake") for e in entries)
    print("\n=== FF++ manifest summary (videos) ===")
    for sp in ["train", "val", "test"]:
        r = c.get((sp, "real"), 0); f = c.get((sp, "fake"), 0)
        print(f"  {sp:5s}: real={r:4d} fake={f:5d} total={r+f:5d}")
    print(f"  TOTAL: {len(entries)} videos  (unmapped skipped: {miss})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VIDEO_FIELDS)
        w.writeheader()
        for e in entries:
            w.writerow(asdict(e))
    print(f"\n[manifest] wrote -> {args.out}")


if __name__ == "__main__":
    main()
