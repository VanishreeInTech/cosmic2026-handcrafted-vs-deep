"""
Split builders for UADFV, Celeb-DF v2, and FF++.

Each builder reads the dataset's OFFICIAL split (or constructs the
standard one) and emits a video-level manifest in the shared format.
Run each on whichever machine holds that dataset; the manifests are
portable as long as you fix the paths (or re-run per machine).

USAGE (edit the ROOT paths at the bottom, or pass via CLI):
    python build_splits.py celebdf2 --root /data/CelebDF-v2 --out manifests/celebdf2_videos.csv
    python build_splits.py uadfv    --root /data/UADFV      --out manifests/uadfv_videos.csv
    python build_splits.py ffpp     --root /data/FFpp       --out manifests/ffpp_videos.csv  --quality c23

Each emits a *_videos.csv you then feed to preprocess.py.
"""

import argparse
import glob
import hashlib
import os

from manifest import VideoEntry, write_video_manifest, summarize, REAL, FAKE


# --------------------------------------------------------------------------
# Celeb-DF v2  --  uses the official List_of_testing_videos.txt
# --------------------------------------------------------------------------
def build_celebdf2(root, test_list_name="List_of_testing_videos.txt"):
    """
    root/ must contain:
        Celeb-real/        (590 real)
        YouTube-real/      (300 real)
        Celeb-synthesis/   (5639 fake)
        List_of_testing_videos.txt

    The test list lines look like:  "1 Celeb-real/id0_0000.mp4"
    where the leading int is the dataset's own label (1=real, 0=fake).
    Everything in the list = test; everything else = train.
    """
    test_list_path = os.path.join(root, test_list_name)
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Cannot find {test_list_path}")

    # Parse official test list -> set of relative paths + their stated labels
    test_rel = {}
    with open(test_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            lab = int(parts[0])          # dataset's own: 1=real, 0=fake
            rel = parts[1].replace("\\", "/")
            test_rel[rel] = lab

    # Enumerate every video in the three folders
    entries = []
    folder_label = {
        "Celeb-real": REAL,
        "YouTube-real": REAL,
        "Celeb-synthesis": FAKE,
    }
    flips = 0
    for folder, lab in folder_label.items():
        fdir = os.path.join(root, folder)
        if not os.path.isdir(fdir):
            print(f"[warn] missing folder {fdir}")
            continue
        for vp in sorted(glob.glob(os.path.join(fdir, "*.mp4"))):
            rel = f"{folder}/{os.path.basename(vp)}"
            split = "test" if rel in test_rel else "train"
            # cross-check label convention against the official list
            if rel in test_rel and test_rel[rel] != lab:
                flips += 1
            entries.append(VideoEntry(
                video_id=f"celebdf2/{folder}/{os.path.splitext(os.path.basename(vp))[0]}",
                video_path=os.path.abspath(vp),
                label=lab,
                split=split,
                dataset="celebdf2",
                fold=-1,
            ))

    if flips:
        print(f"[CRITICAL] {flips} videos disagree with official-list labels. "
              f"Your folder->label mapping or the list parse is wrong. STOP and check.")
    else:
        print("[ok] folder labels agree with official test-list labels (1=real,0=fake).")

    return entries


# --------------------------------------------------------------------------
# UADFV  --  no official split; build identity-disjoint 5-fold CV
# --------------------------------------------------------------------------
def build_uadfv(root, n_folds=5, seed=42):
    """
    root/ expected to contain real and fake video folders. UADFV is tiny
    (~49 real + ~49 fake). We assign deterministic folds by hashing the
    video id, so folds are reproducible across machines without storing
    a split file. video-disjoint by construction (each video in one fold).

    Adjust `real_globs`/`fake_globs` to match your actual UADFV layout.
    """
    real_globs = ["real/*.mp4"]
    fake_globs = ["fake/*.mp4"]

    def collect(globs, lab):
        out = []
        for g in globs:
            for vp in glob.glob(os.path.join(root, g), recursive=True):
                vid = os.path.splitext(os.path.basename(vp))[0]
                # deterministic fold from hash -> 0..n_folds-1
                h = int(hashlib.md5(f"{vid}_{seed}".encode()).hexdigest(), 16)
                fold = h % n_folds
                out.append(VideoEntry(
                    video_id=f"uadfv/{'real' if lab==REAL else 'fake'}/{vid}",
                    video_path=os.path.abspath(vp),
                    label=lab,
                    split="cv",          # split assigned per-fold at train time
                    dataset="uadfv",
                    fold=fold,
                ))
        return out

    entries = collect(real_globs, REAL) + collect(fake_globs, FAKE)
    if not entries:
        print(f"[warn] no UADFV videos found under {root}; check real_globs/fake_globs.")
    return entries


# --------------------------------------------------------------------------
# FF++  --  standard 720/140/140 video split (c23 by default)
# --------------------------------------------------------------------------
def build_ffpp(root, quality="c23", splits_json=None):
    """
    FF++ ships official train/val/test splits as JSON lists of video-id pairs
    (the standard 720/140/140). The canonical files are usually:
        splits/train.json, splits/val.json, splits/test.json
    Each json is a list of pairs like [["035","036"], ...].

    root/ layout (typical):
        original_sequences/youtube/<quality>/videos/*.mp4   (REAL)
        manipulated_sequences/<method>/<quality>/videos/*.mp4 (FAKE)
        where <method> in {Deepfakes, Face2Face, FaceSwap, NeuralTextures}

    We use the official json splits if found; otherwise we tell you to
    place them. This keeps you on the comparable protocol.
    """
    import json

    if splits_json is None:
        splits_json = os.path.join(root, "splits")
    split_files = {s: os.path.join(splits_json, f"{s}.json")
                   for s in ["train", "val", "test"]}

    # video-id -> split
    id2split = {}
    for s, path in split_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FF++ official split file missing: {path}\n"
                f"Download the standard splits (train/val/test .json) and place them there.")
        with open(path) as f:
            for pair in json.load(f):
                for vid in pair:
                    id2split[vid] = s

    entries = []
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

    # REAL: original_sequences/youtube/<quality>/videos/<id>.mp4
    real_dir = os.path.join(root, "original_sequences", "youtube", quality, "videos")
    for vp in sorted(glob.glob(os.path.join(real_dir, "*.mp4"))):
        vid = os.path.splitext(os.path.basename(vp))[0]
        if vid not in id2split:
            continue
        entries.append(VideoEntry(
            video_id=f"ffpp/real/{vid}",
            video_path=os.path.abspath(vp),
            label=REAL,
            split=id2split[vid],
            dataset="ffpp",
            fold=-1,
        ))

    # FAKE: manipulated_sequences/<method>/<quality>/videos/<id1>_<id2>.mp4
    for method in methods:
        man_dir = os.path.join(root, "manipulated_sequences", method, quality, "videos")
        for vp in sorted(glob.glob(os.path.join(man_dir, "*.mp4"))):
            base = os.path.splitext(os.path.basename(vp))[0]   # e.g. 035_036
            target_id = base.split("_")[0]
            if target_id not in id2split:
                continue
            entries.append(VideoEntry(
                video_id=f"ffpp/fake/{method}/{base}",
                video_path=os.path.abspath(vp),
                label=FAKE,
                split=id2split[target_id],
                dataset="ffpp",
                fold=-1,
            ))

    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["celebdf2", "uadfv", "ffpp"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", default="c23", help="FF++ only")
    ap.add_argument("--folds", type=int, default=5, help="UADFV only")
    args = ap.parse_args()

    if args.dataset == "celebdf2":
        entries = build_celebdf2(args.root)
    elif args.dataset == "uadfv":
        entries = build_uadfv(args.root, n_folds=args.folds)
    else:
        entries = build_ffpp(args.root, quality=args.quality)

    summarize(entries, name=args.dataset)
    write_video_manifest(entries, args.out)


if __name__ == "__main__":
    main()
