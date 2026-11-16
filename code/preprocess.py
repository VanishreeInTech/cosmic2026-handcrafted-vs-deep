"""
Preprocessing: video manifest -> 256x256 MTCNN-aligned face crops + frame manifest.

Protocol (DeepfakeBench-aligned, with one stated deviation):
  - resolution: 256x256 master crops, PNG lossless
  - sampling: UNIFORM 32 frames per video (DeepfakeBench standard),
    selected from frames that yielded a valid aligned face
  - detector: MTCNN, largest face, with FIVE-POINT EYE ALIGNMENT
    (rotate so the inter-ocular line is horizontal before cropping).
    [DEVIATION from DeepfakeBench's DLIB 81-landmark alignment: we use
    MTCNN's two eye landmarks for in-plane rotation correction. This
    removes the head-tilt rotation confound that most affects the
    rotation-sensitive descriptors (HOG, LBP), without a second
    dependency. Stated explicitly in the paper.]
  - margin: 1.3x bounding box (DeepfakeBench convention)
  - video_id preserved on every frame for video-level aggregation
  - <32 valid frames: cyclic repetition to reach 32 (toggle with --no-pad)

Downstream:
  - descriptors read 256 directly (resize to 128 for the optional ablation)
  - EfficientNet-B0 resizes 256 -> 224 on the fly in its transform

Run per dataset / per machine:
    python preprocess.py --videos manifests/celebdf2_videos.csv \
                         --out-root crops/celebdf2 \
                         --frame-manifest manifests/celebdf2_frames.csv \
                         --size 256 --n-frames 32

Dependencies: facenet-pytorch (MTCNN), opencv-python, torch, pillow, numpy, tqdm
    pip install facenet-pytorch opencv-python torch pillow numpy tqdm --break-system-packages
"""

import argparse
import csv
import os

import cv2
import numpy as np
from PIL import Image

from manifest import read_video_manifest, FRAME_FIELDS

MARGIN_FRAC = 0.30   # 1.3x box  ->  0.30 expansion split each side


def get_mtcnn(device):
    """keep_all=True so we get boxes AND landmarks for every face; we pick
    the largest, align by its eye landmarks, then crop manually so we
    control the 1.3x margin and 256 resize explicitly."""
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, post_process=False, device=device)


def largest_idx(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return int(max(range(len(areas)), key=lambda i: areas[i]))


def align_by_eyes(frame_rgb, landmarks):
    """Rotate the whole frame so the eye line is horizontal.
    facenet-pytorch landmark order: [left_eye, right_eye, nose,
    mouth_left, mouth_right], each (x, y). Returns rotated frame and the
    affine matrix so we can also map the box if needed (we re-detect-free
    approach: rotate frame about the eye-midpoint, then re-crop by the
    rotated box estimate)."""
    left_eye, right_eye = landmarks[0], landmarks[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0,
                   (left_eye[1] + right_eye[1]) / 2.0)
    h, w = frame_rgb.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(frame_rgb, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def transform_box(box, M):
    """Apply affine M to the 4 corners of box, return new axis-aligned box."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([corners, ones]) @ M.T   # 4x2
    nx1, ny1 = pts[:, 0].min(), pts[:, 1].min()
    nx2, ny2 = pts[:, 0].max(), pts[:, 1].max()
    return [nx1, ny1, nx2, ny2]


def crop_from_box(frame_rgb, box, size):
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * (MARGIN_FRAC / 2.0), bh * (MARGIN_FRAC / 2.0)
    x1 = max(0, int(x1 - mx)); y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx)); y2 = min(h, int(y2 + my))
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # small faces get upscaled to 256; expected and standard.
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)


def detect_align_crop(frame_rgb, mtcnn, size):
    """Full per-frame path: detect -> pick largest -> eye-align ->
    crop aligned box -> 256. Returns crop (RGB) or None."""
    boxes, _, landmarks = mtcnn.detect(Image.fromarray(frame_rgb), landmarks=True)
    i = largest_idx(boxes)
    if i is None or landmarks is None:
        return None
    rotated, M = align_by_eyes(frame_rgb, landmarks[i])
    new_box = transform_box(boxes[i], M)
    return crop_from_box(rotated, new_box, size)


def uniform_indices(n_total, n_sample, pad=True):
    if n_total == 0:
        return []
    if n_total >= n_sample:
        return np.linspace(0, n_total - 1, n_sample).round().astype(int).tolist()
    if pad:
        return [i % n_total for i in range(n_sample)]
    return list(range(n_total))


def process_video(entry, mtcnn, out_root, size, n_frames, pad):
    """Two-pass: (1) detect+align+crop every readable frame, keep valid
    crops in memory; (2) uniformly select n_frames and write to disk."""
    cap = cv2.VideoCapture(entry.video_path)
    if not cap.isOpened():
        return [], f"cannot open {entry.video_path}"

    valid_crops = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        crop = detect_align_crop(frame_rgb, mtcnn, size)
        if crop is not None:
            valid_crops.append(crop)
    cap.release()

    if len(valid_crops) == 0:
        return [], f"zero valid faces from {entry.video_path}"

    sel = uniform_indices(len(valid_crops), n_frames, pad=pad)
    out_dir = os.path.join(out_root, entry.video_id)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for out_i, src_i in enumerate(sel):
        fp = os.path.join(out_dir, f"f{out_i:04d}.png")
        cv2.imwrite(fp, cv2.cvtColor(valid_crops[src_i], cv2.COLOR_RGB2BGR))
        rows.append({
            "video_id": entry.video_id,
            "frame_path": os.path.abspath(fp),
            "label": entry.label,
            "split": entry.split,
            "dataset": entry.dataset,
            "fold": entry.fold,
        })
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="input video manifest csv")
    ap.add_argument("--out-root", required=True, help="where crops are written")
    ap.add_argument("--frame-manifest", required=True, help="output frame manifest csv")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-frames", type=int, default=32, help="uniform frames per video")
    ap.add_argument("--no-pad", action="store_true",
                    help="do NOT cyclically pad videos with <n_frames valid faces")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tqdm import tqdm
    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[warn] CUDA not available, falling back to {device}")
    mtcnn = get_mtcnn(device)

    videos = read_video_manifest(args.videos)
    os.makedirs(os.path.dirname(os.path.abspath(args.frame_manifest)), exist_ok=True)

    failures = []
    n_rows = 0
    with open(args.frame_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
        w.writeheader()
        for entry in tqdm(videos, desc="videos"):
            rows, err = process_video(entry, mtcnn, args.out_root,
                                      args.size, args.n_frames, not args.no_pad)
            for r in rows:
                w.writerow(r)
            n_rows += len(rows)
            if err:
                failures.append((entry.video_id, err))

    print(f"[done] wrote {n_rows} frame rows -> {args.frame_manifest}")
    print(f"[info] {len(videos)-len(failures)} videos ok, {len(failures)} failed")
    if failures:
        fpath = args.frame_manifest + ".failures.txt"
        with open(fpath, "w") as f:
            for vid, err in failures:
                f.write(f"{vid}\t{err}\n")
        print(f"[warn] {len(failures)} videos produced no crops; logged -> {fpath}")
        print("       Decide explicitly how to handle these at eval (don't silently drop).")


if __name__ == "__main__":
    main()
