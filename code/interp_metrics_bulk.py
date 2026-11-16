"""
interp_metrics_bulk.py  -- run LOCALLY. Metrics only, no figures, large n.

Reuses the SAME metric definitions as gradcam_analysis.py so the results are
directly consistent with the ex00-ex07 example numbers, just over many more
test faces (robust statistics instead of n=8).

Run:
  python interp_metrics_bulk.py \
    --model ../cnn/ffpp/model/b0_best.pt \
    --frames ../manifests/ffpp_frames.csv \
    --out ../interpretability/ffpp_bulk --n 400 --seed 42

Deps (same as gradcam_analysis.py):
  pip install grad-cam timm scikit-image opencv-python torch pillow numpy --break-system-packages
"""
import argparse, csv, os, random
import numpy as np
import cv2, torch
import torchvision.transforms as T
from PIL import Image
from skimage.feature import hog

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def tf_eval():
    return T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def hog_energy_map(gray224):
    _, hog_img = hog(gray224, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), block_norm="L2-Hys",
                     visualize=True, feature_vector=False)
    e = hog_img.astype(np.float32)
    if e.max() > e.min():
        e = (e - e.min()) / (e.max() - e.min())
    return e


def face_mask(size=224, frac=0.72):
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, size // 2),
                (int(size * frac / 2), int(size * frac / 2 * 1.15)),
                0, 0, 360, 1, -1)
    return m


def norm01(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def overlap_iou(a, b, thr=0.5):
    A = a >= thr
    B = b >= thr
    return float(np.logical_and(A, B).sum()) / float(np.logical_or(A, B).sum() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400,
                    help="total test faces to sample (half real, half fake)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}", flush=True)

    import timm
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cam = GradCAMPlusPlus(model=model, target_layers=[model.conv_head])

    rows = [r for r in csv.DictReader(open(args.frames)) if r["split"] == "test"]
    real = [r for r in rows if int(r["label"]) == 1]
    fake = [r for r in rows if int(r["label"]) == 0]
    random.seed(args.seed)
    k = args.n // 2
    picks = (random.sample(real, min(k, len(real))) +
             random.sample(fake, min(k, len(fake))))
    random.shuffle(picks)
    print(f"[info] sampling {len(picks)} test faces "
          f"({min(k,len(real))} real + {min(k,len(fake))} fake)", flush=True)

    tfm = tf_eval()
    mask = face_mask(224)
    ious, corrs, onface = [], [], []

    for j, r in enumerate(picks):
        img = Image.open(r["frame_path"]).convert("RGB").resize((224, 224))
        rgb = np.array(img).astype(np.float32) / 255.0
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        x = tfm(img).unsqueeze(0).to(device)
        label = int(r["label"])
        gcam = cam(input_tensor=x, targets=[ClassifierOutputTarget(label)])[0]
        cam_map = norm01(cv2.resize(gcam, (224, 224)))
        hog_map = norm01(hog_energy_map(gray))
        onface.append((cam_map * mask).sum() / (cam_map.sum() + 1e-8))
        ious.append(overlap_iou(cam_map, hog_map, 0.5))
        corrs.append(float(np.corrcoef(cam_map.ravel(), hog_map.ravel())[0, 1]))
        if (j + 1) % 50 == 0:
            print(f"  processed {j+1}/{len(picks)}", flush=True)

    def stat(a):
        a = np.array(a, dtype=np.float64)
        return float(a.mean()), float(a.std())

    mi, si = stat(ious)
    mc, sc = stat(corrs)
    mo, so = stat(onface)
    print("\n=== bulk interpretability summary ===")
    print(f"  n        = {len(picks)}")
    print(f"  IoU      mean={mi:.4f}  sd={si:.4f}")
    print(f"  corr     mean={mc:.4f}  sd={sc:.4f}")
    print(f"  on-face  mean={mo:.4f}  sd={so:.4f}")

    out_csv = os.path.join(args.out, "interp_bulk_summary.csv")
    with open(out_csv, "w") as f:
        f.write("n,iou_mean,iou_sd,corr_mean,corr_sd,onface_mean,onface_sd\n")
        f.write(f"{len(picks)},{mi:.4f},{si:.4f},{mc:.4f},{sc:.4f},"
                f"{mo:.4f},{so:.4f}\n")
    print(f"[done] -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
