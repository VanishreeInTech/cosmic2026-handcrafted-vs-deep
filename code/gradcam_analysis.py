"""
Interpretability: Grad-CAM++ (CNN) vs HOG gradient-energy, on the same faces.

Goal: explain WHY handcrafted HOG complements the compact CNN. If the CNN's
attention (Grad-CAM++) and HOG's gradient energy fall on DIFFERENT facial
regions, that spatial complementarity explains the fusion gain.

For chosen test frames (real + fake):
  1. Grad-CAM++ heatmap from the trained EfficientNet-B0 (last conv block)
  2. HOG gradient-energy map (per-cell gradient magnitude)
  3. overlays on the face + side-by-side figure
  4. quantify: on-face energy ratio + CNN-HOG spatial overlap (IoU of hot
     regions) and correlation. Low overlap => complementary => explains fusion.

Fallback: if Grad-CAM++ maps are too diffuse, pass --method scorecam.

Run:
  python gradcam_analysis.py \
     --model ../cnn/ffpp/model/b0_best.pt \
     --frames ../manifests/ffpp_frames.csv \
     --out ../interpretability/ffpp --n-examples 8 --method gradcampp

Deps: pip install grad-cam --break-system-packages
"""
import argparse, csv, os, random
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torchvision.transforms as T
from skimage.feature import hog

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def tf_eval():
    return T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def hog_energy_map(gray224):
    """Per-cell HOG gradient energy, upsampled to 224x224 in [0,1]."""
    _, hog_img = hog(gray224, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), block_norm="L2-Hys",
                     visualize=True, feature_vector=False)
    e = hog_img.astype(np.float32)
    if e.max() > e.min():
        e = (e - e.min()) / (e.max() - e.min())
    return e


def face_mask(size=224, frac=0.72):
    """Central elliptical face-region mask (crops are face-centered)."""
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, size // 2),
                (int(size * frac / 2), int(size * frac / 2 * 1.15)),
                0, 0, 360, 1, -1)
    return m


def norm01(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def overlap_iou(a, b, thr=0.5):
    A = a >= thr; B = b >= thr
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter) / float(union + 1e-8)


def build_cam(model, method, device):
    from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM
    # EfficientNet-B0 last conv block target layer (timm): conv_head
    target_layers = [model.conv_head]
    cls = GradCAMPlusPlus if method == "gradcampp" else ScoreCAM
    return cls(model=model, target_layers=target_layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--method", default="gradcampp", choices=["gradcampp", "scorecam"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} method={args.method}", flush=True)

    import timm
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cam = build_cam(model, args.method, device)

    # pick balanced test examples: n/2 real, n/2 fake
    rows = [r for r in csv.DictReader(open(args.frames)) if r["split"] == "test"]
    real = [r for r in rows if int(r["label"]) == 1]
    fake = [r for r in rows if int(r["label"]) == 0]
    random.seed(args.seed)
    k = args.n_examples // 2
    picks = random.sample(real, k) + random.sample(fake, k)

    tfm = tf_eval()
    mask = face_mask(224)
    metrics = []

    for i, r in enumerate(picks):
        img = Image.open(r["frame_path"]).convert("RGB").resize((224, 224))
        rgb = np.array(img).astype(np.float32) / 255.0
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        x = tfm(img).unsqueeze(0).to(device)
        label = int(r["label"])
        grayscale_cam = cam(input_tensor=x,
                            targets=[ClassifierOutputTarget(label)])[0]
        cam_map = norm01(cv2.resize(grayscale_cam, (224, 224)))
        hog_map = norm01(hog_energy_map(gray))

        # metrics
        cam_onface = (cam_map * mask).sum() / (cam_map.sum() + 1e-8)
        hog_onface = (hog_map * mask).sum() / (hog_map.sum() + 1e-8)
        iou = overlap_iou(cam_map, hog_map, 0.5)
        corr = float(np.corrcoef(cam_map.ravel(), hog_map.ravel())[0, 1])
        metrics.append(dict(idx=i, label=label, cam_onface=cam_onface,
                            hog_onface=hog_onface, iou=iou, corr=corr))

        # figure: original | Grad-CAM++ overlay | HOG overlay
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.5))
        ax[0].imshow(rgb); ax[0].set_title("face"); ax[0].axis("off")
        ax[1].imshow(rgb); ax[1].imshow(cam_map, cmap="jet", alpha=0.5)
        ax[1].set_title(f"{args.method}"); ax[1].axis("off")
        ax[2].imshow(rgb); ax[2].imshow(hog_map, cmap="jet", alpha=0.5)
        ax[2].set_title("HOG energy"); ax[2].axis("off")
        tag = "real" if label == 1 else "fake"
        fig.suptitle(f"{tag}  IoU={iou:.2f} corr={corr:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"ex{i:02d}_{tag}.png"), dpi=140)
        plt.close()

    # aggregate metrics
    with open(os.path.join(args.out, "interp_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "label", "cam_onface",
                                          "hog_onface", "iou", "corr"])
        w.writeheader()
        for m in metrics:
            w.writerow(m)

    mean_iou = np.mean([m["iou"] for m in metrics])
    mean_corr = np.mean([m["corr"] for m in metrics])
    mean_cam_on = np.mean([m["cam_onface"] for m in metrics])
    print(f"\n=== interpretability summary ===")
    print(f"  mean CNN on-face energy : {mean_cam_on:.3f}")
    print(f"  mean CNN-HOG IoU        : {mean_iou:.3f}")
    print(f"  mean CNN-HOG corr       : {mean_corr:.3f}")
    print(f"  (low IoU/corr => spatially complementary => explains fusion gain)")
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
