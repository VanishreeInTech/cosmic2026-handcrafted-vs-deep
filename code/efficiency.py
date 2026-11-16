"""
Efficiency analysis: compact CNN vs handcrafted descriptors.

Reports, for the paper's "lightweight" claim:
  CNN (EfficientNet-B0): #params, FLOPs (per 224x224 frame), model size (MB),
                         GPU inference time per frame + per video (32 frames)
  Handcrafted (per 256x256 frame, CPU): extraction time per descriptor,
                         feature dimensionality
  Classifiers: negligible; note train-once, fast inference.

Run:
  python efficiency.py --model ../cnn/ffpp/model/b0_best.pt \
     --sample-frame <path-to-any-crop.png> --out ../results/efficiency.csv

If --sample-frame omitted, uses a random tensor for timing.
Deps: torch, timm, thop (FLOPs), opencv, skimage, numpy
  pip install thop --break-system-packages
"""
import argparse, csv, os, time
import numpy as np


def cnn_stats(model_path, device, n_warm=5, n_iter=50):
    import torch, timm
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    if model_path and os.path.exists(model_path):
        m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device).eval()

    n_params = sum(p.numel() for p in m.parameters())
    size_mb = n_params * 4 / 1e6  # float32

    # FLOPs
    flops = None
    try:
        from thop import profile
        dummy = torch.randn(1, 3, 224, 224).to(device)
        flops, _ = profile(m, inputs=(dummy,), verbose=False)
    except Exception as e:
        print(f"[warn] FLOPs via thop failed: {e}")

    # timing
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warm):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        per_frame = (time.time() - t0) / n_iter
    return dict(params=n_params, size_mb=size_mb, flops=flops,
                per_frame_s=per_frame, per_video_s=per_frame * 32)


def hog_time(sample_gray, n_iter=20):
    from skimage.feature import hog
    t0 = time.time()
    for _ in range(n_iter):
        hog(sample_gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return (time.time() - t0) / n_iter


def descriptor_times(sample_gray):
    import cv2
    from skimage.feature import local_binary_pattern
    out = {}
    # HOG
    out["hog"] = hog_time(sample_gray)
    # SIFT
    sift = cv2.SIFT_create()
    t0 = time.time()
    for _ in range(20):
        sift.detectAndCompute(sample_gray, None)
    out["sift"] = (time.time() - t0) / 20
    # ORB
    orb = cv2.ORB_create(nfeatures=500)
    t0 = time.time()
    for _ in range(20):
        orb.detectAndCompute(sample_gray, None)
    out["orb"] = (time.time() - t0) / 20
    # LBP
    t0 = time.time()
    for _ in range(20):
        local_binary_pattern(sample_gray, 12, 2, method="uniform")
    out["lbp"] = (time.time() - t0) / 20
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample-frame", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}", flush=True)

    # sample gray 256 for descriptors
    if args.sample_frame and os.path.exists(args.sample_frame):
        import cv2
        gray = cv2.imread(args.sample_frame, cv2.IMREAD_GRAYSCALE)
        if gray.shape[0] != 256:
            gray = cv2.resize(gray, (256, 256))
    else:
        gray = (np.random.rand(256, 256) * 255).astype(np.uint8)

    cnn = cnn_stats(args.model, device)
    desc = descriptor_times(gray)

    dims = {"hog": 34596, "sift": 128, "orb": 16000, "lbp": 14}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "params_or_dim", "size_MB", "GFLOPs",
                    "time_per_frame_ms", "time_per_video_ms", "hardware"])
        w.writerow(["EfficientNet-B0", cnn["params"], f"{cnn['size_mb']:.1f}",
                    f"{cnn['flops']/1e9:.3f}" if cnn["flops"] else "NA",
                    f"{cnn['per_frame_s']*1000:.2f}",
                    f"{cnn['per_video_s']*1000:.2f}", "GPU (RTX 4070)"])
        for d in ["hog", "sift", "orb", "lbp"]:
            w.writerow([f"{d.upper()} (descriptor)", dims[d], "-", "-",
                        f"{desc[d]*1000:.2f}", f"{desc[d]*32*1000:.2f}", "CPU"])

    print("\n=== EFFICIENCY SUMMARY ===")
    print(f"EfficientNet-B0: {cnn['params']/1e6:.2f}M params, "
          f"{cnn['size_mb']:.1f}MB, "
          f"{cnn['flops']/1e9:.2f} GFLOPs, "
          f"{cnn['per_frame_s']*1000:.2f} ms/frame (GPU)")
    for d in ["hog", "sift", "orb", "lbp"]:
        print(f"{d.upper():5s}: dim={dims[d]:6d}  {desc[d]*1000:.2f} ms/frame (CPU)")
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
