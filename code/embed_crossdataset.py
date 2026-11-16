"""
Cross-dataset embedding extraction: run the FF++-trained EfficientNet-B0
over Celeb-DF TEST crops (official 518) to get per-frame P(real) scores
AND 1280-D embeddings. These feed the cross-dataset fusion comparison.

Why: cross-dataset means the FF++ model sees Celeb-DF for the first time.
The Celeb-DF embeddings saved earlier came from the Celeb-DF-trained model
(wrong model). This regenerates them from the FF++ model.

Reads the Celeb-DF frame manifest, filters to split==test (the official 518),
loads crops, runs the FF++ model, saves:
  <out>/celebdf_test_ffppmodel_scores.npz : video_id, label, frame_score
  <out>/celebdf_test_ffppmodel_emb.npz    : video_id, label, X (1280-D)

Run:
  python embed_crossdataset.py \
     --frames ../manifests/celebdf2_frames.csv \
     --model ../cnn/ffpp/model/b0_best.pt \
     --out ../crossdataset
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REAL = 1


def tf_eval():
    return T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--model", required=True, help="FF++ b0_best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", help="which split to embed")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}", flush=True)

    import timm
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[info] loaded FF++ model <- {args.model}", flush=True)

    rows = [r for r in csv.DictReader(open(args.frames)) if r["split"] == args.split]
    print(f"[info] {len(rows)} frames in split={args.split}", flush=True)
    tfm = tf_eval()

    vids, labs, scores, embs = [], [], [], []
    from tqdm import tqdm
    batch_imgs, batch_vid, batch_lab = [], [], []

    @torch.no_grad()
    def flush_batch():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        feat = model.forward_features(x)
        pooled = model.forward_head(feat, pre_logits=True)   # [B,1280]
        logits = model.forward_head(feat)                     # [B,2]
        p = torch.softmax(logits, 1)[:, REAL].cpu().numpy()
        e = pooled.cpu().numpy().astype(np.float32)
        for i in range(len(batch_imgs)):
            vids.append(batch_vid[i]); labs.append(batch_lab[i])
            scores.append(float(p[i])); embs.append(e[i])
        batch_imgs.clear(); batch_vid.clear(); batch_lab.clear()

    for r in tqdm(rows, desc="celebdf-test via ffpp-model"):
        try:
            img = Image.open(r["frame_path"]).convert("RGB")
        except Exception:
            continue
        batch_imgs.append(tfm(img))
        batch_vid.append(r["video_id"]); batch_lab.append(int(r["label"]))
        if len(batch_imgs) >= args.batch:
            flush_batch()
    flush_batch()

    vids = np.array(vids); labs = np.array(labs, dtype=np.int64)
    scores = np.array(scores, dtype=np.float32); embs = np.vstack(embs)

    np.savez_compressed(os.path.join(args.out, "celebdf_test_ffppmodel_scores.npz"),
                        video_id=vids, label=labs, frame_score=scores)
    np.savez_compressed(os.path.join(args.out, "celebdf_test_ffppmodel_emb.npz"),
                        video_id=vids, label=labs, X=embs)
    print(f"[done] scores {scores.shape} + emb {embs.shape} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
