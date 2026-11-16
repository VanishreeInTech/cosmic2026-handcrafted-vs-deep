"""
EfficientNet-B0 deepfake detector (compact CNN arm) -- with full paper outputs.

Trains on the SAME crops / official splits / video-level aggregation as the
handcrafted pipeline, so results are directly comparable.

OUTPUTS (organized under --out-dir):
  model/        b0_best.pt  (+ per-fold for CV)
  history/      training_history.csv  (per-epoch loss/val metrics)
  figures/      learning_curve.png, confusion_matrix.png, roc_curve.png
  predictions/  frame_predictions.csv, video_predictions.csv
  metrics.csv   final video-level metrics (also returned)

Design:
  - 256 PNG crop -> resize 224 -> ImageNet normalize (B0 native res)
  - EfficientNet-B0 via timm, 2-class head
  - class-weighted cross-entropy (Celeb-DF skew)
  - SPLIT mode: official test untouched; video-disjoint val carved from train
  - CV mode (UADFV): 5-fold; within each, one val carve-out. Figures/preds
    are saved for the LAST fold + aggregated metrics across folds.
  - early stopping on val video-level AUC
  - video eval: mean per-frame P(real) -> macro-P/R/F1, Acc, AUC

Run (Celeb-DF):
  python train_cnn.py --frames ../manifests/celebdf2_frames.csv \
     --dataset celebdf2 --out-dir ../cnn/celebdf2 --epochs 15 --batch 32
Run (UADFV 5-fold):
  python train_cnn.py --frames ../manifests/uadfv_frames.csv \
     --dataset uadfv --out-dir ../cnn/uadfv --epochs 20 --batch 32 --cv

Deps: torch torchvision timm pillow numpy scikit-learn matplotlib tqdm
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             roc_auc_score, confusion_matrix, roc_curve)

RANDOM_STATE = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REAL = 1


def read_frames(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def tf_train():
    return T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def tf_eval():
    return T.Compose([T.Resize((224, 224)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class FrameDS(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows; self.t = transform
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["frame_path"]).convert("RGB")
        return self.t(img), int(r["label"]), r["video_id"]


def build_model(device):
    import timm
    return timm.create_model("efficientnet_b0", pretrained=True,
                             num_classes=2).to(device)


def class_weights(rows, device):
    n = defaultdict(int)
    for r in rows:
        n[int(r["label"])] += 1
    total = sum(n.values())
    return torch.tensor([total / (2 * n.get(0, 1)), total / (2 * n.get(1, 1))],
                        dtype=torch.float32, device=device)


@torch.no_grad()
def compute_val_loss(model, loader, device, crit):
    """Average CrossEntropy loss over a loader (for the learning curve)."""
    model.eval()
    tot, nb = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        tot += crit(model(x), y).item(); nb += 1
    return tot / max(1, nb)


@torch.no_grad()
def predict_frames(model, loader, device):
    """Return arrays: video_ids, y_true(frame), score(frame)=P(real)."""
    model.eval()
    vids, ys, ss = [], [], []
    for x, y, v in loader:
        x = x.to(device)
        p = torch.softmax(model(x), 1)[:, REAL].cpu().numpy()
        vids.extend(v); ys.extend(y.numpy().tolist()); ss.extend(p.tolist())
    return np.array(vids), np.array(ys), np.array(ss, dtype=np.float32)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Return per-frame penultimate features (1280-D for B0) + ids/labels,
    for the hybrid fusion stage. Uses timm forward_features + pooling."""
    model.eval()
    vids, ys, feats = [], [], []
    for x, y, v in loader:
        x = x.to(device)
        f = model.forward_features(x)              # [B,1280,H,W]
        f = model.forward_head(f, pre_logits=True) # [B,1280] pooled pre-logits
        feats.append(f.cpu().numpy().astype(np.float32))
        vids.extend(v); ys.extend(y.numpy().tolist())
    X = np.concatenate(feats, axis=0)
    return np.array(vids), np.array(ys, dtype=np.int64), X


def save_embeddings(base, tag, vids, ys, X):
    """Save embeddings as .npz for the fusion script (train/test separately)."""
    d = os.path.join(base, "embeddings")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"emb_{tag}.npz"),
                        video_id=vids, label=ys, X=X)


def aggregate_video(vids, ys, ss):
    by = defaultdict(lambda: {"yt": None, "s": []})
    for v, y, s in zip(vids, ys, ss):
        by[v]["yt"] = int(y); by[v]["s"].append(float(s))
    vv = list(by.keys())
    yt = np.array([by[v]["yt"] for v in vv])
    sc = np.array([np.mean(by[v]["s"]) for v in vv])
    yp = (sc >= 0.5).astype(int)
    return vv, yt, yp, sc


def vmetrics(yt, yp, sc):
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro",
                                                  zero_division=0)
    acc = accuracy_score(yt, yp)
    try:
        auc = roc_auc_score(yt, sc)
    except ValueError:
        auc = float("nan")
    return dict(P=p, R=r, F1=f1, Acc=acc, AUC=auc)


def split_val_from_train(rows, val_frac=0.15, seed=42):
    vids = sorted({r["video_id"] for r in rows})
    rng = np.random.RandomState(seed); rng.shuffle(vids)
    n_val = max(1, int(len(vids) * val_frac))
    val = set(vids[:n_val])
    return ([r for r in rows if r["video_id"] not in val],
            [r for r in rows if r["video_id"] in val])


def ensure_dirs(base):
    for sub in ["model", "history", "figures", "predictions"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


def save_learning_curve(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(ep, [h["train_loss"] for h in history], "b-o", label="train loss")
    ax1.plot(ep, [h["val_loss"] for h in history], "c-d", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_auc"] for h in history], "r-s", label="val AUC")
    ax2.plot(ep, [h["val_f1"] for h in history], "g-^", label="val macro-F1")
    ax2.set_ylabel("val metric", color="r"); ax2.set_ylim(0, 1.02)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=8)
    plt.title("EfficientNet-B0 learning curve")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def save_confusion(yt, yp, path):
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, where=row_sums != 0) * 100.0
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["fake", "real"]); ax.set_yticklabels(["fake", "real"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n(n={cm[i, j]})",
                    ha="center", va="center",
                    color="white" if cm_pct[i, j] > 50 else "black", fontsize=9)
    plt.title("Confusion matrix (video-level, row-normalized)")
    plt.colorbar(im, label="% of true class")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return cm


def save_roc(yt, sc, path):
    try:
        fpr, tpr, _ = roc_curve(yt, sc)
        auc = roc_auc_score(yt, sc)
    except ValueError:
        return
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, "b-", label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (video-level)")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def save_predictions(base, frame_vids, frame_y, frame_s, vv, yt, yp, sc):
    with open(os.path.join(base, "predictions", "frame_predictions.csv"), "w",
              newline="") as f:
        w = csv.writer(f); w.writerow(["video_id", "label", "score_real"])
        for v, y, s in zip(frame_vids, frame_y, frame_s):
            w.writerow([v, int(y), f"{s:.6f}"])
    with open(os.path.join(base, "predictions", "video_predictions.csv"), "w",
              newline="") as f:
        w = csv.writer(f); w.writerow(["video_id", "label", "pred", "score_real"])
        for v, y, p, s in zip(vv, yt, yp, sc):
            w.writerow([v, int(y), int(p), f"{s:.6f}"])


def train_one(tr, va, te, device, args, ckpt, history_out=None):
    model = build_model(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr, device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trl = DataLoader(FrameDS(tr, tf_train()), batch_size=args.batch, shuffle=True,
                     num_workers=args.workers, pin_memory=True)
    val = DataLoader(FrameDS(va, tf_eval()), batch_size=args.batch, shuffle=False,
                     num_workers=args.workers, pin_memory=True)
    from tqdm import tqdm
    best_auc, best_state, patience, history = -1, None, 0, []
    for ep in range(args.epochs):
        model.train(); tot, nb = 0.0, 0
        for x, y, _ in tqdm(trl, desc=f"ep{ep+1}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        vv, yt, ss = predict_frames(model, val, device)
        _, vyt, vyp, vsc = aggregate_video(vv, yt, ss)
        vm = vmetrics(vyt, vyp, vsc)
        vl = compute_val_loss(model, val, device, crit)
        tl = tot / max(1, nb)
        history.append({"epoch": ep + 1, "train_loss": tl, "val_loss": vl,
                        "val_auc": vm["AUC"], "val_f1": vm["F1"]})
        print(f"  epoch {ep+1}: train_loss={tl:.4f} val_loss={vl:.4f} "
              f"val_AUC={vm['AUC']:.3f} val_F1={vm['F1']:.3f}", flush=True)
        if vm["AUC"] > best_auc:
            best_auc = vm["AUC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0; torch.save(best_state, ckpt)
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop @ epoch {ep+1}", flush=True); break
    model.load_state_dict(best_state)
    if history_out:
        with open(history_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss",
                                              "val_auc", "val_f1"])
            w.writeheader(); w.writerows(history)
    tel = DataLoader(FrameDS(te, tf_eval()), batch_size=args.batch, shuffle=False,
                     num_workers=args.workers, pin_memory=True)
    fv, fy, fs = predict_frames(model, tel, device)
    return model, history, (fv, fy, fs)


def write_metrics(path, dataset, m, mode):
    with open(path, "w") as f:
        f.write("dataset,model,P,R,F1,Acc,AUC,mode\n")
        f.write(f"{dataset},EfficientNet-B0,{m['P']:.4f},{m['R']:.4f},"
                f"{m['F1']:.4f},{m['Acc']:.4f},{m['AUC']:.4f},{mode}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip saving CNN embeddings (fusion stage won't be possible)")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}", flush=True)
    ensure_dirs(args.out_dir)
    rows = read_frames(args.frames)

    if args.cv:
        folds = sorted({int(r["fold"]) for r in rows})
        fold_m = []
        last = None
        for k in folds:
            te = [r for r in rows if int(r["fold"]) == k]
            tall_tr = [r for r in rows if int(r["fold"]) != k]
            tr, va = split_val_from_train(tall_tr, seed=RANDOM_STATE + k)
            print(f"\n=== fold {k}: train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)
            ckpt = os.path.join(args.out_dir, "model", f"b0_fold{k}.pt")
            model, hist, (fv, fy, fs) = train_one(tr, va, te, device, args, ckpt)
            vv, yt, yp, sc = aggregate_video(fv, fy, fs)
            m = vmetrics(yt, yp, sc)
            print(f"  fold {k} TEST F1={m['F1']:.3f} AUC={m['AUC']:.3f}", flush=True)
            fold_m.append(m); last = (hist, fv, fy, fs, vv, yt, yp, sc)
        # aggregate
        agg = {kk: float(np.nanmean([fm[kk] for fm in fold_m]))
               for kk in ["P", "R", "F1", "Acc", "AUC"]}
        print(f"\n[CV mean] F1={agg['F1']:.3f} AUC={agg['AUC']:.3f}", flush=True)
        write_metrics(os.path.join(args.out_dir, "metrics.csv"), args.dataset, agg, "5foldCV")
        # figures/preds from last fold
        hist, fv, fy, fs, vv, yt, yp, sc = last
        save_learning_curve(hist, os.path.join(args.out_dir, "figures", "learning_curve.png"))
        save_confusion(yt, yp, os.path.join(args.out_dir, "figures", "confusion_matrix.png"))
        save_roc(yt, sc, os.path.join(args.out_dir, "figures", "roc_curve.png"))
        save_predictions(args.out_dir, fv, fy, fs, vv, yt, yp, sc)
    else:
        tr_all = [r for r in rows if r["split"] == "train"]
        te = [r for r in rows if r["split"] == "test"]
        existing_val = [r for r in rows if r["split"] == "val"]
        if existing_val:
            tr, va = tr_all, existing_val
            print(f"[info] using manifest's existing val split "
                  f"(identity-disjoint)", flush=True)
        else:
            tr, va = split_val_from_train(tr_all, seed=RANDOM_STATE)
            print(f"[info] no val in manifest; carved val from train", flush=True)
        print(f"\n=== split: train={len(tr)} val={len(va)} test={len(te)} ===", flush=True)
        ckpt = os.path.join(args.out_dir, "model", "b0_best.pt")
        hist_out = os.path.join(args.out_dir, "history", "training_history.csv")
        model, hist, (fv, fy, fs) = train_one(tr, va, te, device, args, ckpt, hist_out)
        vv, yt, yp, sc = aggregate_video(fv, fy, fs)
        m = vmetrics(yt, yp, sc)
        print(f"\n[TEST] F1={m['F1']:.3f} AUC={m['AUC']:.3f} Acc={m['Acc']:.3f} "
              f"P={m['P']:.3f} R={m['R']:.3f}", flush=True)
        write_metrics(os.path.join(args.out_dir, "metrics.csv"), args.dataset, m, "split")
        save_learning_curve(hist, os.path.join(args.out_dir, "figures", "learning_curve.png"))
        save_confusion(yt, yp, os.path.join(args.out_dir, "figures", "confusion_matrix.png"))
        save_roc(yt, sc, os.path.join(args.out_dir, "figures", "roc_curve.png"))
        save_predictions(args.out_dir, fv, fy, fs, vv, yt, yp, sc)
        # embeddings for hybrid fusion (train + test, eval transform)
        if not args.no_embed:
            print("[embed] extracting CNN embeddings for fusion ...", flush=True)
            tr_loader = DataLoader(FrameDS(tr, tf_eval()), batch_size=args.batch,
                                   shuffle=False, num_workers=args.workers, pin_memory=True)
            ev, ey, eX = extract_embeddings(model, tr_loader, device)
            save_embeddings(args.out_dir, "train", ev, ey, eX)
            te_loader2 = DataLoader(FrameDS(te, tf_eval()), batch_size=args.batch,
                                    shuffle=False, num_workers=args.workers, pin_memory=True)
            tv2, ty2, tX2 = extract_embeddings(model, te_loader2, device)
            save_embeddings(args.out_dir, "test", tv2, ty2, tX2)
            print(f"[embed] saved train {eX.shape} + test {tX2.shape}", flush=True)

    print(f"\n[done] outputs -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
