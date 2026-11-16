# Comparative Analysis of Handcrafted and Deep Feature Representations for Deepfake Detection

Code, execution logs and result files for the paper of the same title, IEEE COSMIC 2026 (Paper ID 992).

Authors: Vanishree B S, Karunakar A. Kotegar, Ramakrishna M, Shavantrevva Bilakeri
(Manipal Institute of Technology, MAHE)

## Repository contents

| Folder | Contents |
|---|---|
| `code/` | Scripts used for all reported experiments |
| `logs/` | Execution logs of descriptor extraction, classifier evaluation and CNN training runs |
| `results/` | Result files behind the tables in the paper |
| `cnn/` | EfficientNet-B0 metrics, training history, frame- and video-level predictions and figures for each dataset |
| `interpretability/` | Summary of the Grad-CAM++ and HOG overlap analysis |
| `manifests/` | Video and frame lists with labels and train/validation/test splits |

## Scripts

| Stage | Script |
|---|---|
| Split construction | `build_splits.py` (UADFV 5-fold, Celeb-DF v2 official list), `build_splits_ffpp.py` (FF++ official split), `build_idsplit.py` (Celeb-DF v2 identity-disjoint split) |
| Split checks | `verify_official_split.py`, `check_leakage.py`, `check_leakage2.py` |
| Preprocessing | `preprocess.py` (MTCNN face detection, eye alignment, 1.3x crop, 256x256, 32 frames per video), `manifest.py` (shared manifest format) |
| Handcrafted features | `extract_descriptors.py` (HOG, SIFT, ORB, LBP) |
| Handcrafted classifiers | `evaluate.py` (default training), `evaluate_balanced.py` (class-weighted training), `eval_idsplit_handcrafted_NEW_CELEB.py` (Celeb-DF v2 identity-disjoint split) |
| CNN baseline | `train_cnn.py` (EfficientNet-B0) |
| Per-manipulation analysis | `per_method_ffpp_v2.py`, `per_method_ffpp.py` |
| Cross-dataset and fusion | `embed_crossdataset.py`, `fusion_crossdataset.py` |
| Inference time | `efficiency.py` |
| Interpretability | `gradcam_analysis.py` (Fig. 2), `interp_metrics_bulk.py` (Section IV-E) |

Usage for each script is given in the comment block at the top of the file.

## Mapping of paper results to files

| Paper | File(s) |
|---|---|
| Table I | `results/uadfv_default.csv`, `results/ffpp_default.csv` (best ROC-AUC of the three classifiers) |
| Table II | `results/uadfv_default.csv`, `results/ffpp_default.csv`, `results/celebdf2_idsplit_hog_ALL3.csv` |
| Table III | Handcrafted rows as in Table II, CNN rows from `cnn/*/metrics.csv` |
| Section IV-A, official Celeb-DF v2 list | `results/celebdf2_official_bal4.csv`, `cnn/celebdf2/metrics.csv` |
| Table IV | HOG column from `results/ffpp_per_method_v2.csv`, CNN column from `results/ffpp_per_method.csv` |
| Table V | `results/crossdataset_fusion.csv` |
| Table VI | `results/efficiency.csv` |
| Section IV-E | `interpretability/interp_bulk_summary.csv` |

Note on Table IV: `per_method_ffpp.py` produced the CNN column. Its HOG column was computed with a different HOG model and is superseded by `per_method_ffpp_v2.py`, which uses the same HOG and linear SVM model as Table III.

## Not included

- Datasets: UADFV, FaceForensics++ (c23) and Celeb-DF v2 must be obtained from their original providers under their licences. The manifests refer to local file paths.
- Model weights and CNN embeddings, because of their size. They are available from the authors on request.

## Environment

Python packages are listed in `requirements.txt`. CNN training and timing used an NVIDIA RTX 4070 GPU. Handcrafted descriptors were run on CPU.
