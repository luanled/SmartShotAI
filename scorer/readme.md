# SmartShot — Scorer (Student Aesthetic Model via KD)

Compact aesthetic scorer for SmartShot.
A student model (EMOv2) is distilled from a strong teacher (SigLIP + LAION Aesthetics v2.5).
Supports KD on AADB/TAD66K and evaluation-only CPC baseline.

## 1. structure
```commandline
repo-root/
├─ scorer/
│  ├─ train_distill_emo.py          # entry point: student KD trainer
│  ├─ models/                       # EMOv2 wrapper → MLP → scalar [0,1]
│  ├─ teacher/                      # SigLIP + LAION v2.5 predictor
│  ├─ datasets/
│  │  ├─ aadb.py                    # AADB loader (URL→filename resolver)
│  │  └─ tad66k.py                  # TAD66K loader (merge/train,test CSVs)
│  ├─ utils/                        # metrics, common helpers
│  └─ requirements.txt              # pip deps
├─ data/
│  ├─ AADB/                         # AADB dataset root
│  ├─ TAD66K/                       # TAD66K dataset root
│  └─ CPC/                          # CPC dataset (eval-only)
└─ runs/                            # outputs: logs, ckpts, metrics
```
**Note.** This repo hosts only the scorer (student + teacher + loaders). RL/APP code lives outside and consumes the scorer’s score(image) → `ŝ ∈ [0,1]` API.

## 2. installation
### 2.1 Create env
```commandline
# conda recommended (Python 3.10+)
conda create -y -n smartshot python=3.10
conda activate smartshot

# install PyTorch that matches your CUDA (or CPU)
# (example for CUDA 12.x – adjust per pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
### 2.2 Python deps
```commandline
# from repo-root/
pip install -r scorer/requirements.txt
```
## 3. Datasets (locations only)
- AADB — crowd-rated aesthetics; original `[1,5]` scores are mapped to `[0,1]`.
- TAD66K — large, thematically diverse photo corpus; used for KD and generalization.
- CPC — crop-ranking benchmark from AutoPhoto; evaluation-only for baseline.
Place them under data/ (or anywhere) and pass roots with CLI flags

## 4. Quickstart
### 4.1 KD on AADB (teacher guidance only)
```commandline
# from repo-root/
python scorer/train_distill_emo.py ^
  --dataset aadb ^
  --aadb_root data/AADB ^
  --variant emo2_5m_224 ^
  --epochs 10 ^
  --batch_size 32 ^
  --amp ^
  --use_teacher --teacher_kind v25 ^
  --out_dir runs/aadb_kd_v25
```
### 4.2 KD on TAD66K
```commandline
# from repo-root/
python scorer/train_distill_emo.py ^
  --dataset aadb ^
  --aadb_root data/AADB ^
  --variant emo2_5m_224 ^
  --epochs 10 ^
  --batch_size 32 ^
  --amp ^
  --use_teacher --teacher_kind v25 ^
  --out_dir runs/aadb_kd_v25
```
**Common flags**
- --num_workers -1 (auto) or 0 on Windows
- --resume runs/.../ckpt_best.pt to continue
- --early_stop_patience 8 (default)
- --lambda_supervised 0.0 (pure KD; set >0 to blend GT when available)
Outputs in `runs/...: ckpt_best.pt`, `results_val.json`, `results_test.json`, logs.