# SmartShot — Scorer (Student Aesthetic Model via KD)

Compact aesthetic scorer for SmartShot.
A student model (EMOv2) is distilled from a strong teacher (SigLIP + LAION Aesthetics v2.5).
Supports KD on AADB/TAD66K and evaluation-only CPC baseline.

## 1. structure
```commandline
repo-root/
├─ scorer/
│  ├── models/                  # EMOv2 backbone wrapper
│  ├── datasets/                # AVA / AADB / TAD66K loaders
│  ├── utils/                   # SRCC, PLCC, checkpoint helpers
│  ├── train_ava.py             # Two-stage AVA training entry point
│  ├── train_distill_emo.py     # Knowledge-distillation training
│  ├── runs/
│  │   └── ava_finetuned/
│  │       └── ckpt_best.pt     # Final model checkpoint (5.2 M params)
│  └── requirements.txt
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
```commandline
# Install dependencies (conda)
conda install --file requirements.txt

python train_ava.py \
        --ava_root ..\data\AVA_dataset \
        --variant emo2_5m_224 \
        --epochs 15 \
        --batch_size 112 \
        --lr 3e-4 \
        --freeze_backbone \
        --amp \
        --out_dir runs\ava_frozen
```