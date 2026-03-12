# RetinaAVSeg

Deep learning pipeline for retinal artery-vein segmentation from fundus images using Multi-Class Attention U-Net.

---

## 📁 Project Structure

```
retinaavseg/
├── main/
│   ├── dataset/                         # Dataset loader & preprocessing
│   ├── models/                          # Saved models
│   ├── job.sh                           # Single GPU training
│   ├── job_ddp.sh                       # Multi-GPU training
│   ├── train.py                     # Single GPU trainer
│   ├── train_ddp.py                     # DDP trainer
│   ├── model.py
│   ├── metrics.py
│   ├── utils.py
│   ├── visualizers/vessel_analyzer.py   # to perfrom analysis on av
│   ├── visualizers/visualizer.py        # To visualize 
├── samples/                             # Sample images
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
conda create -n retinaavseg python=3.11
conda activate retinaavseg
conda install -c conda-forge uv
uv pip install -r requirements.txt
```

### 2. Configure Paths

Update the following in training scripts:

- **Conda path**: `source /path/to/miniconda/bin/activate retinaavseg`
- **Dataset path**: Update in `train_old.py` or `train_ddp.py`
- **GPU settings**: Set GPU IDs as needed

### 3. Train Model

**Single GPU:**
```bash
cd main
bash job.sh
```

**Multi-GPU (DDP):**
```bash
cd main
bash job_ddp.sh
```

### 4. Run Inference

```bash
cd main/visualizers
python visualize.py
```

Open the Gradio interface to segment fundus images.

---

## 📦 Model Output

Trained models are saved in `main/files/`. Copy to `main/models/` for inference.

---

## 📝 Notes

- Update conda paths, dataset locations, and GPU configs before training
- Compatible with local systems and HPC clusters
- All core scripts are in `main/` directory