# 🧠 SCAD: Surveillance Curated Accident Dataset  
*A Benchmark for Fair and Reproducible Accident Detection in Surveillance Videos*

[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC--BY--NC--4.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)](https://pytorch.org/)
[![Dataset Size](https://img.shields.io/badge/clips-513-lightgrey)](#-dataset-description)

---

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Highlights](#-highlights)
3. [Dataset Description](#-dataset-description)
4. [Usage Guide](#-usage-guide)
5. [Model Architecture](#-model--architecture)
6. [Paper (Under Review)](#-paper-under-review)
7. [Related Work](#-related-work)
8. [Credits & Acknowledgments](#-credits--acknowledgments)
9. [Citation](#-citation)

---

## 📘 Overview
Existing anomaly detection benchmarks often contain **scene duplication**, **annotation artifacts**, and **cross-partition overlaps**, leading to inflated model performance.  
**SCAD** (Surveillance Curated Accident Dataset) addresses these flaws by providing a **curated, transparent, and ethically cleaned dataset** for **reliable accident detection benchmarking**.

SCAD supports both **research reproducibility** and **fair comparison** across models. The dataset is designed to eliminate training bias and reflect real-world surveillance variability across locations, lighting, and camera angles.

---

## 🔑 Highlights
- 🧹 **Clean & Verified Dataset** – 513 manually reviewed accident/non-accident clips  
- ⚙️ **Lightweight 3D Model** – X3D backbone optimized for modest GPUs  
- 🧪 **End-to-End Pipeline** – segmentation → training → evaluation → inference  
- 🔍 **Zero Overlap** – no cross-contamination between training and testing data  
- 📊 **Benchmark Ready** – pretrained models provided for baseline performance  
- 🧠 **Supports Model Generalization Testing** – assess transferability across scenes  

> ⚠️ **Academic Use Only**  
> SCAD is released exclusively for **non-commercial research and benchmarking**.  
> Please acknowledge and cite the repository and paper when using this dataset.

---

## 🎬 Dataset Description
Each raw video was divided into **4-second overlapping clips (3-second overlap)** to preserve motion context and event continuity.  
Duplicate, blurred, or redundant frames were removed using hash-based and optical-flow consistency checks to ensure **scene uniqueness**.

| Source | Reference | Role in SCAD |
|---------|------------|--------------|
| **CADP** | IEEE AVSS 2018 | CCTV-based traffic accident dataset |
| **So-TAD** | Neurocomputing 2025 | Surveillance-oriented traffic anomaly dataset |
| **TU-DAT** | Sensors 2025 | Road traffic anomaly dataset |
| **UCF-Crime** | CVPR 2018 | General anomaly dataset including accidents |

**Key Statistics**
- Total Clips: 513  
- Clip Duration: 4 seconds  
- Accident vs Normal: Balanced (≈1:1)  
- Resolution Range: 128×128 – 182×182  
- Frame Rate: 25–30 fps

---

## 🚀 Usage Guide

### 1. Segment Raw Videos
```bash
python segment.py
```
**Inside `segment.py`:**
```python
input_folder  = "videos/"
output_folder = "accident/"
```

### 2. Prepare Train/Test Splits
```bash
accident_train.txt
accident_test.txt
```
Example entries:
```
anomalous/clip_001.mp4
normal/clip_145.mp4
```

### 3. Train the X3D Model
```bash
python train.py
```
**Inside `train.py`:**
```python
main_folder  = "accident/"
train_split  = "accident_train.txt"
test_split   = "accident_test.txt"
```

### 4. Evaluate Model Performance
```bash
python test.py
```
**Inside `test.py`:**
```python
dataset_folder = "accident/"
test_split     = "accident_test.txt"
model_path     = "84.48model.pth"
```

### 5. Real-Time Inference
Run accident detection on live or recorded surveillance streams:
```bash
python inference.py
```

---

## 🧠 Model & Architecture

| Component | Description |
|------------|-------------|
| **Backbone** | X3D (efficient spatio-temporal CNN by Facebook AI) |
| **Framework** | PyTorch |
| **Input** | RGB frame sequences |
| **Output** | Binary classification (accident / non-accident) |

**Included pretrained models:**

| Model | Resolution | Frames | Description |
|--------|-------------|---------|-------------|
| `81.9model.pth` | 128×128 | 30 | Baseline model |
| `84.48model.pth` | 182×182 | 15 | Enhanced configuration |

---

## 🧾 Paper (Under Review)
*The associated research paper providing methodological details, evaluation metrics, and extended ablation results is currently under double-blind peer review.*  
Details will be added upon acceptance.

---

## 🔗 Related Work
SCAD is conceptually aligned with prior spatio-temporal event detection studies, incorporating principles from:
- **STEAD (A. Gao et al.)** – Efficient event detection and reproducibility frameworks  
- **X3D (C. Feichtenhofer et al., Facebook AI)** – Scalable 3D CNN for video understanding  
- **UCF-Crime, CADP, So-TAD, TU-DAT** – Foundational surveillance anomaly benchmarks  

---

## 👥 Credits & Acknowledgments
- **STEAD Repository** — by *Andrew Gao & Jun Liu*  
- **X3D Backbone** — by *C. Feichtenhofer et al.*, Facebook AI  
- **Base Datasets Integrated:** CADP, So-TAD, TU-DAT, and UCF-Crime  
- **Special Thanks:** Open-source research community contributing to anomaly detection transparency  

---

## 🧾 Citation
If you use SCAD in your research, please cite:

```bibtex
@inproceedings{scad2025,
  title     = {not released yet},
  author    = {not released yet},
  year      = {2025},
  note      = {Under review}
}

@inproceedings{shah2018cadp,
  title     = {CADP: A Novel Dataset for CCTV Traffic Camera based Accident Analysis},
  author    = {Shah, A. P. and Lamare, J.-B. and Nguyen-Anh, T. and Hauptmann, A.},
  booktitle = {AVSS},
  year      = {2018}
}

@article{chen2025sotad,
  title     = {SO-TAD: A Surveillance-Oriented Benchmark for Traffic Accident Detection},
  journal   = {Neurocomputing},
  year      = {2025}
}

@article{pradeep2025tudat,
  title     = {TU-DAT: A Computer Vision Dataset on Road Traffic Anomalies},
  journal   = {Sensors},
  year      = {2025}
}

@inproceedings{sultani2018ucfcrime,
  title     = {Real-World Anomaly Detection in Surveillance Videos},
  author    = {Sultani, W. and Chen, C. and Shah, M.},
  booktitle = {CVPR},
  year      = {2018}
}
```

⭐ **If this repository supports your research, please consider starring it** — your support encourages open, transparent, and reproducible benchmarking in accident detection.

---

### 🧩 Repository Structure
```
SCAD/
├── videos/
├── accident/
│   ├── anomalous/
│   ├── normal/
│   ├── accident_train.txt
│   ├── accident_test.txt
├── segment.py
├── train.py
├── test.py
├── inference.py
└── models/
    ├── 81.9model.pth
    └── 84.48model.pth
```

---

© 2025 SCAD Authors — Released under **CC BY-NC 4.0 License**
