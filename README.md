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
   - [Segment Raw Videos](#1-segment-raw-videos)
   - [Prepare Train/Test Splits](#2-prepare-traintest-splits)
   - [Train the Model](#3-train-the-x3d-model)
   - [Evaluate Performance](#4-evaluate-model-performance)
   - [Run Inference](#5-real-time-inference)
5. [Model Architecture](#-model--architecture)
6. [Related Work](#-related-work)
7. [Credits & Acknowledgments](#-credits--acknowledgments)
8. [Citation](#-citation)

---

## 📘 Overview
Existing anomaly detection benchmarks often contain **scene duplication**, **annotation artifacts**, and **cross-partition overlaps**, leading to inflated model performance.  
**SCAD** (Surveillance Curated Accident Dataset) corrects these issues by offering a **transparent, ethically cleaned, and rigorously curated dataset** designed for **fair and reproducible benchmarking** in accident detection.

---

## 🔑 Highlights
- 🧹 **Clean & Verified Dataset** – 513 manually reviewed clips covering accident and non-accident scenarios  
- ⚙️ **Lightweight 3D Model** – X3D backbone ensures efficient training on modest GPUs  
- 🧪 **End-to-End Pipeline** – segmentation → training → evaluation → inference  
- 🔍 **Reproducible Splits** – no cross-contamination between train/test data  
- 📈 **Benchmark-ready Baselines** – includes pretrained models (81.9% and 84.48% accuracy)

> ⚠️ **Academic Use Only**  
> SCAD is provided strictly for **non-commercial research and academic benchmarking**.  
> Please cite the repository and the associated paper upon use.

---

## 🎬 Dataset Description
SCAD was built by curating, cleaning, and harmonizing accident clips from **multiple public surveillance datasets**.  
Each raw video was **segmented into 4-second overlapping clips (3-second overlap)** to ensure temporal continuity while maintaining clip diversity.  
All redundant or near-duplicate frames were eliminated to preserve **benchmarking integrity**.

| Source | Reference | Role in SCAD |
|---------|------------|--------------|
| **CADP** | IEEE AVSS 2018 | CCTV-based traffic accident dataset |
| **So-TAD** | Neurocomputing 2025 | Surveillance-oriented traffic anomaly dataset |
| **TU-DAT** | Sensors 2025 | Road traffic anomaly dataset |
| **UCF-Crime** | CVPR 2018 | General anomaly dataset including traffic accidents |

---

## 🚀 Usage Guide

### 1. Segment Raw Videos
Generate 4-second overlapping clips from long surveillance recordings.

```bash
python segment.py
```

**Edit inside `segment.py`:**
```python
input_folder  = "videos/"
output_folder = "accident/"
```

---

### 2. Prepare Train/Test Splits
Create two text files in your dataset directory:
```
accident_train.txt
accident_test.txt
```

Each line should specify a relative path to a clip:
```
anomalous/clip_001.mp4
normal/clip_145.mp4
```

---

### 3. Train the X3D Model
Fine-tune a pretrained X3D model on SCAD.

```bash
python train.py
```

**Edit inside `train.py`:**
```python
main_folder  = "accident/"
train_split  = "accident_train.txt"
test_split   = "accident_test.txt"
```

Optional adjustments:
- Input resolution  
- Frame sampling rate  
- Optimizer and learning rate  
- Pretrained checkpoint path  

---

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

---

### 5. Real-Time Inference
Run detection on live or recorded video streams.

```bash
python inference.py
```

The script continuously processes frames and prints alerts when anomalous (accident) activity is detected.

---

## 🧠 Model & Architecture

| Component | Description |
|------------|-------------|
| **Backbone** | X3D (efficient spatio-temporal CNN by Facebook AI) |
| **Framework** | PyTorch |
| **Input** | RGB frame sequences |
| **Output** | Binary classification — accident / non-accident |

**Included pretrained models**

| Model | Resolution | Frames | Description |
|--------|-------------|---------|-------------|
| `81.9model.pth` | 128×128 | 30 | Baseline configuration |
| `84.48model.pth` | 182×182 | 25 | Enhanced configuration |

---

## 🔗 Related Work
The SCAD implementation draws from foundational research in spatio-temporal event detection, extending efficiency and reproducibility principles from the **STEAD repository (A. Gao et al.)** and integrating them with the **X3D architecture (Feichtenhofer et al., Facebook AI)** to achieve a compact, high-accuracy accident detection benchmark.

---

## 👥 Credits & Acknowledgments
- **STEAD** – by *Andrew Gao & Jun Liu*, reference for efficient anomaly detection  
- **X3D** – by *C. Feichtenhofer et al.*, Facebook AI, for compact 3D CNN backbone  
- **Source Datasets** – *CADP, So-TAD, TU-DAT, UCF-Crime*  

---

## 🧾 Citation
If you use SCAD in your research, please cite:

```bibtex
@inproceedings{scad2025,
  title     = {From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNN},
  author    = {Authors of SCAD},
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

⭐ **If this repository supports your work, please consider starring it** — help promote open, transparent, and reproducible research in surveillance accident detection.

---

### 🧩 Repository Structure Example
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
