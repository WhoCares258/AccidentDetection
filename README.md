# 🚦 SCAD: Surveillance Curated Accident Dataset and X3D-Based Detection

> Official repository for **“From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNN.”**  
> This project introduces **SCAD**, a *clean, reproducible*, and *benchmark-ready* dataset for accident detection in surveillance videos, alongside a **lightweight X3D-based 3D CNN** reference model.

---

## 📘 Overview  
SCAD addresses a critical problem in existing anomaly-detection datasets: contamination from *scene duplication, annotation artifacts, and cross-partition overlap* that lead to inflated benchmark metrics.  
We curate and clean multiple public surveillance datasets to build a balanced, transparent, and ready-to-train accident detection benchmark.

**Key Features:**
- 🧹 **High-quality clips** — 513 accident and non-accident scenes, filtered and segmented  
- ⚙️ **Lightweight model** — based on X3D, suitable for limited compute  
- 🧪 **End-to-end pipeline** — segmentation → training → testing → inference  
- 🔍 **Transparent benchmarking** — no cross-over contamination, reproducible splits  

---

## 🗂️ Repository Structure

AccidentDetection/
├── videos/ # Unsegmented source videos
│
├── accident/
│ ├── anomalous/ # Clips with accidents
│ ├── normal/ # Clips without accidents
│ ├── train/ # Training split folder
│ ├── test/ # Testing split folder
│ ├── accident_train.txt # List of training clips
│ └── accident_test.txt # List of testing clips
│
├── segment.py # Script to segment videos
├── train.py # Training / fine-tuning script
├── test.py # Testing / evaluation script
├── inference.py # Real-time inference script
├── 81.9model.pth # Baseline pretrained checkpoint
├── 84.48model.pth # Enhanced pretrained checkpoint
└── README.md # This file

yaml
Copy code

---

## 🎬 Dataset Description

SCAD is constructed from multiple established public datasets. Each video is subdivided into **4-second clips**, overlapping by **3 seconds**, ensuring full capture of events. We carefully remove duplicates and cross-dataset overlaps to preserve benchmark integrity.

| Source | Reference | Role in SCAD |
|--------|-----------|---------------|
| **CADP** | IEEE AVSS 2018 | CCTV-based traffic accident dataset |
| **So-TAD** | Neurocomputing 2025 | Surveillance-oriented traffic anomaly dataset |
| **TU-DAT** | Sensors 2025 | Road traffic anomaly dataset |
| **UCF-Crime** | CVPR 2018 | General anomaly dataset containing accidents |

> **Usage**: SCAD is provided for **academic research only**. When you use the dataset or model, please cite both this repository and the corresponding paper (once accepted).

---

## 🚀 Usage Guide

### 1️⃣ Segment Raw Videos  
Use the segmentation script:

```bash
python segment.py
Before running, edit:

python
Copy code
# segment.py
input_folder  = "videos/"  
output_folder = "accident/"
This script outputs 4-second clips into anomalous/ or normal/ subfolders.

2️⃣ Prepare Train/Test Splits
Create the files:

accident_train.txt

accident_test.txt

Each line should list a clip path relative to accident/, e.g.:

bash
Copy code
anomalous/clip_001.mp4  
normal/clip_145.mp4  
3️⃣ Train the X3D Model
Run:

bash
Copy code
python train.py
Within train.py, configure:

python
Copy code
main_folder  = "accident/"  
train_split  = "accident_train.txt"  
test_split   = "accident_test.txt"
You can also adjust: input size, frame count, sampling rate, learning rate, optimizer, pretrained checkpoint, etc.

4️⃣ Evaluate Model
Run:

bash
Copy code
python test.py
Edit:

python
Copy code
dataset_folder = "accident/"  
test_split     = "accident_test.txt"  
model_path     = "84.48model.pth"
5️⃣ Real-Time Inference
Run:

bash
Copy code
python inference.py
It automatically segments incoming frames and outputs alerts when it detects anomalous (accident) events.

🧠 Model & Architecture
Backbone: X3D (efficient spatio-temporal architecture)

Framework: PyTorch

Input: RGB frame sequences

Output: Binary classification (accident / non-accident)

🔗 Related Work & Link
We build upon the STEAD anomaly detection repository, whose efficient spatio-temporal modeling inspired aspects of our design.
See more at: agao8/STEAD 
GitHub

👥 Credits & Inspirations
STEAD by Andrew Gao & Jun Liu — efficient anomaly detection foundation

X3D (Feichtenhofer et al.) — lightweight 3D CNN architecture

Original datasets: CADP, So-TAD, TU-DAT, UCF-Crime

🧾 Citation
bibtex
Copy code
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
⭐ If this work supports your research, please star the repository and help others by citing this paper when it's published.

pgsql
Copy code

If you like, I can also generate a **compact version** (for the GitHub homepage) and a **detailed document** (for `/docs/`) for you. Do you want me to prepare those?
::contentReference[oaicite:1]{index=1}
