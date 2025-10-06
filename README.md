📘 Overview
Current anomaly detection benchmarks often suffer from scene duplication, annotation artifacts, and cross-partition overlaps — issues that inflate model performance.
SCAD addresses these flaws by providing a curated, transparent, and ethically cleaned dataset that allows true benchmarking in accident detection.
🔑 Highlights
🧹 Clean & verified dataset — 513 labeled accident and non-accident clips
⚙️ Lightweight 3D model — X3D-based for efficient training on modest GPUs
🧪 End-to-end pipeline — segmentation → training → evaluation → inference
🔍 Reproducible splits — zero cross-contamination between train/test data
🎬 Dataset Description
The SCAD Dataset was meticulously constructed from multiple public surveillance datasets. Each raw video was segmented into 4-second clips (with 3-second overlap) to ensure complete scene coverage while maintaining clip balance. All redundant and overlapping frames were removed to ensure benchmarking purity.
Source	Reference	Role in SCAD
CADP	IEEE AVSS 2018	CCTV-based traffic accident dataset
So-TAD	Neurocomputing 2025	Surveillance-oriented traffic anomaly dataset
TU-DAT	Sensors 2025	Road traffic anomaly dataset
UCF-Crime	CVPR 2018	General anomaly dataset including accidents
⚠️ Academic Use Only
SCAD is intended for research and benchmarking purposes under academic usage terms.
Please cite this repository and the associated paper once published.
🚀 Usage Guide
🧩 1. Segment Raw Videos
Generate 4-second overlapping clips from long videos.
code
Bash
python segment.py
Before running, adjust your paths inside segment.py:
code
Python
input_folder  = "videos/"
output_folder = "accident/"
🧾 2. Prepare Train/Test Splits
Create two text files in your main dataset folder:
code
Code
accident_train.txt
accident_test.txt
Each line should specify the relative path of a clip, for example:
code
Bash
anomalous/clip_001.mp4
normal/clip_145.mp4
🧠 3. Train the X3D Model
Fine-tune the pretrained X3D model on SCAD:
code
Bash
python train.py
Edit these variables before training:
code
Python
main_folder  = "accident/"
train_split   = "accident_train.txt"
test_split   = "accident_test.txt"
You can also modify:
Input resolution
Frame sampling rate
Optimizer and learning rate
Pretrained checkpoint path
🧪 4. Evaluate Model Performance
code
Bash
python test.py
Edit inside test.py:
code
Python
dataset_folder = "accident/"
test_split     = "accident_test.txt"
model_path     = "84.48model.pth"
⚡ 5. Real-Time Inference
Run detection on live or recorded video:
code
Bash
python inference.py
The script continuously processes frames and prints an alert when anomalous activity (accident) is detected.
🧠 Model & Architecture
Component	Description
Backbone	X3D (efficient spatio-temporal CNN by Facebook AI)
Framework	PyTorch
Input	RGB frame sequences
Output	Binary classification (accident / non-accident)
Two pretrained models are included for immediate benchmarking:
Model	Resolution	Frames	Description
81.9model.pth	128×128	30	Baseline model
84.48model.pth	182×182	25	Enhanced configuration
🔗 Related Work
The SCAD implementation draws inspiration from prior work in spatio-temporal event detection.
Notably, it extends ideas from the STEAD repository by A. Gao, adapting them for curated accident detection and fair benchmarking.
👥 Credits & Acknowledgments
STEAD by Andrew Gao & Jun Liu — foundational reference for anomaly detection efficiency
X3D by C. Feichtenhofer et al. (Facebook AI) — compact 3D CNN backbone
Source datasets: CADP, So-TAD, TU-DAT, UCF-Crime
🧾 Citation
If you use this work, please cite as follows:
code
Bibtex
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
⭐ If this repository helps your research, please consider giving it a star!
Help promote open, reproducible benchmarking in surveillance accident detection.
