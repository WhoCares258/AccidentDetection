import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.v2 import CenterCrop, Normalize

from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis


# --------------------------
# 1. Setup
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

num_classes = 2
frames_per_second = 30
batch_size = 4
num_epochs = 10
lr = 1e-4
save_dir = "x3d-s-pretrained-checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Training params (your config)
transform_params = {
    "side_size": 182,
    "crop_size": 182,
    "num_frames": 30,   # your choice
    "sampling_rate": 4,
}
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]


# --------------------------
# 2. Transform
# --------------------------
class Permute(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return torch.permute(x, self.dims)

transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(transform_params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),  # [C,T,H,W] -> [T,C,H,W]
        Normalize(mean, std),
        ShortSideScale(size=transform_params["side_size"]),
        CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
        Permute((1, 0, 2, 3))  # back to [C,T,H,W]
    ])
)


# --------------------------
# 3. Datasets
# --------------------------
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

def build_dataset(txt_file):
    with open(txt_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    labeled_paths = [
        ("accident/" + p,
         {"label": 0 if "normal" in p else 1})
        for p in paths
    ]
    return LabeledVideoDataset(
        labeled_video_paths=labeled_paths,
        clip_sampler=UniformClipSampler(clip_duration),
        transform=transform,
        decode_audio=False
    )

train_dataset = build_dataset("accident_train.txt")
val_dataset   = build_dataset("accident_test.txt")

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)


# --------------------------
# 4. Model (pretrained)
# --------------------------
model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)

# Replace classification head with 2 classes
model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
model = model.to(device)


# --------------------------
# 4.1 Stable FLOPs calculation
# --------------------------
print("\n--- FLOPs Check ---")

# Canonical FLOPs (official config 13×182×182)
dummy_canonical = torch.randn(1, 3, 13, 182, 182).to(device)
flops_canonical = FlopCountAnalysis(model, dummy_canonical)
print(f"GFLOPs per clip (canonical X3D-S, 13×182×182): {flops_canonical.total() / 1e9:.2f}")

# FLOPs for your actual config (backbone only)
backbone = torch.nn.Sequential(*list(model.blocks[:-1]))
dummy_actual = torch.randn(
    1, 3, transform_params["num_frames"],
    transform_params["crop_size"], transform_params["crop_size"]
).to(device)
flops_actual = FlopCountAnalysis(backbone, dummy_actual)
print(f"Approx GFLOPs per clip (your config {transform_params['num_frames']}f {transform_params['crop_size']}x{transform_params['crop_size']}): {flops_actual.total() / 1e9:.2f}")


# --------------------------
# 5. Training setup
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# --------------------------
# 6. Train + Validate
# --------------------------
for epoch in range(1, num_epochs+1):
    model.train()
    train_losses, train_preds, train_labels = [], [], []

    print(f"\n--- Epoch {epoch}/{num_epochs} ---")
    for batch in tqdm(train_loader, desc="Training", leave=False):
        inputs = batch["video"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)

    # --- Validation ---
    model.eval()
    val_preds, val_labels, val_losses = [], [], []
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        with torch.no_grad():
            inputs = batch["video"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_losses.append(loss.item())
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)

    print(f"Epoch {epoch}/{num_epochs} "
          f"| Train Loss: {np.mean(train_losses):.4f} Acc: {train_acc:.2%} "
          f"| Val Loss: {np.mean(val_losses):.4f} Acc: {val_acc:.2%}")

    # Save checkpoint every epoch
    ckpt_path = os.path.join(save_dir, f"x3d_s_pretrained_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Saved {ckpt_path}")

# Final report
print("\nFinal Validation Report:")
print(classification_report(val_labels, val_preds, target_names=["normal", "anomalous"]))
