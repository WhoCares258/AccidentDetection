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
import torch.nn.functional as F


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

# Config (MATCHED with inference)
transform_params = {
    "side_size": 182,
    "crop_size": 182,
    "num_frames": 15,
    "sampling_rate": 8,  # 30 * 4 / 30 = 4.0s
}
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

multi_clip_eval = True
num_val_clips = 5


# --------------------------
# 2. Transform (Uniform Sampling)
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
        UniformTemporalSubsample(transform_params["num_frames"]),  # ✅ uniform frame spacing
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),  # [C,T,H,W] -> [T,C,H,W]
        Normalize(mean, std),
        ShortSideScale(size=transform_params["side_size"]),
        CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
        Permute((1, 0, 2, 3))  # back to [C,T,H,W]
    ])
)


# --------------------------
# 3. Dataset
# --------------------------
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

def build_dataset(txt_file):
    with open(txt_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    labeled_paths = [
        ("accident_segmented/" + p,
         {"label": 0 if "normal" in p else 1})
        for p in paths
    ]
    return LabeledVideoDataset(
        labeled_video_paths=labeled_paths,
        clip_sampler=UniformClipSampler(clip_duration),  # ✅ same 4s uniform sampling
        transform=transform,
        decode_audio=False
    )

train_dataset = build_dataset("accident_train.txt")
val_dataset   = build_dataset("accident_test.txt")

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader   = DataLoader(val_dataset, batch_size=batch_size)


# --------------------------
# 4. Model
# --------------------------
model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
model = model.to(device)


# --------------------------
# 4.1 FLOPs check
# --------------------------
print("\n--- FLOPs Check ---")
dummy_canonical = torch.randn(1, 3, 13, 128, 128).to(device)
flops_canonical = FlopCountAnalysis(model, dummy_canonical)
print(f"GFLOPs per clip (canonical X3D-S): {flops_canonical.total() / 1e9:.2f}")

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
# 6. Validation helper (MATCHED to inference)
# --------------------------
def validate(model, loader, device, multi_clip=False, num_clips=5):
    model.eval()
    all_preds, all_labels, all_losses = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            inputs = batch["video"].to(device)
            labels = batch["label"].to(device)

            if not multi_clip:
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
            else:
                # ✅ uniform evaluation: average probs over fixed clips
                clip_probs = []
                for _ in range(num_clips):
                    outputs = model(inputs)
                    clip_probs.append(F.softmax(outputs, dim=1))
                probs = torch.stack(clip_probs, dim=0).mean(dim=0)

            loss = criterion(probs, labels)
            preds = probs.argmax(dim=1)

            all_losses.append(loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return np.mean(all_losses), acc, all_labels, all_preds


# --------------------------
# 7. Train + Validate
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
    val_loss, val_acc, val_labels, val_preds = validate(
        model, val_loader, device, multi_clip=multi_clip_eval, num_clips=num_val_clips
    )

    print(f"Epoch {epoch}/{num_epochs} "
          f"| Train Loss: {np.mean(train_losses):.4f} Acc: {train_acc:.2%} "
          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

    ckpt_path = os.path.join(save_dir, f"x3d_s_pretrained_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Saved {ckpt_path}")

# Final report
print("\nFinal Validation Report:")
print(classification_report(val_labels, val_preds, target_names=["normal", "anomalous"]))
