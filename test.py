import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.v2 import CenterCrop, Normalize
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from pytorchvideo.models.x3d import create_x3d
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------
# 1. Setup (editable)
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 2
frames_per_second = 30

# Paths (edit here)
dataset_root = "accident_segmented"   # root folder with normal/ and anomalous/
test_file = "accident_test.txt"       # file with test video list (only paths)
ckpt_path = "819model.pth"

# Transform parameters
transform_params = {
    "side_size": 180,
    "crop_size": 180,
    "num_frames": 30,
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
        Permute((1, 0, 2, 3)),
        Normalize(mean, std),
        ShortSideScale(size=transform_params["side_size"]),
        CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
        Permute((1, 0, 2, 3))
    ])
)


# --------------------------
# 3. Load Fine-tuned Model
# --------------------------
model = create_x3d(
    input_clip_length=transform_params["num_frames"],
    input_crop_size=transform_params["crop_size"],
    model_num_class=num_classes
)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt)
model = model.to(device).eval()


# --------------------------
# 4. Prediction Function
# --------------------------
def predict_video(video_path):
    clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second
    video = EncodedVideo.from_path(video_path)
    clip = video.get_clip(start_sec=0, end_sec=clip_duration)["video"]
    clip = transform({"video": clip})["video"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(clip)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs.cpu().numpy()


# --------------------------
# 5. Load Test List & Evaluate
# --------------------------
y_true, y_pred = [], []

with open(test_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

for rel_path in lines:
    # infer label from folder name
    if rel_path.startswith("normal/"):
        label = 0
    elif rel_path.startswith("anomalous/"):
        label = 1
    else:
        print(f"[WARNING] Could not infer label from path: {rel_path}")
        continue

    path = os.path.join(dataset_root, rel_path)

    if not os.path.exists(path):
        print(f"[WARNING] Missing file: {path}")
        continue

    pred, probs = predict_video(path)

    y_true.append(label)
    y_pred.append(pred)

    print(f"{rel_path:40} | True: {'normal' if label==0 else 'anomalous':9} | "
          f"Pred: {'normal' if pred==0 else 'anomalous':9} | "
          f"Probs: {probs.round(3)}")


# --------------------------
# 6. Report Metrics
# --------------------------
print("\n=== Overall Results ===")
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"Accuracy: {accuracy:.2%}")

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["normal", "anomalous"]))

