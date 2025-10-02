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
# 1. Setup
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 2
frames_per_second = 30

transform_params = {
    "side_size": 128,
    "crop_size": 128,
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

ckpt = torch.load("x3d-s-checkpoints/819model.pth", map_location=device)
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
# 5. Evaluate on Test Folders
# --------------------------
test_dirs = {
    "normal": "accident_segmented/test/normal",
    "anomalous": "accident_segmented/test/anomalous"
}

y_true, y_pred = [], []

for label_str, folder in test_dirs.items():
    label = 0 if label_str == "normal" else 1
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        path = os.path.join(folder, fname)
        pred, probs = predict_video(path)

        y_true.append(label)
        y_pred.append(pred)

        print(f"{fname:40} | True: {label_str:9} | "
              f"Pred: {'anomalous' if pred else 'normal':9} | "
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

