import torch
import numpy as np
from tqdm import tqdm
import time, os
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.v2 import CenterCrop, Normalize
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from pytorchvideo.models.x3d import create_x3d


# --------------------------
# 1. Setup
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 2
frames_per_second = 30

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

transform_params = {
    "side_size": 128,
    "crop_size": 128,
    "num_frames": 30,
    "sampling_rate": 4,
}


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

ckpt = torch.load("x3d-s-checkpoints/x3d_s_epoch7.pth", map_location=device)
model.load_state_dict(ckpt)
model = model.to(device).eval()

# Remove classification head -> backbone only
del model.blocks[-1]


# --------------------------
# 4. Dataset
# --------------------------
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / frames_per_second

all_list = list(open("accident_train.txt")) + list(open("accident_test.txt"))
all_list = [
    ('accident_segmented/' + path.strip(),
     {"label": 0 if "normal" in path else 1,
      "video_label": 'finetuned_accident/' + path.strip()})
    for path in all_list
    if not os.path.isfile('finetuned_accident/' + path.strip()[:-3] + 'npy')
]

dataset = LabeledVideoDataset(
    labeled_video_paths=all_list,
    clip_sampler=UniformClipSampler(clip_duration),
    transform=transform,
    decode_audio=False
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# --------------------------
# 5. Feature Extraction
# --------------------------
start_time = time.time()
clip_count = 0

label = None
current = None

for inputs in tqdm(loader, desc="Extracting features"):
    feats = model(inputs['video'].to(device)).detach().cpu().numpy()
    clip_count += feats.shape[0]

    for i, f in enumerate(feats):
        if inputs['video_label'][i][:-3] != label:
            # save previous video if done
            if label is not None:
                os.makedirs(os.path.dirname(label), exist_ok=True)
                np.save(label + 'npy', current.squeeze())
            # start new video
            label = inputs['video_label'][i][:-3]
            current = f[None, ...]
        else:
            # pool across clips (max pooling)
            current = np.max(
                np.concatenate((current, f[None, ...]), axis=0),
                axis=0
            )[None, ...]

# save last one
np.save(label + 'npy', current.squeeze())

# Report timing
elapsed = time.time() - start_time
print(f"\nProcessed {clip_count} clips in {elapsed:.2f}s "
      f"({elapsed/clip_count:.3f} s/clip)")
print("✅ Features saved in finetuned_accident/")
