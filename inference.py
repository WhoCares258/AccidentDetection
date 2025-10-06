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
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips
from sklearn.metrics import confusion_matrix, classification_report

# --------------------------
# 1. Segmentation Utilities
# --------------------------
def get_subclip(video, start, end):
    if hasattr(video, "subclip"):  # MoviePy v1.x
        return video.subclip(start, end)
    elif hasattr(video, "subclipped"):  # MoviePy v2.x
        return video.subclipped(start, end)
    else:
        raise AttributeError("Neither subclip nor subclipped found in VideoFileClip")

def set_clip_duration_fps(image_clip, duration, fps):
    if hasattr(image_clip, "set_duration"):  # MoviePy v1.x
        return image_clip.set_duration(duration).set_fps(fps)
    elif hasattr(image_clip, "with_duration"):  # MoviePy v2.x
        return image_clip.with_duration(duration).with_fps(fps)
    else:
        raise AttributeError("Neither set_duration/with_duration found in ImageClip")

def segment_video(input_path, output_dir, clip_duration=4, interval=3):
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(input_path)
    video_duration = video.duration

    start = 0
    clip_index = 1
    clips = []

    while start < video_duration:
        end = min(start + clip_duration, video_duration)
        subclip = get_subclip(video, start, end)

        if (end - start) < clip_duration:
            last_frame = subclip.get_frame(subclip.duration - 0.01)
            freeze_frame = set_clip_duration_fps(
                ImageClip(last_frame),
                clip_duration - (end - start),
                video.fps
            )
            subclip = concatenate_videoclips([subclip, freeze_frame])

        output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_clip_{clip_index:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        clips.append(output_path)
        start += interval
        clip_index += 1

    video.close()
    return clips


# --------------------------
# 2. Inference Setup
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 2
frames_per_second = 30

ckpt_path = "819model.pth"

transform_params = {
    "side_size": 180,
    "crop_size": 180,
    "num_frames": 30,
    "sampling_rate": 4,
}

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

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

# Load model
model = create_x3d(
    input_clip_length=transform_params["num_frames"],
    input_crop_size=transform_params["crop_size"],
    model_num_class=num_classes
)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt)
model = model.to(device).eval()


# --------------------------
# 3. Prediction Function
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
# 4. Full Pipeline
# --------------------------
if __name__ == "__main__":
    input_video = "input.mp4"   # path to one video
    output_segments = "segmented_clips"

    # Step 1: Segment video into clips
    clips = segment_video(input_video, output_segments, clip_duration=4, interval=3)

    # Step 2: Run inference on each clip
    for clip_path in clips:
        pred, probs = predict_video(clip_path)
        if pred == 1:  # anomalous
            print(f"🚨 Anomalous: {clip_path} | Probs: {probs.round(3)}")
