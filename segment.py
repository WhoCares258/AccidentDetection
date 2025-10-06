from moviepy import VideoFileClip, ImageClip, concatenate_videoclips
import os


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

    while start < video_duration:
        end = min(start + clip_duration, video_duration)
        subclip = get_subclip(video, start, end)

        # If clip is shorter than desired duration, pad with last frame
        if (end - start) < clip_duration:
            last_frame = subclip.get_frame(subclip.duration - 0.01)
            freeze_frame = set_clip_duration_fps(
                ImageClip(last_frame),
                clip_duration - (end - start),
                video.fps
            )
            subclip = concatenate_videoclips([subclip, freeze_frame])

        # Store all clips directly in the main output folder
        output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_clip_{clip_index:03d}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"Saved: {output_path}")

        start += interval
        clip_index += 1

    video.close()


def process_input(input_path, output_folder, clip_duration=4, interval=3):
    if os.path.isfile(input_path):
        segment_video(input_path, output_folder, clip_duration, interval)

    elif os.path.isdir(input_path):
        # Process videos in folder (including subfolders) but store outputs together
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    file_path = os.path.join(root, file)
                    segment_video(file_path, output_folder, clip_duration, interval)


if __name__ == "__main__":
    input_path = "scfd"  # Change this to a video file or folder path
    output_folder = "output_clips"
    process_input(input_path, output_folder)
