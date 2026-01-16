# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

SELECTED_PERSON = "A1_JAKE" # this is the only person for whom QA are published as of Dec 1, 2025.
SELECTED_DAY = "DAY7"
DATA_DIR = "" # path to egolife dataset 

INPUT_DIR = Path(f"{DATA_DIR}/EgoLife/{SELECTED_PERSON}/{SELECTED_DAY}")
OUTPUT_DIR = Path(f"{DATA_DIR}/EgoLife/image_1fps_{SELECTED_PERSON}")
OUTPUT_DIR.mkdir(exist_ok=True)

def process_video(video_path):
    video_name = video_path.stem.split("_")[-1]
    out_subdir = OUTPUT_DIR / SELECTED_DAY
    out_subdir.mkdir(exist_ok=True)
    # FFmpeg command to sample 1 frame per second
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", "fps=1",
        str(out_subdir / f"{video_name}_%02d.jpg"),
        "-hide_banner", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

def main():
    mp4_files = list(INPUT_DIR.glob("*.mp4"))
    if not mp4_files:
        print("No MP4 files found in", INPUT_DIR)
        return

    with Pool(processes=cpu_count()) as pool:
        pool.map(process_video, mp4_files)

    print("Done extracting 1 fps frames from all videos.")

if __name__ == "__main__":
    main()


from datetime import datetime, timedelta

day = int(SELECTED_DAY[3])

# Folder containing your frames
frames_dir = f"{DATA_DIR}/EgoLife/image_1fps_{SELECTED_PERSON}/DAY{day}"   # change this to your folder
output_dir = f"{DATA_DIR}/EgoLife/image_1fps_{SELECTED_PERSON}/DAY{day}"
os.makedirs(output_dir, exist_ok=True)

def parse_time_from_video_name(video_name: str) -> datetime:
    """
    Parse the base time from a video filename (without extension).
    Example: '11094208' -> datetime with hours, minutes, seconds, centiseconds
    """
    # Format: HHMMSSCC (hours, minutes, seconds, centiseconds/frames)
    h = int(video_name[0:2])
    m = int(video_name[2:4])
    s = int(video_name[4:6])
    cs = int(video_name[6:8])  # centiseconds
    return datetime(2024, 4, 15, h, m, s, cs * 10000)  # rough estimate based on Egolife Earth Day

def format_time(dt: datetime) -> str:
    """
    Convert datetime back into HHMMSSCC-style filename string.
    """
    hms = dt.strftime("%H%M%S")
    cs = f"{dt.microsecond // 10000:02d}"
    return hms + cs

def rename_videos()
    for fname in os.listdir(frames_dir):
        if not fname.endswith(".jpg"):
            continue
        
        base, idx = fname.split("_")
        frame_idx = int(idx.split(".")[0])  # e.g. "_01.jpg" -> 1
        
        # Parse start time from video name
        start_dt = parse_time_from_video_name(base)
        
        # Add (frame_idx - 1) seconds
        frame_dt = start_dt + timedelta(seconds=frame_idx - 1)
        
        # Format new filename
        new_name = format_time(frame_dt) + ".jpg"
        
        # Rename (or copy) file
        old_path = os.path.join(frames_dir, fname)
        new_path = os.path.join(output_dir, new_name)
        # print(f'{old_path} -> {new_path}')
        os.rename(old_path, new_path)
    
    print("Renaming complete.")
