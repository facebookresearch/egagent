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

from datetime import datetime, timedelta
import os
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys

# Allow running this script from inside prepare_datasources/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import EGOLIFE_ROOT

SELECTED_PERSON = "A1_JAKE"  # only person with published QA as of Dec 1, 2025
DAYS = [f"DAY{d}" for d in range(1, 8)]  # DAY1 .. DAY7

# Root for 1 fps frames (e.g. EGOLIFE_ROOT/image_1fps_A1_JAKE/)
output_dir = Path(f"{EGOLIFE_ROOT}/image_1fps_{SELECTED_PERSON}")
output_dir.mkdir(exist_ok=True)


def process_video(video_path: Path, day_dir: Path) -> None:
    """Extract 1 frame per second from a single video into day_dir."""
    video_name = video_path.stem.split("_")[-1]
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", "fps=1",
        str(day_dir / f"{video_name}_%02d.jpg"),
        "-hide_banner", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)


def process_one_day(day_name: str) -> None:
    """Process all videos for one day: extract 1 fps frames, then rename to timestamps."""
    input_dir = Path(f"{EGOLIFE_ROOT}/{SELECTED_PERSON}/{day_name}")
    day_dir = output_dir / day_name
    day_dir.mkdir(parents=True, exist_ok=True)
    mp4_files = sorted(input_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files in {input_dir}, skipping.")
        return
    for video_path in mp4_files:
        process_video(video_path, day_dir)
    rename_frames_to_timestamps(day_dir)
    print(f"Done {day_name}.")

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

def rename_frames_to_timestamps(day_dir: Path):
    """Rename the frames to have the correct timestamp."""
    for fname in day_dir.iterdir():
        if fname.suffix != ".jpg":
            continue
        stem = fname.stem  # e.g. "11094208_01"
        if "_" not in stem:
            continue
        base, idx = stem.split("_", 1)
        frame_idx = int(idx)
        start_dt = parse_time_from_video_name(base)
        frame_dt = start_dt + timedelta(seconds=frame_idx - 1)
        new_name = format_time(frame_dt) + ".jpg"
        new_path = day_dir / new_name
        if new_path != fname:
            fname.rename(new_path)
    print("Renaming complete.")

def main():
    """Extract 1 fps frames from DAY1..DAY7 in parallel, then rename to timestamp-based filenames."""
    with Pool(processes=min(len(DAYS), cpu_count())) as pool:
        pool.map(process_one_day, DAYS)
    print("Done all days.")

if __name__ == "__main__":
    main()