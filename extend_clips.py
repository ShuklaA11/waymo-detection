"""Re-extract clips with extra seconds added to the beginning and/or end.

Reads detection_log.csv to find the original frame ranges, then re-cuts
each clip from the source video with the requested extra padding.

Usage:
    # Add 3 seconds to the beginning of all clips
    python extend_clips.py --output-dir output/sjb --video-dir "D:/Video_recording/..." --add-before 3

    # Add 5 seconds to the end of all clips
    python extend_clips.py --output-dir output/sjb --video-dir "D:/Video_recording/..." --add-after 5

    # Add 2 seconds to both sides
    python extend_clips.py --output-dir output/sjb --video-dir "D:/Video_recording/..." --add-before 2 --add-after 2

    # Only extend specific clips (by filename substring)
    python extend_clips.py --output-dir output/sjb --video-dir "D:/..." --add-before 3 --filter clip001

    # Preview what would happen without re-extracting
    python extend_clips.py --output-dir output/sjb --video-dir "D:/..." --add-before 3 --dry-run

    # Save extended clips to a separate folder instead of overwriting
    python extend_clips.py --output-dir output/sjb --video-dir "D:/..." --add-before 3 --save-to output/sjb_extended
"""

from __future__ import annotations

import argparse
import csv
import glob
import shutil
import subprocess
import sys
from pathlib import Path


def find_source_video(video_stem: str, video_dirs: list[str]) -> str | None:
    """Find the source video file matching a stem across search directories."""
    for video_dir in video_dirs:
        # Direct glob for the stem with any extension
        for ext in ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mkv", "*.mov"):
            for match in glob.glob(str(Path(video_dir) / "**" / ext), recursive=True):
                if Path(match).stem == video_stem:
                    return match
    return None


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        return 0.0


def get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        # r_frame_rate is a fraction like "25/1"
        num, den = result.stdout.strip().split("/")
        return float(num) / float(den)
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        return 25.0  # fallback


def extract_video_stem_from_clip(clip_name: str) -> str:
    """Extract the source video stem from a clip filename.

    Clip format: waymo_{video_stem}_clip{NNN}_{day|night}.mp4
    """
    # Remove "waymo_" prefix and ".mp4" suffix
    base = clip_name.removeprefix("waymo_").removesuffix(".mp4")
    # Find the last occurrence of _clip followed by digits and _day/_night
    # Work backwards: remove _day or _night, then _clipNNN
    for suffix in ("_day", "_night"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    # Remove _clipNNN
    idx = base.rfind("_clip")
    if idx != -1:
        return base[:idx]
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract clips with extra seconds at the beginning/end."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory containing detection_log.csv and day/night folders (e.g., output/sjb)"
    )
    parser.add_argument(
        "--video-dir", required=True, nargs="+",
        help="Directory(ies) to search for source videos (searched recursively)"
    )
    parser.add_argument(
        "--add-before", type=float, default=0.0,
        help="Extra seconds to add before the clip start (default: 0)"
    )
    parser.add_argument(
        "--add-after", type=float, default=0.0,
        help="Extra seconds to add after the clip end (default: 0)"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process clips whose filename contains this substring"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without actually re-extracting"
    )
    parser.add_argument(
        "--save-to", type=str, default=None,
        help="Save extended clips to this directory instead of overwriting originals"
    )
    # Output encoding options (match pipeline defaults)
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    parser.add_argument("--width", type=int, default=1920, help="Output width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output height (default: 1080)")
    parser.add_argument("--crf", type=int, default=23, help="H.264 CRF (default: 23)")

    args = parser.parse_args()

    if args.add_before == 0 and args.add_after == 0:
        print("ERROR: Specify at least one of --add-before or --add-after")
        sys.exit(1)

    # Check ffmpeg
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ERROR: ffmpeg/ffprobe not found on PATH")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    log_path = output_dir / "detection_log.csv"
    if not log_path.exists():
        print(f"ERROR: detection_log.csv not found in {output_dir}")
        sys.exit(1)

    # Read detection log
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No clips found in detection_log.csv")
        return

    # Apply filter
    if args.filter:
        rows = [r for r in rows if args.filter in r["clip_file"]]
        if not rows:
            print(f"No clips match filter '{args.filter}'")
            return

    print(f"Found {len(rows)} clip(s) to extend")
    print(f"  Adding {args.add_before:.1f}s before, {args.add_after:.1f}s after")
    if args.save_to:
        print(f"  Saving to: {args.save_to}")
    else:
        print(f"  Overwriting originals in: {output_dir}")
    print()

    # Cache: video_stem -> (source_path, fps, duration)
    video_cache: dict[str, tuple[str, float, float]] = {}
    success_count = 0
    fail_count = 0

    for row in rows:
        clip_name = row["clip_file"]
        time_of_day = row["time_of_day"]
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])

        video_stem = extract_video_stem_from_clip(clip_name)

        # Look up or find source video
        if video_stem not in video_cache:
            source_path = find_source_video(video_stem, args.video_dir)
            if source_path is None:
                print(f"  SKIP {clip_name}: source video '{video_stem}' not found")
                fail_count += 1
                continue
            fps = get_video_fps(source_path)
            duration = get_video_duration(source_path)
            video_cache[video_stem] = (source_path, fps, duration)
            print(f"  Found source: {source_path} ({fps:.1f} fps, {duration:.0f}s)")

        source_path, fps, video_duration = video_cache[video_stem]

        # Calculate new timestamps
        # Original clip timing (these are the event frames, before any existing padding)
        orig_start_sec = start_frame / fps
        orig_end_sec = end_frame / fps

        # The pipeline already added padding_before_sec (3s) and padding_after_sec (3s)
        # We add our extra on top of that existing padding
        # So the actual clip starts at: orig_start_sec - existing_padding - add_before
        # Read existing padding from config? No — just extend from the event boundaries
        # with the total desired padding.
        existing_padding_before = 3.0  # from config.yaml default
        existing_padding_after = 3.0

        new_start_sec = max(0, orig_start_sec - existing_padding_before - args.add_before)
        new_end_sec = orig_end_sec + existing_padding_after + args.add_after
        if video_duration > 0:
            new_end_sec = min(new_end_sec, video_duration)

        new_duration = new_end_sec - new_start_sec

        # Determine output path
        if args.save_to:
            out_dir = Path(args.save_to) / time_of_day
        else:
            out_dir = output_dir / time_of_day
        out_path = out_dir / clip_name

        orig_duration = (end_frame - start_frame) / fps
        print(
            f"  {clip_name}: event {orig_start_sec:.1f}-{orig_end_sec:.1f}s "
            f"({orig_duration:.1f}s) -> clip {new_start_sec:.1f}-{new_end_sec:.1f}s "
            f"({new_duration:.1f}s)"
        )

        if args.dry_run:
            success_count += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{new_start_sec:.3f}",
            "-i", source_path,
            "-t", f"{new_duration:.3f}",
            "-vf", f"scale={args.width}:{args.height}",
            "-r", str(args.fps),
            "-c:v", "libx264",
            "-crf", str(args.crf),
            "-preset", "medium",
            "-an",
            "-movflags", "+faststart",
            str(out_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: {e.stderr[:200]}")
            fail_count += 1
        except subprocess.TimeoutExpired:
            print(f"    ERROR: ffmpeg timed out")
            fail_count += 1

    print(f"\nDone: {success_count} succeeded, {fail_count} failed")
    if args.dry_run:
        print("(dry run — no files were modified)")


if __name__ == "__main__":
    main()
