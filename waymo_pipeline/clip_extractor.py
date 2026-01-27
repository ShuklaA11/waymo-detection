"""Extract video clips using ffmpeg for each Waymo event."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .event_detector import WaymoEvent


def _check_ffmpeg():
    """Verify ffmpeg is installed and on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH.\n"
            "Install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: download from https://ffmpeg.org/download.html and add to PATH"
        )


def extract_clip(
    video_path: str,
    event: WaymoEvent,
    source_fps: float,
    output_path: str,
    config: PipelineConfig,
    video_duration_sec: float | None = None,
) -> bool:
    """
    Extract a clip from the source video for a Waymo event.

    Uses ffmpeg with fast seek (-ss before -i) for efficiency.
    Re-encodes to 1080p, 30fps, H.264 MP4.

    Returns True on success, False on failure.
    """
    # Calculate timestamps with padding
    start_sec = max(0, (event.start_frame / source_fps) - config.padding_before_sec)
    end_sec = (event.end_frame / source_fps) + config.padding_after_sec

    # Clamp to video duration if known
    if video_duration_sec is not None:
        end_sec = min(end_sec, video_duration_sec)
    duration = end_sec - start_sec

    # Safety clamp: max duration + padding
    max_total = config.max_clip_duration_sec + config.padding_before_sec + config.padding_after_sec
    duration = min(duration, max_total)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    _check_ffmpeg()

    cmd = [
        "ffmpeg",
        "-y",                                       # overwrite
        "-ss", f"{start_sec:.3f}",                  # fast seek (before -i)
        "-i", video_path,
        "-t", f"{duration:.3f}",                    # duration
        "-vf", f"scale={config.output_width}:{config.output_height}",
        "-r", str(config.output_fps),               # 30fps output
        "-c:v", config.output_codec,
        "-crf", str(config.output_crf),
        "-preset", "medium",
        "-an",                                       # no audio
        "-movflags", "+faststart",                   # web-friendly MP4
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR extracting clip: {e.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ERROR: ffmpeg timed out extracting clip to {output_path}")
        return False
