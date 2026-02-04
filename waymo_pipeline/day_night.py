"""Day/night classification based on frame luminance."""

from __future__ import annotations

import cv2
import numpy as np

from .config import PipelineConfig


def classify_day_night(
    video_path: str,
    start_sec: float,
    end_sec: float,
    config: PipelineConfig,
) -> str:
    """
    Classify a video segment as 'day' or 'night' by sampling frame luminance.

    Samples 5 evenly-spaced frames from the segment, computes mean grayscale
    luminance, and thresholds against config.day_night_luminance_threshold.

    Returns "day" or "night".
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video for day/night classification, defaulting to 'day'")
        return "day"

    sample_times = np.linspace(start_sec, end_sec, 5)
    luminances = []

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luminances.append(float(gray.mean()))

    cap.release()

    if not luminances:
        print(f"  WARNING: Could not sample frames for day/night, defaulting to 'day'")
        return "day"

    avg_luminance = np.mean(luminances)

    if avg_luminance >= config.day_night_luminance_threshold:
        return "day"
    else:
        return "night"


def classify_video_day_night(
    video_path: str,
    duration_sec: float,
    config: PipelineConfig,
) -> str:
    """
    Classify an entire video as 'day' or 'night' by sampling frames across
    the full duration. Use this once per video instead of per-event.

    Samples 10 evenly-spaced frames across the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open video for day/night classification, defaulting to 'day'")
        return "day"

    sample_times = np.linspace(0, duration_sec, 10)
    luminances = []

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luminances.append(float(gray.mean()))

    cap.release()

    if not luminances:
        print(f"  WARNING: Could not sample frames for day/night, defaulting to 'day'")
        return "day"

    avg_luminance = np.mean(luminances)

    if avg_luminance >= config.day_night_luminance_threshold:
        return "day"
    else:
        return "night"
