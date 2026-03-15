"""Pipeline orchestrator: ties all stages together."""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import torch
from ultralytics import YOLO

from .clip_extractor import extract_clip
from .config import PipelineConfig
from .day_night import classify_video_day_night
from .detect_and_track import TrackData, detect_and_track, get_video_metadata
from .event_detector import WaymoEvent, extract_events
from .track_interpolator import interpolate_track_gaps
from .waymo_classifier import WaymoClassifier


def _resolve_video_paths(patterns: list[str], base_dir: str) -> list[str]:
    """Expand glob patterns to actual video file paths."""
    paths = []
    for pattern in patterns:
        # Make relative patterns absolute
        if not Path(pattern).is_absolute():
            pattern = str(Path(base_dir) / pattern)
        matches = sorted(glob.glob(pattern))
        paths.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in paths:
        rp = str(Path(p).resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


def _load_classifier(config: PipelineConfig, base_dir: str = ".") -> object:
    """Load the appropriate classifier based on detection mode."""
    if config.detection_mode == "finetuned":
        # In finetuned mode, YOLO handles classification directly
        return None

    # Two-stage mode: try to load the trained classifier
    classifier_path = config.classifier_weights
    if not Path(classifier_path).is_absolute():
        classifier_path = str(Path(base_dir) / classifier_path)

    if not Path(classifier_path).exists():
        raise FileNotFoundError(
            f"Classifier weights not found: {classifier_path}\n"
            f"Train the classifier first with: python training/train_classifier.py\n"
            f"Or provide a valid path via classifier_weights in config.yaml"
        )
    return WaymoClassifier(classifier_path, device=config.device)


def _log_events(
    events: list[tuple[WaymoEvent, str, str]],
    log_path: str,
):
    """Write detection log CSV."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_file",
                "time_of_day",
                "track_id",
                "start_frame",
                "end_frame",
                "duration_frames",
                "avg_confidence",
                "avg_waymo_score",
                "waymo_ratio",
            ]
        )
        for event, tod, clip_path in events:
            writer.writerow(
                [
                    clip_path,
                    tod,
                    event.track_id,
                    event.start_frame,
                    event.end_frame,
                    event.duration_frames,
                    f"{event.avg_confidence:.3f}",
                    f"{event.avg_waymo_score:.3f}",
                    f"{event.waymo_ratio:.3f}",
                ]
            )


def run_pipeline(config: PipelineConfig, base_dir: str = "."):
    """
    Run the full Waymo detection pipeline.

    Steps:
    1. Discover input videos
    2. For each video: detect + track vehicles and pedestrians
    3. Interpolate track gaps for smoother bounding boxes
    4. Convert tracks to Waymo events
    5. Classify day/night
    6. Extract annotated clips with AV/HDV/PED labels
    7. Log results
    """
    # Resolve video paths
    video_paths = _resolve_video_paths(config.input_videos, base_dir)
    if not video_paths:
        print("ERROR: No input videos found matching patterns:")
        for p in config.input_videos:
            print(f"  {p}")
        return

    print(f"Found {len(video_paths)} video(s) to process")

    # Load models once — reuse across all videos to avoid GPU memory leaks
    classifier = _load_classifier(config, base_dir=base_dir)
    yolo_model = YOLO(config.model_weights)

    all_logged_events = []
    faulty_clips = []
    total_clips = 0

    for vid_idx, video_path in enumerate(video_paths, 1):
        print(f"\n[{vid_idx}/{len(video_paths)}] Processing: {video_path}")

        # Get video metadata
        try:
            meta = get_video_metadata(video_path)
        except ValueError as e:
            print(f"  ERROR: {e}, skipping")
            continue

        print(
            f"  Video: {meta['width']}x{meta['height']}, "
            f"{meta['fps']:.1f}fps, "
            f"{meta['duration_sec']:.0f}s ({meta['frame_count']} frames)"
        )

        # Stage 1: Detection + Tracking (now with AV/HDV/PED labels)
        print("  Stage 1: Detecting and tracking vehicles + pedestrians...")
        tracks = detect_and_track(video_path, config, classifier=classifier, model=yolo_model)

        # Free fragmented GPU memory between videos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        # Stage 2: Interpolate track gaps (smoother bounding boxes)
        if config.interpolate_gaps:
            print("  Stage 2: Interpolating track gaps...")
            tracks = interpolate_track_gaps(tracks, config)

        # Stage 3: Event detection (find Waymo enter/exit events)
        print("  Stage 3: Extracting Waymo events...")
        events = extract_events(tracks, meta["fps"], config)
        print(f"  Found {len(events)} Waymo event(s)")

        if not events:
            print("  No Waymo events detected in this video")
            continue

        # Stage 4: Extract clips
        print("  Stage 4: Extracting annotated clips...")
        video_stem = Path(video_path).stem

        # Classify day/night once for the whole video
        time_of_day = classify_video_day_night(
            video_path, meta["duration_sec"], config
        )

        for i, event in enumerate(events, 1):
            duration = event.duration_sec(meta["fps"])

            # Build output path
            clip_name = f"waymo_{video_stem}_clip{i:03d}_{time_of_day}.mp4"
            output_subdir = Path(config.output_dir) / time_of_day
            output_path = str(output_subdir / clip_name)

            print(
                f"    Clip {i}/{len(events)}: "
                f"frames {event.start_frame}-{event.end_frame} "
                f"({duration:.1f}s), {time_of_day}, "
                f"waymo_score={event.avg_waymo_score:.2f}"
            )

            success = extract_clip(
                video_path, event, meta["fps"], output_path, config,
                video_duration_sec=meta["duration_sec"],
            )

            if success:
                total_clips += 1
                all_logged_events.append((event, time_of_day, clip_name))

                # Flag clips with unexpected duration
                if duration < config.min_expected_clip_sec:
                    reason = f"too short ({duration:.1f}s < {config.min_expected_clip_sec}s)"
                    faulty_clips.append((clip_name, time_of_day, duration, reason))
                    print(f"    WARNING: {reason}")
                elif duration > config.max_expected_clip_sec:
                    reason = f"too long ({duration:.1f}s > {config.max_expected_clip_sec}s)"
                    faulty_clips.append((clip_name, time_of_day, duration, reason))
                    print(f"    WARNING: {reason}")
            else:
                print(f"    FAILED to extract clip {i}")

    # Write detection log
    if all_logged_events:
        log_path = str(Path(config.output_dir) / "detection_log.csv")
        _log_events(all_logged_events, log_path)
        print(f"\nDetection log written to: {log_path}")

    # Write faulty clips log
    if faulty_clips:
        faulty_path = str(Path(config.output_dir) / "faulty_clips.csv")
        Path(faulty_path).parent.mkdir(parents=True, exist_ok=True)
        with open(faulty_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_file", "time_of_day", "duration_sec", "reason"])
            for clip_name, tod, dur, reason in faulty_clips:
                writer.writerow([clip_name, tod, f"{dur:.1f}", reason])
        print(f"\nWARNING: {len(faulty_clips)} faulty clip(s) flagged — see {faulty_path}")

    print(f"\nPipeline complete: {total_clips} clip(s) exported")
