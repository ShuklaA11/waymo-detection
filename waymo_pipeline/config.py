"""Pipeline configuration with Pydantic validation and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator


def detect_device() -> str:
    """Auto-detect the best available compute device: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PipelineConfig(BaseModel):
    """All tunable parameters for the Waymo detection pipeline."""

    # --- Paths ---
    input_videos: list[str] = ["data/*.mp4"]
    output_dir: str = "output"
    model_weights: str = "yolo26s.pt"
    classifier_weights: str = "models/waymo_classifier.pth"

    # --- Detection mode ---
    # "two_stage": pretrained YOLO + secondary classifier
    # "finetuned": single fine-tuned YOLO with waymo class
    detection_mode: str = "two_stage"

    # --- YOLO detection ---
    vehicle_classes: list[int] = [2, 5, 7]  # COCO: car, bus, truck
    pedestrian_classes: list[int] = [0]  # COCO: person
    detection_confidence: float = 0.4

    # --- Spatial filtering ---
    min_y_fraction: float = 0.15  # ignore detections in top 15% of frame
    min_bbox_area: float = 500.0  # minimum bbox area in pixels
    max_aspect_ratio: float = 4.0  # max height/width ratio (filters poles)
    max_bbox_area_fraction: float = 0.08  # max bbox as fraction of frame (filters glare)

    # --- Cross-class NMS ---
    cross_class_iou_threshold: float = 0.5

    # --- Track deduplication ---
    track_dedup_iou_threshold: float = 0.3  # IoU to consider two tracks as duplicates

    # --- Waymo classifier ---
    classifier_confidence: float = 0.75
    crop_roof_extension: float = 0.2  # extend crop upward by 20% of bbox height

    # --- ByteTrack ---
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 60  # frames to keep lost tracks alive

    # --- Event detection ---
    min_track_frames: int = 8
    gap_tolerance_frames: int = 50  # ~2 sec at 25fps
    max_clip_duration_sec: float = 60.0
    padding_before_sec: float = 3.0
    padding_after_sec: float = 3.0
    waymo_frame_ratio: float = 0.4  # fraction of track frames classified as waymo
    dedup_overlap_threshold_sec: float = 5.0  # merge events closer than this (seconds)

    # --- Output encoding ---
    output_fps: int = 30
    output_width: int = 1920
    output_height: int = 1080
    output_codec: str = "libx264"
    output_crf: int = 23

    # --- Day/Night ---
    day_night_luminance_threshold: float = 80.0

    # --- Track interpolation ---
    interpolate_gaps: bool = True
    max_interpolation_gap: int = 30  # only interpolate gaps <= N frames

    # --- Processing ---
    device: str = "auto"
    process_every_n_frames: int = 1  # set to 2 to skip frames for speed
    max_frames: Optional[int] = None  # limit frames processed (for testing)

    @field_validator("detection_mode")
    @classmethod
    def validate_detection_mode(cls, v: str) -> str:
        if v not in ("two_stage", "finetuned"):
            raise ValueError(f"detection_mode must be 'two_stage' or 'finetuned', got '{v}'")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        import torch

        if v == "auto":
            resolved = detect_device()
            print(f"Auto-detected device: {resolved}")
            return resolved
        if v == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("WARNING: MPS not available, falling back to CPU")
            return "cpu"
        if v == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            return "cpu"
        return v


def load_config(config_path: Optional[str] = None, **overrides) -> PipelineConfig:
    """Load config from YAML file with optional overrides."""
    data = {}
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return PipelineConfig(**data)
