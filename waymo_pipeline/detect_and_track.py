"""Stage 1: YOLO26 vehicle detection + ByteTrack tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import PipelineConfig


# Object label constants
LABEL_AV = "AV"
LABEL_HDV = "HDV"
LABEL_PED = "PED"


@dataclass
class Detection:
    """A single detection in one frame."""

    frame_num: int
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    label: str = LABEL_HDV  # AV, HDV, or PED
    is_waymo: bool = False
    waymo_score: float = 0.0


@dataclass
class TrackData:
    """Accumulated data for a single tracked object."""

    track_id: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_num

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_num

    @property
    def frame_count(self) -> int:
        return len(self.detections)

    @property
    def waymo_frames(self) -> int:
        return sum(1 for d in self.detections if d.is_waymo)

    @property
    def waymo_ratio(self) -> float:
        if not self.detections:
            return 0.0
        return self.waymo_frames / len(self.detections)

    @property
    def avg_waymo_score(self) -> float:
        scores = [d.waymo_score for d in self.detections]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_confidence(self) -> float:
        confs = [d.confidence for d in self.detections]
        return sum(confs) / len(confs) if confs else 0.0

    @property
    def dominant_label(self) -> str:
        """Most common label across all detections in this track."""
        if not self.detections:
            return LABEL_HDV
        from collections import Counter
        counts = Counter(d.label for d in self.detections)
        return counts.most_common(1)[0][0]


def crop_bbox(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    roof_extension: float = 0.2,
) -> np.ndarray:
    """Crop bounding box from frame, extending upward to capture LIDAR dome."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_h = y2 - y1

    # Extend upward for roof-mounted sensor
    y1_ext = max(0, y1 - int(box_h * roof_extension))

    # Clamp to frame bounds
    x1 = max(0, x1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)

    crop = frame[y1_ext:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _apply_spatial_filter(
    detections: list[Detection],
    frame_height: int,
    config: PipelineConfig,
) -> list[Detection]:
    """
    Remove false positives caused by streetlights/poles/glare.

    Filters:
    1. Detections in the top portion of frame (min_y_fraction)
    2. Detections with bbox area below min_bbox_area
    3. Detections with extreme aspect ratios (tall thin poles)
    4. Detections with bbox area exceeding max fraction of frame (glare/phantom boxes)
    """
    min_y = frame_height * config.min_y_fraction
    frame_area = frame_height * (frame_height * 16 / 9)  # approximate frame area
    max_area = frame_area * config.max_bbox_area_fraction
    filtered = []

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Filter 1: top of frame (streetlights)
        if y2 < min_y:
            continue

        # Filter 2: too small
        if area < config.min_bbox_area:
            continue

        # Filter 3: extreme aspect ratio (poles/posts)
        if w > 0 and (h / w) > config.max_aspect_ratio:
            continue

        # Filter 4: too large (headlight glare / phantom detections)
        if area > max_area:
            continue

        filtered.append(det)

    return filtered


def _apply_cross_class_nms(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """
    Suppress duplicate boxes across different YOLO classes.

    When the same object gets detected as both "car" and "truck",
    keep only the higher-confidence detection.
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []

    for det in sorted_dets:
        is_duplicate = False
        for kept in keep:
            if _compute_iou(det.bbox, kept.bbox) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(det)

    return keep


def _containment_ratio(box_a: tuple, box_b: tuple) -> float:
    """Fraction of box_a's area that is inside box_b."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    if area_a == 0:
        return 0.0
    return inter / area_a


def _deduplicate_tracks(
    tracks: dict[int, TrackData],
    iou_threshold: float,
) -> dict[int, TrackData]:
    """
    Remove duplicate tracks that follow the same object.

    When ByteTrack assigns two track IDs to the same vehicle, we detect this
    by checking if two tracks have high spatial overlap on frames where both
    are present. We keep the track with more detections (more stable tracking).
    """
    track_ids = list(tracks.keys())
    to_remove = set()

    for i in range(len(track_ids)):
        if track_ids[i] in to_remove:
            continue
        track_a = tracks[track_ids[i]]

        for j in range(i + 1, len(track_ids)):
            if track_ids[j] in to_remove:
                continue
            track_b = tracks[track_ids[j]]

            # Check temporal overlap: do these tracks co-exist?
            a_frames = {d.frame_num: d for d in track_a.detections}
            b_frames = {d.frame_num: d for d in track_b.detections}
            shared_frames = set(a_frames.keys()) & set(b_frames.keys())

            if len(shared_frames) < 3:
                continue

            # Check spatial overlap on shared frames
            overlap_count = 0
            for fn in shared_frames:
                iou = _compute_iou(a_frames[fn].bbox, b_frames[fn].bbox)
                containment = max(
                    _containment_ratio(a_frames[fn].bbox, b_frames[fn].bbox),
                    _containment_ratio(b_frames[fn].bbox, a_frames[fn].bbox),
                )
                if iou > iou_threshold or containment > 0.7:
                    overlap_count += 1

            overlap_ratio = overlap_count / len(shared_frames)
            if overlap_ratio > 0.5:
                # Remove the track with fewer detections (less stable)
                if track_a.frame_count >= track_b.frame_count:
                    to_remove.add(track_ids[j])
                else:
                    to_remove.add(track_ids[i])
                    break  # track_a is removed, stop comparing it

    for tid in to_remove:
        del tracks[tid]

    return tracks


def _stitch_broken_tracks(
    tracks: dict[int, TrackData],
    max_gap_frames: int = 60,
    max_distance: float = 200.0,
) -> dict[int, TrackData]:
    """
    Stitch tracks that are likely the same vehicle re-acquired with a new ID.

    When ByteTrack loses a vehicle and re-detects it shortly after, it gets
    a new track ID. This function merges sequential tracks of the same label
    that end and start close in time and space.
    """
    # Group tracks by label
    label_groups: dict[str, list[int]] = {}
    for tid, track in tracks.items():
        label = track.dominant_label
        label_groups.setdefault(label, []).append(tid)

    merged_into: dict[int, int] = {}  # tid -> target tid it was merged into

    for label, tids in label_groups.items():
        # Sort by start frame
        sorted_tids = sorted(tids, key=lambda t: tracks[t].start_frame)

        for i in range(len(sorted_tids)):
            tid_a = sorted_tids[i]
            if tid_a in merged_into:
                continue
            track_a = tracks[tid_a]

            for j in range(i + 1, len(sorted_tids)):
                tid_b = sorted_tids[j]
                if tid_b in merged_into:
                    continue
                track_b = tracks[tid_b]

                # Check temporal gap: track_a ends, track_b starts
                gap = track_b.start_frame - track_a.end_frame
                if gap < 0 or gap > max_gap_frames:
                    continue

                # Check spatial proximity: last bbox of A vs first bbox of B
                last_a = track_a.detections[-1].bbox
                first_b = track_b.detections[0].bbox
                center_a = ((last_a[0] + last_a[2]) / 2, (last_a[1] + last_a[3]) / 2)
                center_b = ((first_b[0] + first_b[2]) / 2, (first_b[1] + first_b[3]) / 2)
                dist = ((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2) ** 0.5

                if dist <= max_distance:
                    # Merge B into A
                    track_a.detections.extend(track_b.detections)
                    track_a.detections.sort(key=lambda d: d.frame_num)
                    # Update track_id on merged detections
                    for det in track_a.detections:
                        det.track_id = tid_a
                    merged_into[tid_b] = tid_a

    for tid in merged_into:
        del tracks[tid]

    return tracks


def _merge_concurrent_av_tracks(
    tracks: dict[int, TrackData],
) -> dict[int, TrackData]:
    """
    When multiple AV tracks overlap in time, keep only the longest one.

    There is only one Waymo vehicle, so concurrent AV tracks are always
    ByteTrack splitting the same car into multiple IDs.
    """
    av_tracks = [
        (tid, t) for tid, t in tracks.items() if t.dominant_label == LABEL_AV
    ]
    if len(av_tracks) <= 1:
        return tracks

    # Sort by frame count descending — keep the longest
    av_tracks.sort(key=lambda x: x[1].frame_count, reverse=True)

    # For each pair, check if the shorter track overlaps temporally with a longer one
    keep_av = {av_tracks[0][0]}
    for i in range(1, len(av_tracks)):
        tid_i, track_i = av_tracks[i]
        i_frames = set(d.frame_num for d in track_i.detections)

        is_concurrent = False
        for kept_tid in keep_av:
            kept_frames = set(d.frame_num for d in tracks[kept_tid].detections)
            shared = i_frames & kept_frames
            if len(shared) >= 3:
                is_concurrent = True
                break

        if not is_concurrent:
            # This AV track doesn't overlap with any kept one — it's a different pass
            keep_av.add(tid_i)
        else:
            # Concurrent with a longer AV track — discard it
            del tracks[tid_i]

    return tracks


def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration_sec"] = meta["frame_count"] / meta["fps"] if meta["fps"] > 0 else 0
    cap.release()
    return meta


def detect_and_track(
    video_path: str,
    config: PipelineConfig,
    classifier=None,
    model=None,
) -> dict[int, TrackData]:
    """
    Run YOLO26 detection + ByteTrack on a video.

    Detects vehicles (car/bus/truck) and pedestrians (person).
    Labels each detection as AV, HDV, or PED.

    Args:
        model: Pre-loaded YOLO model. If None, creates a new one.
               Passing a shared model avoids GPU memory leaks across videos.

    Returns dict mapping track_id -> TrackData with all detections.
    """
    if model is None:
        model = YOLO(config.model_weights)

    # Reset tracker state from any previous video
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor.trackers = []

    # Combine vehicle + pedestrian classes for detection
    all_classes = config.vehicle_classes + config.pedestrian_classes
    ped_class_set = set(config.pedestrian_classes)

    tracker_kwargs = {
        "tracker": "bytetrack.yaml",
    }

    results_gen = model.track(
        source=video_path,
        stream=True,
        classes=all_classes,
        conf=config.detection_confidence,
        device=config.device,
        vid_stride=config.process_every_n_frames,
        verbose=False,
        **tracker_kwargs,
    )

    tracks: dict[int, TrackData] = {}
    frame_idx = 0

    for result in results_gen:
        frame_num = frame_idx * config.process_every_n_frames
        frame_idx += 1

        if config.max_frames and frame_num >= config.max_frames:
            break

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        frame_height = result.orig_img.shape[0]

        # Collect all detections for this frame
        frame_dets = []
        for box in boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            bbox = tuple(box.xyxy[0].tolist())
            class_id = int(box.cls.item())

            # Assign initial label based on COCO class
            if class_id in ped_class_set:
                label = LABEL_PED
            else:
                label = LABEL_HDV

            det = Detection(
                frame_num=frame_num,
                track_id=track_id,
                bbox=bbox,
                confidence=float(box.conf.item()),
                class_id=class_id,
                label=label,
            )

            # Two-stage Waymo classification (only for vehicles, not pedestrians)
            if classifier is not None and label != LABEL_PED:
                crop = crop_bbox(
                    result.orig_img,
                    bbox,
                    roof_extension=config.crop_roof_extension,
                )
                if crop is not None and crop.size > 0:
                    det.waymo_score = classifier.predict(crop)
                    det.is_waymo = det.waymo_score >= config.classifier_confidence
                    if det.is_waymo:
                        det.label = LABEL_AV

            frame_dets.append(det)

        # Apply spatial filter (removes streetlights/poles)
        frame_dets = _apply_spatial_filter(frame_dets, frame_height, config)

        # Apply cross-class NMS (removes double boxes)
        frame_dets = _apply_cross_class_nms(
            frame_dets, config.cross_class_iou_threshold
        )

        # Accumulate into tracks
        for det in frame_dets:
            if det.track_id not in tracks:
                tracks[det.track_id] = TrackData(track_id=det.track_id)
            tracks[det.track_id].detections.append(det)

        if frame_idx % 500 == 0:
            print(f"  Processed frame {frame_num}, active tracks: {len(tracks)}")

    # Assign final labels to tracks: if a track qualifies as AV, relabel all its detections
    for track in tracks.values():
        if track.waymo_ratio >= config.waymo_frame_ratio:
            for det in track.detections:
                if det.label != LABEL_PED:
                    det.label = LABEL_AV
                    det.is_waymo = True

    # Stitch broken tracks (same vehicle re-acquired with new ID)
    tracks = _stitch_broken_tracks(
        tracks,
        max_gap_frames=config.gap_tolerance_frames,
        max_distance=200.0,
    )

    # Remove duplicate tracks (two track IDs on the same object)
    tracks = _deduplicate_tracks(tracks, config.track_dedup_iou_threshold)

    # Merge concurrent AV tracks — there is only one Waymo, so two
    # simultaneous AV tracks means ByteTrack split the same vehicle.
    # Keep the longer AV track, discard shorter ones.
    tracks = _merge_concurrent_av_tracks(tracks)

    print(
        f"  Detection complete: {frame_idx} frames, {len(tracks)} total tracks "
        f"({sum(1 for t in tracks.values() if t.dominant_label == LABEL_AV)} AV, "
        f"{sum(1 for t in tracks.values() if t.dominant_label == LABEL_HDV)} HDV, "
        f"{sum(1 for t in tracks.values() if t.dominant_label == LABEL_PED)} PED)"
    )
    return tracks
