"""Convert tracked vehicle data into discrete Waymo enter/exit events."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig
from .detect_and_track import TrackData


@dataclass
class WaymoEvent:
    """A single Waymo vehicle entering and exiting the frame."""

    track_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    avg_confidence: float
    avg_waymo_score: float
    waymo_ratio: float

    def duration_sec(self, fps: float) -> float:
        return self.duration_frames / fps if fps > 0 else 0.0


def _find_contiguous_segments(
    frame_nums: list[int],
    gap_tolerance: int,
) -> list[tuple[int, int]]:
    """
    Find contiguous segments in a sorted list of frame numbers.
    Merges segments separated by gaps <= gap_tolerance.

    Returns list of (start_frame, end_frame) tuples.
    """
    if not frame_nums:
        return []

    segments = []
    seg_start = frame_nums[0]
    seg_end = frame_nums[0]

    for fn in frame_nums[1:]:
        if fn - seg_end <= gap_tolerance:
            seg_end = fn
        else:
            segments.append((seg_start, seg_end))
            seg_start = fn
            seg_end = fn

    segments.append((seg_start, seg_end))
    return segments


def _split_long_segment(
    track: TrackData,
    start_frame: int,
    end_frame: int,
    max_frames: int,
) -> list[tuple[int, int]]:
    """
    Split a segment that exceeds max duration.
    Splits at the frame where the bbox is smallest (vehicle farthest from camera),
    creating natural break points.
    """
    # Get detections within this segment
    segment_dets = [
        d for d in track.detections if start_frame <= d.frame_num <= end_frame
    ]

    if not segment_dets or (end_frame - start_frame) <= max_frames:
        return [(start_frame, end_frame)]

    # Calculate bbox areas for each detection
    def bbox_area(d):
        x1, y1, x2, y2 = d.bbox
        return (x2 - x1) * (y2 - y1)

    results = []
    current_start = start_frame

    while (end_frame - current_start) > max_frames:
        # Find the detection with smallest bbox in the window around the split point
        split_target = current_start + max_frames
        # Search window: last 30% of the max_frames window
        search_start = current_start + int(max_frames * 0.7)
        candidates = [
            d for d in segment_dets if search_start <= d.frame_num <= split_target
        ]

        if candidates:
            # Split at the frame where vehicle is smallest (farthest away)
            split_det = min(candidates, key=bbox_area)
            split_frame = split_det.frame_num
        else:
            split_frame = split_target

        results.append((current_start, split_frame))
        current_start = split_frame + 1

    if current_start <= end_frame:
        results.append((current_start, end_frame))

    return results


def extract_events(
    tracks: dict[int, TrackData],
    source_fps: float,
    config: PipelineConfig,
) -> list[WaymoEvent]:
    """
    Convert tracks into discrete Waymo events.

    Filtering pipeline:
    1. Discard tracks with < min_track_frames detections
    2. Discard tracks where waymo_ratio < waymo_frame_ratio threshold
    3. Find contiguous segments of waymo detection within each track
    4. Merge segments within gap_tolerance
    5. Split segments exceeding max_clip_duration
    """
    max_frames = int(config.max_clip_duration_sec * source_fps)
    events = []

    for track_id, track in tracks.items():
        # Filter 1: minimum track length
        if track.frame_count < config.min_track_frames:
            continue

        # Filter 2: waymo frame ratio
        if track.waymo_ratio < config.waymo_frame_ratio:
            continue

        # Track passed the waymo ratio filter — use full track extent for clip
        # boundaries. The ratio filter is the gatekeeper; once confirmed as
        # Waymo, the clip should cover the entire vehicle pass (entry to exit).
        segments = [(track.start_frame, track.end_frame)]

        # Process each segment
        for seg_start, seg_end in segments:
            seg_duration = seg_end - seg_start

            # Skip tiny segments
            if seg_duration < config.min_track_frames:
                continue

            # Split if too long
            sub_segments = _split_long_segment(
                track, seg_start, seg_end, max_frames
            )

            for sub_start, sub_end in sub_segments:
                # Calculate stats for this sub-segment
                sub_dets = [
                    d
                    for d in track.detections
                    if sub_start <= d.frame_num <= sub_end
                ]
                if not sub_dets:
                    continue

                event = WaymoEvent(
                    track_id=track_id,
                    start_frame=sub_start,
                    end_frame=sub_end,
                    duration_frames=sub_end - sub_start,
                    avg_confidence=sum(d.confidence for d in sub_dets)
                    / len(sub_dets),
                    avg_waymo_score=sum(d.waymo_score for d in sub_dets)
                    / len(sub_dets),
                    waymo_ratio=sum(1 for d in sub_dets if d.is_waymo)
                    / len(sub_dets),
                )
                events.append(event)

    # Sort events by start frame
    events.sort(key=lambda e: e.start_frame)

    # Deduplicate overlapping events (different track IDs for same vehicle)
    events = _deduplicate_events(events, source_fps, config.dedup_overlap_threshold_sec)

    return events


def _deduplicate_events(
    events: list[WaymoEvent], fps: float, overlap_threshold_sec: float,
) -> list[WaymoEvent]:
    """Remove overlapping events that likely correspond to the same vehicle pass."""
    if len(events) <= 1:
        return events

    overlap_frames = int(overlap_threshold_sec * fps)
    deduplicated = [events[0]]

    for event in events[1:]:
        prev = deduplicated[-1]
        # Check if this event overlaps significantly with the previous one
        if event.start_frame <= prev.end_frame + overlap_frames:
            # Keep the one with higher waymo score
            if event.avg_waymo_score > prev.avg_waymo_score:
                deduplicated[-1] = event
        else:
            deduplicated.append(event)

    return deduplicated
