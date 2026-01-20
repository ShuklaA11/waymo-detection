"""Interpolate gaps in tracked object bounding boxes for smoother tracking."""

from __future__ import annotations

from .config import PipelineConfig
from .detect_and_track import Detection, TrackData


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at fraction t."""
    return a + (b - a) * t


def interpolate_track_gaps(
    tracks: dict[int, TrackData],
    config: PipelineConfig,
) -> dict[int, TrackData]:
    """
    Fill gaps in tracks by linearly interpolating bounding boxes.

    When a track has missing frames (detection dropped for a few frames),
    this creates synthetic detections with interpolated bbox positions
    so the bounding box follows the object smoothly.

    Only fills gaps <= max_interpolation_gap frames.
    """
    if not config.interpolate_gaps:
        return tracks

    max_gap = config.max_interpolation_gap

    for track in tracks.values():
        if len(track.detections) < 2:
            continue

        # Sort detections by frame number
        track.detections.sort(key=lambda d: d.frame_num)

        interpolated = []
        for i in range(len(track.detections) - 1):
            det_a = track.detections[i]
            det_b = track.detections[i + 1]
            interpolated.append(det_a)

            gap = det_b.frame_num - det_a.frame_num
            if gap <= 1 or gap > max_gap:
                continue

            # Interpolate bbox for each missing frame
            ax1, ay1, ax2, ay2 = det_a.bbox
            bx1, by1, bx2, by2 = det_b.bbox

            for frame_offset in range(1, gap):
                t = frame_offset / gap
                interp_bbox = (
                    _lerp(ax1, bx1, t),
                    _lerp(ay1, by1, t),
                    _lerp(ax2, bx2, t),
                    _lerp(ay2, by2, t),
                )

                interp_det = Detection(
                    frame_num=det_a.frame_num + frame_offset,
                    track_id=track.track_id,
                    bbox=interp_bbox,
                    confidence=(det_a.confidence + det_b.confidence) / 2,
                    class_id=det_a.class_id,
                    label=det_a.label,
                    is_waymo=det_a.is_waymo,
                    waymo_score=(det_a.waymo_score + det_b.waymo_score) / 2,
                )
                interpolated.append(interp_det)

        # Don't forget the last detection
        interpolated.append(track.detections[-1])
        track.detections = interpolated

    return tracks
