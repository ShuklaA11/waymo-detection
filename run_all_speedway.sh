#!/bin/bash
# Run pipeline on all Speedway dates sequentially on GPU 1
# Tracks progress per-video in progress_speedway.log — safe to restart
PYTHON="/c/Users/as244492/AppData/Local/Programs/Python/Python312/python.exe"
WORKDIR="/d/waymo-detection-master"
PROGRESS="$WORKDIR/progress_speedway.log"
CONFIG="$WORKDIR/config_speedway.yaml"
export PATH="/c/Users/as244492/Desktop/ffmpeg:$PATH"

cd "$WORKDIR"

# Create progress file if it doesn't exist
touch "$PROGRESS"

# Write a single reusable config (--video overrides input_videos)
cat > "$CONFIG" <<EOF
input_videos: []
output_dir: "output/speedway"
model_weights: "yolo26s.pt"
classifier_weights: "models/waymo_classifier.pth"
detection_mode: "two_stage"
vehicle_classes: [2, 5, 7]
pedestrian_classes: [0]
detection_confidence: 0.35
min_y_fraction: 0.15
min_bbox_area: 500
max_aspect_ratio: 4.0
max_bbox_area_fraction: 0.08
cross_class_iou_threshold: 0.5
track_dedup_iou_threshold: 0.3
classifier_confidence: 0.75
crop_roof_extension: 0.2
track_buffer: 60
min_track_frames: 8
gap_tolerance_frames: 50
max_clip_duration_sec: 60
padding_before_sec: 3.0
padding_after_sec: 3.0
waymo_frame_ratio: 0.4
dedup_overlap_threshold_sec: 5.0
min_expected_clip_sec: 12.0
max_expected_clip_sec: 45.0
output_fps: 30
output_width: 1920
output_height: 1080
output_codec: "libx264"
output_crf: 23
day_night_luminance_threshold: 80.0
interpolate_gaps: true
max_interpolation_gap: 30
device: "cuda:1"
process_every_n_frames: 1
EOF

DATES=(
  02082026
  02092026
  02102026
  02112026
  02192026
  02202026
  02212026
  02222026
  02232026
  02242026
  02252026
  02262026
  02272026
  02282026
)

TOTAL_SKIPPED=0
TOTAL_PROCESSED=0
TOTAL_FAILED=0

for DATE in "${DATES[@]}"; do
  VIDEO_DIR="D:/Video_recording/Connected_intersection/EDK_st and Speedway/${DATE}"

  # Find all mp4 files in this date folder
  VIDEOS=("$VIDEO_DIR"/*.mp4)

  # Check if glob matched anything
  if [ ! -f "${VIDEOS[0]}" ]; then
    echo "WARNING: No videos found in $VIDEO_DIR, skipping"
    continue
  fi

  echo "=========================================="
  echo "Date: $DATE — ${#VIDEOS[@]} video(s)"
  echo "=========================================="

  for VID in "${VIDEOS[@]}"; do
    STEM=$(basename "$VID" .mp4)

    # Skip if already completed
    if grep -qF "DONE_VIDEO $STEM" "$PROGRESS" 2>/dev/null; then
      echo "  Skipping $STEM (already completed)"
      TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
      continue
    fi

    echo "  Processing: $STEM at $(date)"

    "$PYTHON" -m waymo_pipeline.run --config "$CONFIG" --video "$VID" 2>&1
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
      echo "DONE_VIDEO $STEM" >> "$PROGRESS"
      echo "  Finished $STEM (success)"
      TOTAL_PROCESSED=$((TOTAL_PROCESSED + 1))
    else
      echo "FAILED_VIDEO $STEM $(date) exit=$EXIT_CODE" >> "$PROGRESS"
      echo "  FAILED $STEM (exit code $EXIT_CODE)"
      TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
  done
done

echo ""
echo "=========================================="
echo "ALL SPEEDWAY DATES COMPLETE at $(date)"
echo "  Processed: $TOTAL_PROCESSED | Skipped: $TOTAL_SKIPPED | Failed: $TOTAL_FAILED"
echo "=========================================="
