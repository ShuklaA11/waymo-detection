#!/bin/bash
# Finish remaining 02082026 videos, then continue with all other dates
PYTHON="/c/Users/as244492/AppData/Local/Programs/Python/Python312/python.exe"
WORKDIR="/d/waymo-detection-master"
PROGRESS="$WORKDIR/progress_sjb.log"
export PATH="/c/Users/as244492/Desktop/ffmpeg:$PATH"

cd "$WORKDIR"
touch "$PROGRESS"

# --- Step 1: Finish the 9 missing videos from 02082026 ---
echo "=========================================="
echo "Finishing remaining 02082026 videos at $(date)"
echo "=========================================="

MISSING_VIDEOS=(
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-160331-165605.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-165621-174854.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-174914-184148.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-184207-193441.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-193500-202734.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-202753-212027.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-212046-221321.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-221336-230610.mp4"
  "D:/Video_recording/Connected_intersection/EDK_st and SJB/02082026/EDK st and SJB-00-230625-235859.mp4"
)

for VID in "${MISSING_VIDEOS[@]}"; do
  STEM=$(basename "$VID" .mp4)
  echo "Processing: $STEM at $(date)"
  "$PYTHON" -m waymo_pipeline.run --config config_sjb_02082026.yaml --video "$VID" 2>&1
  echo "Done: $STEM at $(date)"
done

echo "DONE 02082026" >> "$PROGRESS"
echo "02082026 COMPLETE at $(date)"

# --- Step 2: Continue with remaining dates ---
DATES=(
  02092026
  02102026
  02112026
  02122026
  02132026
  02142026
  02152026
  02162026
  02172026
  02182026
  02192026
  02202026
  02212026
  02222026
  02232026
  02242026
  02252026
  02262026
  02272026
)

for DATE in "${DATES[@]}"; do
  if grep -q "^DONE $DATE$" "$PROGRESS" 2>/dev/null; then
    echo "Skipping $DATE (already completed)"
    continue
  fi

  echo "=========================================="
  echo "Starting SJB $DATE at $(date)"
  echo "=========================================="
  echo "STARTED $DATE $(date)" >> "$PROGRESS"

  cat > "config_sjb_${DATE}.yaml" <<EOF
input_videos:
  - "D:/Video_recording/Connected_intersection/EDK_st and SJB/${DATE}/*.mp4"
output_dir: "output/sjb"
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
device: "cuda:0"
process_every_n_frames: 1
EOF

  "$PYTHON" -m waymo_pipeline.run --config "config_sjb_${DATE}.yaml" 2>&1
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ]; then
    echo "DONE $DATE" >> "$PROGRESS"
    echo "Finished SJB $DATE at $(date) (success)"
  else
    echo "FAILED $DATE $(date) exit=$EXIT_CODE" >> "$PROGRESS"
    echo "FAILED SJB $DATE at $(date) (exit code $EXIT_CODE)"
  fi
done

echo "ALL SJB DATES COMPLETE at $(date)"
