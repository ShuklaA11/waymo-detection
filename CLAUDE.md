# CLAUDE.md

## Project Summary

Two-stage computer vision pipeline that detects Waymo self-driving cars (white Jaguar I-PACE with rooftop LIDAR dome) in elevated intersection surveillance video and exports short clips of each pass. Videos are 5-20 total, day and night.

**Core principle: Precision over recall.** A missed Waymo is acceptable; a clip of a random white car is not.

## Architecture

```
YOLO26s + ByteTrack → ResNet18 classifier → Event detection → ffmpeg clip extraction
```

1. **YOLO26s** detects all vehicles; **ByteTrack** assigns persistent track IDs
2. **ResNet18** binary classifier distinguishes Waymo (LIDAR dome) from regular cars — crops extended 20% upward to capture roof
3. Tracks converted to enter/exit events with strict filtering (min frames, waymo ratio, dedup)
4. **ffmpeg** extracts clips at 1080p/30fps, sorted into `output/day/` and `output/night/`

Additional processing: track interpolation (gap fill), cross-class NMS, track deduplication, spatial filtering (streetlights/poles/glare).

## Directory Layout

```
├── config.yaml              ← ALL tunable parameters (thresholds, paths, encoding)
├── requirements.txt         ← torch, torchvision, ultralytics, opencv, pydantic, pyyaml, numpy
├── waymo_pipeline/          ← main pipeline code
│   ├── run.py               ← CLI entry point (argparse)
│   ├── pipeline.py          ← orchestrator
│   ├── config.py            ← Pydantic config model + YAML loading + device auto-detect
│   ├── detect_and_track.py  ← YOLO26 + ByteTrack + spatial filtering + cross-class NMS + track dedup/stitching
│   ├── waymo_classifier.py  ← ResNet18 binary classifier (WaymoClassifier + DummyClassifier)
│   ├── track_interpolator.py← linear bbox interpolation for gap filling
│   ├── event_detector.py    ← track→event conversion + filtering + dedup
│   ├── clip_extractor.py    ← ffmpeg clip cutting with padding
│   └── day_night.py         ← luminance-based day/night classification
├── training/                ← classifier training tools
│   ├── train_classifier.py  ← ResNet18 training (ImageFolder, heavy augmentation, weighted sampling)
│   ├── extract_crops.py     ← extract all vehicle crops from video
│   ├── extract_white_crops.py ← extract only white/light vehicle crops (HSV filter)
│   ├── filter_white_crops.py  ← move non-white crops out of not_waymo/
│   └── label_batches.py     ← interactive batch labeling (grid display, select by number)
├── models/                  ← trained weights (git-ignored)
│   └── waymo_classifier.pth ← ResNet18 weights (~43MB)
├── data/                    ← raw input videos (git-ignored)
├── output/                  ← pipeline output (git-ignored)
│   ├── day/                 ← daytime clips
│   ├── night/               ← nighttime clips
│   ├── detection_log.csv    ← per-clip metadata
│   └── faulty_clips.csv     ← clips outside expected duration range
└── training/dataset/        ← labeled images (git-ignored)
    ├── waymo/
    ├── not_waymo/
    └── unsorted/
```

## Key Commands

```bash
# Run full pipeline
python -m waymo_pipeline.run

# Single video
python -m waymo_pipeline.run --video data/my_video.mp4

# Custom config / test mode
python -m waymo_pipeline.run --config custom_config.yaml
python -m waymo_pipeline.run --max-frames 2000

# Override device
python -m waymo_pipeline.run --device cpu

# Extract crops for labeling
python training/extract_crops.py --video data/my_video.mp4 --output training/dataset/unsorted/
python training/extract_white_crops.py --video data/my_video.mp4

# Interactive labeling
python training/label_batches.py --batch-size 20

# Train classifier
python training/train_classifier.py --data training/dataset/ --output models/waymo_classifier.pth
```

## Dependencies

- Python 3.10+
- `torch>=2.0`, `torchvision>=0.15`, `ultralytics>=8.0`, `opencv-python>=4.8`, `pydantic>=2.0`, `pyyaml>=6.0`, `numpy>=1.24`
- **ffmpeg** (external system binary, not a pip package)
- GPU: auto-detects CUDA > MPS > CPU

## Key Thresholds (config.yaml)

| Parameter | Default | Purpose |
|---|---|---|
| `classifier_confidence` | `0.75` | High bar for Waymo classification |
| `waymo_frame_ratio` | `0.4` | 40% of track frames must be Waymo-positive |
| `min_track_frames` | `8` | Ignore tracks < 8 frames |
| `max_clip_duration_sec` | `60` | Hard cap on clip length |
| `gap_tolerance_frames` | `50` | Bridge ~2 sec gaps within a track |
| `detection_confidence` | `0.35` | YOLO threshold (moderate; classifier does real filtering) |
| `day_night_luminance_threshold` | `80.0` | Below this = night |
| `track_buffer` | `60` | ByteTrack lost-track memory (~2.4 sec) |
| `max_interpolation_gap` | `30` | Interpolate gaps ≤ 30 frames (~1.2 sec) |

## Pipeline Execution Flow

```
run.py → load_config (YAML + CLI overrides)
  → pipeline.run_pipeline()
    → For each video:
      1. detect_and_track() — YOLO26s + ByteTrack + spatial filter + classifier + track stitch/dedup
      2. interpolate_track_gaps() — linear bbox interpolation
      3. extract_events() — filter by min_frames/waymo_ratio, split long segments, dedup overlaps
      4. classify_video_day_night() — sample 10 frames, compute mean luminance
      5. extract_clip() per event — ffmpeg with padding, 1080p/30fps H.264
      6. Write detection_log.csv + faulty_clips.csv
```

## Key Data Structures

- **Detection** (`detect_and_track.py`): frame_num, track_id, bbox, confidence, class_id, label (AV/HDV/PED), is_waymo, waymo_score
- **TrackData** (`detect_and_track.py`): track_id + list[Detection], properties for waymo_ratio, avg_waymo_score, dominant_label
- **WaymoEvent** (`event_detector.py`): track_id, start/end frame, confidence scores, waymo_ratio
- **PipelineConfig** (`config.py`): Pydantic BaseModel with 50+ validated fields

## Development Rules

1. **Plan before coding** — describe approach and wait for approval before making changes
2. **Small changesets** — keep changes to ≤3 files at a time
3. **Precision > recall** — when tuning thresholds, always favor fewer false positives
4. **Config-driven** — all thresholds and parameters go in config.yaml, never hardcode magic numbers
5. **Test incrementally** — validate each stage independently before running full pipeline
6. **Detection modes** — `two_stage` (YOLO + ResNet18) is the primary mode; `finetuned` exists for single-model approach

## Known Constraints

- Designed for macOS Apple Silicon (MPS), also supports CUDA and CPU
- All videos from the same elevated intersection camera
- Night footage is noisy/blurry — classifier must handle this
- Vehicles appear small in frame — crops are low-resolution
- The LIDAR dome on the Waymo roof is the primary distinguishing feature
- YOLO model (`yolo26s.pt`) auto-downloads on first run via Ultralytics
- Model weights and data directories are git-ignored

## Output Format

- **Clips**: `output/{day,night}/waymo_<videostem>_clip<NNN>_{day,night}.mp4` — 1920x1080, 30fps, H.264 (CRF 23)
- **detection_log.csv**: clip_file, time_of_day, track_id, start_frame, end_frame, duration_frames, avg_confidence, avg_waymo_score, waymo_ratio
- **faulty_clips.csv**: clips outside expected duration range (< 12s or > 45s)
