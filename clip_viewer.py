"""Clip Viewer & Extender UI.

Browse pipeline output clips, preview them, and re-extract with adjusted
padding (add/remove seconds from beginning or end).

Usage:
    python clip_viewer.py --output-dir output/sjb --video-dir "D:/Video_recording/..."
    python clip_viewer.py --output-dir output/speedway --video-dir "D:/path1" "D:/path2"
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_source_video(video_stem: str, video_dirs: list[str]) -> str | None:
    """Find the source video file matching a stem across search directories."""
    for video_dir in video_dirs:
        for ext in ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mkv", "*.mov"):
            for match in glob.glob(str(Path(video_dir) / "**" / ext), recursive=True):
                if Path(match).stem == video_stem:
                    return match
    return None


def get_video_fps_ffprobe(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        num, den = result.stdout.strip().split("/")
        return float(num) / float(den)
    except Exception:
        return 25.0


def get_video_duration_ffprobe(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_video_stem_from_clip(clip_name: str) -> str:
    """Extract source video stem from clip filename.

    Format: waymo_{video_stem}_clip{NNN}_{day|night}.mp4
    """
    base = clip_name.removeprefix("waymo_").removesuffix(".mp4")
    for suffix in ("_day", "_night"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    idx = base.rfind("_clip")
    if idx != -1:
        return base[:idx]
    return base


# ---------------------------------------------------------------------------
# Clip data
# ---------------------------------------------------------------------------

class ClipInfo:
    def __init__(self, row: dict, output_dir: str):
        self.clip_name: str = row["clip_file"]
        self.time_of_day: str = row["time_of_day"]
        self.track_id: int = int(row["track_id"])
        self.start_frame: int = int(row["start_frame"])
        self.end_frame: int = int(row["end_frame"])
        self.duration_frames: int = int(row["duration_frames"])
        self.avg_confidence: float = float(row["avg_confidence"])
        self.avg_waymo_score: float = float(row["avg_waymo_score"])
        self.waymo_ratio: float = float(row["waymo_ratio"])
        self.video_stem: str = extract_video_stem_from_clip(self.clip_name)

        # Resolve clip path on disk
        self.clip_path: str = str(
            Path(output_dir) / self.time_of_day / self.clip_name
        )
        self.exists: bool = Path(self.clip_path).is_file()


# ---------------------------------------------------------------------------
# Video playback widget (plays clip in a Label using cv2 + PIL)
# ---------------------------------------------------------------------------

class VideoPlayer:
    """Plays a video file frame-by-frame in a tkinter Label."""

    def __init__(self, label: tk.Label):
        self.label = label
        self.cap: cv2.VideoCapture | None = None
        self.playing = False
        self.fps = 30.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_photo = None  # prevent GC

    def load(self, path: str):
        self.stop()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Show first frame
        self._show_next_frame()

    def play(self):
        if self.cap is None or not self.cap.isOpened():
            return
        if self.playing:
            return
        self.playing = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def pause(self):
        self.playing = False
        self._stop_event.set()

    def stop(self):
        self.pause()
        if self.cap:
            self.cap.release()
            self.cap = None

    def restart(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if not self.playing:
                self._show_next_frame()

    def _play_loop(self):
        delay = 1.0 / self.fps
        while not self._stop_event.is_set():
            if not self._show_next_frame():
                # Reached end — loop back
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(delay)
        self.playing = False

    def _show_next_frame(self) -> bool:
        if self.cap is None:
            return False
        ret, frame = self.cap.read()
        if not ret:
            return False
        # Resize to fit display
        h, w = frame.shape[:2]
        max_w, max_h = 800, 450
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(img)
        self._current_photo = photo  # prevent GC
        try:
            self.label.configure(image=photo)
        except tk.TclError:
            pass  # widget destroyed
        return True


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class ClipViewerApp:
    # Config: existing padding applied by the pipeline
    EXISTING_PAD_BEFORE = 3.0
    EXISTING_PAD_AFTER = 3.0

    def __init__(self, root: tk.Tk, clips: list[ClipInfo], video_dirs: list[str], output_dir: str):
        self.root = root
        self.clips = clips
        self.video_dirs = video_dirs
        self.output_dir = output_dir
        self.current_idx = 0

        # Cache: video_stem -> (path, fps, duration)
        self.video_cache: dict[str, tuple[str, float, float]] = {}

        self.root.title("Waymo Clip Viewer & Extender")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1e1e1e")

        self._build_ui()
        self._load_clip(0)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#e0e0e0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TScale", background="#1e1e1e")

        # --- Top: Navigation ---
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.btn_prev = ttk.Button(nav_frame, text="< Prev", command=self._prev_clip)
        self.btn_prev.pack(side="left")

        self.lbl_counter = ttk.Label(nav_frame, text="", style="Header.TLabel")
        self.lbl_counter.pack(side="left", padx=20)

        self.btn_next = ttk.Button(nav_frame, text="Next >", command=self._next_clip)
        self.btn_next.pack(side="left")

        # Jump to clip
        ttk.Label(nav_frame, text="  Go to #:").pack(side="left", padx=(30, 5))
        self.entry_goto = ttk.Entry(nav_frame, width=5)
        self.entry_goto.pack(side="left")
        self.entry_goto.bind("<Return>", self._goto_clip)
        ttk.Button(nav_frame, text="Go", command=self._goto_clip).pack(side="left", padx=5)

        # --- Middle: Video + Info ---
        mid_frame = ttk.Frame(self.root)
        mid_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Video display
        video_frame = ttk.Frame(mid_frame)
        video_frame.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(video_frame, bg="#000000", width=800, height=450)
        self.video_label.pack(fill="both", expand=True)

        # Playback controls
        ctrl_frame = ttk.Frame(video_frame)
        ctrl_frame.pack(fill="x", pady=5)

        self.btn_play = ttk.Button(ctrl_frame, text="Play", command=self._toggle_play)
        self.btn_play.pack(side="left", padx=5)
        ttk.Button(ctrl_frame, text="Restart", command=self._restart).pack(side="left", padx=5)

        self.lbl_clip_file = ttk.Label(ctrl_frame, text="", style="TLabel")
        self.lbl_clip_file.pack(side="left", padx=20)

        # Info panel (right side)
        info_frame = ttk.Frame(mid_frame, width=280)
        info_frame.pack(side="right", fill="y", padx=(10, 0))
        info_frame.pack_propagate(False)

        ttk.Label(info_frame, text="Clip Details", style="Header.TLabel").pack(anchor="w", pady=(0, 10))

        self.info_text = tk.Text(
            info_frame, width=32, height=14, bg="#2d2d2d", fg="#e0e0e0",
            font=("Consolas", 10), relief="flat", state="disabled", wrap="word"
        )
        self.info_text.pack(fill="x")

        # --- Bottom: Adjustment controls ---
        adj_frame = ttk.LabelFrame(self.root, text="Extend Clip", padding=10)
        adj_frame.pack(fill="x", padx=10, pady=(5, 10))

        # Add before
        row1 = ttk.Frame(adj_frame)
        row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="Add BEFORE (sec):").pack(side="left")
        self.var_before = tk.DoubleVar(value=0.0)
        self.scale_before = ttk.Scale(
            row1, from_=0, to=15, variable=self.var_before,
            orient="horizontal", length=300, command=self._update_before_label
        )
        self.scale_before.pack(side="left", padx=10)
        self.lbl_before_val = ttk.Label(row1, text="0.0s", width=6)
        self.lbl_before_val.pack(side="left")

        # Add after
        row2 = ttk.Frame(adj_frame)
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="Add AFTER  (sec):").pack(side="left")
        self.var_after = tk.DoubleVar(value=0.0)
        self.scale_after = ttk.Scale(
            row2, from_=0, to=15, variable=self.var_after,
            orient="horizontal", length=300, command=self._update_after_label
        )
        self.scale_after.pack(side="left", padx=10)
        self.lbl_after_val = ttk.Label(row2, text="0.0s", width=6)
        self.lbl_after_val.pack(side="left")

        # Buttons
        btn_row = ttk.Frame(adj_frame)
        btn_row.pack(fill="x", pady=(10, 0))

        self.btn_extract = ttk.Button(
            btn_row, text="Re-extract This Clip", command=self._re_extract_current
        )
        self.btn_extract.pack(side="left", padx=5)

        self.btn_extract_all = ttk.Button(
            btn_row, text="Re-extract ALL Clips", command=self._re_extract_all
        )
        self.btn_extract_all.pack(side="left", padx=5)

        self.lbl_status = ttk.Label(btn_row, text="", foreground="#4ec9b0")
        self.lbl_status.pack(side="left", padx=20)

        # Video player
        self.player = VideoPlayer(self.video_label)

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_clip())
        self.root.bind("<Right>", lambda e: self._next_clip())
        self.root.bind("<space>", lambda e: self._toggle_play())

    def _update_before_label(self, _=None):
        self.lbl_before_val.config(text=f"{self.var_before.get():.1f}s")

    def _update_after_label(self, _=None):
        self.lbl_after_val.config(text=f"{self.var_after.get():.1f}s")

    def _load_clip(self, idx: int):
        if idx < 0 or idx >= len(self.clips):
            return
        self.current_idx = idx
        clip = self.clips[idx]

        self.lbl_counter.config(text=f"Clip {idx + 1} / {len(self.clips)}")
        self.lbl_clip_file.config(text=clip.clip_name)

        # Update info
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")

        fps = self._get_source_fps(clip)
        event_start = clip.start_frame / fps if fps else 0
        event_end = clip.end_frame / fps if fps else 0

        info = (
            f"File: {clip.clip_name}\n"
            f"Time: {clip.time_of_day}\n"
            f"Track ID: {clip.track_id}\n\n"
            f"Event frames: {clip.start_frame}-{clip.end_frame}\n"
            f"Event time: {event_start:.1f}s - {event_end:.1f}s\n"
            f"Duration: {clip.duration_frames}f ({clip.duration_frames / fps:.1f}s)\n\n"
            f"Waymo score: {clip.avg_waymo_score:.3f}\n"
            f"Waymo ratio: {clip.waymo_ratio:.3f}\n"
            f"Confidence: {clip.avg_confidence:.3f}\n\n"
            f"Source: {clip.video_stem}\n"
            f"Exists: {'Yes' if clip.exists else 'NO'}"
        )
        self.info_text.insert("1.0", info)
        self.info_text.config(state="disabled")

        # Load video
        if clip.exists:
            self.player.load(clip.clip_path)
        else:
            self.player.stop()

    def _get_source_fps(self, clip: ClipInfo) -> float:
        """Get FPS for the source video (cached)."""
        if clip.video_stem in self.video_cache:
            return self.video_cache[clip.video_stem][1]
        # Try to find and cache
        source = find_source_video(clip.video_stem, self.video_dirs)
        if source:
            fps = get_video_fps_ffprobe(source)
            dur = get_video_duration_ffprobe(source)
            self.video_cache[clip.video_stem] = (source, fps, dur)
            return fps
        return 25.0  # fallback

    def _get_source_info(self, clip: ClipInfo) -> tuple[str, float, float] | None:
        """Get (path, fps, duration) for source video."""
        if clip.video_stem in self.video_cache:
            return self.video_cache[clip.video_stem]
        source = find_source_video(clip.video_stem, self.video_dirs)
        if source:
            fps = get_video_fps_ffprobe(source)
            dur = get_video_duration_ffprobe(source)
            self.video_cache[clip.video_stem] = (source, fps, dur)
            return (source, fps, dur)
        return None

    def _prev_clip(self):
        if self.current_idx > 0:
            self._load_clip(self.current_idx - 1)

    def _next_clip(self):
        if self.current_idx < len(self.clips) - 1:
            self._load_clip(self.current_idx + 1)

    def _goto_clip(self, event=None):
        try:
            n = int(self.entry_goto.get()) - 1
            if 0 <= n < len(self.clips):
                self._load_clip(n)
            else:
                messagebox.showwarning("Invalid", f"Enter 1-{len(self.clips)}")
        except ValueError:
            messagebox.showwarning("Invalid", "Enter a number")

    def _toggle_play(self):
        if self.player.playing:
            self.player.pause()
            self.btn_play.config(text="Play")
        else:
            self.player.play()
            self.btn_play.config(text="Pause")

    def _restart(self):
        self.player.restart()

    def _set_status(self, msg: str):
        self.lbl_status.config(text=msg)
        self.root.update_idletasks()

    def _re_extract_clip(self, clip: ClipInfo, add_before: float, add_after: float) -> bool:
        """Re-extract a single clip with extended padding. Returns True on success."""
        info = self._get_source_info(clip)
        if info is None:
            self._set_status(f"Source video not found: {clip.video_stem}")
            return False

        source_path, fps, video_duration = info

        # Calculate new timestamps from event frames
        event_start_sec = clip.start_frame / fps
        event_end_sec = clip.end_frame / fps

        new_start = max(0, event_start_sec - self.EXISTING_PAD_BEFORE - add_before)
        new_end = event_end_sec + self.EXISTING_PAD_AFTER + add_after
        if video_duration > 0:
            new_end = min(new_end, video_duration)
        duration = new_end - new_start

        out_path = clip.clip_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{new_start:.3f}",
            "-i", source_path,
            "-t", f"{duration:.3f}",
            "-vf", "scale=1920:1080",
            "-r", "30",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-an",
            "-movflags", "+faststart",
            out_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            return True
        except subprocess.CalledProcessError as e:
            self._set_status(f"ffmpeg error: {e.stderr[:100]}")
            return False
        except subprocess.TimeoutExpired:
            self._set_status("ffmpeg timed out")
            return False

    def _re_extract_current(self):
        clip = self.clips[self.current_idx]
        add_before = self.var_before.get()
        add_after = self.var_after.get()

        if add_before == 0 and add_after == 0:
            messagebox.showinfo("Nothing to do", "Set add-before or add-after to a value > 0")
            return

        self.player.stop()
        self._set_status("Re-extracting...")

        def work():
            ok = self._re_extract_clip(clip, add_before, add_after)
            self.root.after(0, lambda: self._on_extract_done(ok, clip))

        threading.Thread(target=work, daemon=True).start()

    def _on_extract_done(self, ok: bool, clip: ClipInfo):
        if ok:
            self._set_status(f"Done! +{self.var_before.get():.1f}s before, +{self.var_after.get():.1f}s after")
            clip.exists = True
            # Reload the clip to show the new version
            self.player.load(clip.clip_path)
        else:
            self._set_status("Failed — check source video path")

    def _re_extract_all(self):
        add_before = self.var_before.get()
        add_after = self.var_after.get()

        if add_before == 0 and add_after == 0:
            messagebox.showinfo("Nothing to do", "Set add-before or add-after to a value > 0")
            return

        count = len(self.clips)
        if not messagebox.askyesno(
            "Confirm",
            f"Re-extract ALL {count} clips with "
            f"+{add_before:.1f}s before, +{add_after:.1f}s after?\n\n"
            f"This will overwrite existing clips."
        ):
            return

        self.player.stop()
        self.btn_extract_all.config(state="disabled")
        self.btn_extract.config(state="disabled")

        def work():
            success = 0
            fail = 0
            for i, clip in enumerate(self.clips):
                self.root.after(0, lambda i=i: self._set_status(
                    f"Processing {i + 1}/{count}..."
                ))
                if self._re_extract_clip(clip, add_before, add_after):
                    success += 1
                    clip.exists = True
                else:
                    fail += 1
            self.root.after(0, lambda: self._on_extract_all_done(success, fail))

        threading.Thread(target=work, daemon=True).start()

    def _on_extract_all_done(self, success: int, fail: int):
        self.btn_extract_all.config(state="normal")
        self.btn_extract.config(state="normal")
        self._set_status(f"All done: {success} OK, {fail} failed")
        # Reload current clip
        self._load_clip(self.current_idx)

    def on_close(self):
        self.player.stop()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clip Viewer & Extender UI")
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory with detection_log.csv (e.g., output/sjb)"
    )
    parser.add_argument(
        "--video-dir", required=True, nargs="+",
        help="Directory(ies) to search for source videos (recursive)"
    )
    args = parser.parse_args()

    # Check ffmpeg
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ERROR: ffmpeg/ffprobe not found on PATH")
        sys.exit(1)

    # Load detection log
    output_dir = args.output_dir
    log_path = Path(output_dir) / "detection_log.csv"
    if not log_path.exists():
        print(f"ERROR: detection_log.csv not found in {output_dir}")
        sys.exit(1)

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No clips found in detection_log.csv")
        sys.exit(1)

    clips = [ClipInfo(row, output_dir) for row in rows]
    existing = sum(1 for c in clips if c.exists)
    print(f"Loaded {len(clips)} clips ({existing} exist on disk)")

    # Launch UI
    root = tk.Tk()
    app = ClipViewerApp(root, clips, args.video_dir, output_dir)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
