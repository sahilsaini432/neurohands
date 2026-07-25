import json
import queue
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from detect_hands_helper import draw_gesture_for_fixed_frame, result_to_frame_dict, save_photo_with_metadata


DATASET_DIR = Path(__file__).parent.parent / "dataset"
JSONL_PATH = DATASET_DIR / "landmarks.jsonl"


class DatasetCollector:
    def __init__(self, signer="unknown", initial_label=None):
        self.signer = signer
        self.current_label = initial_label or ""
        self.mode = "static"
        self._counts: Counter = Counter()
        self._seq_frames: list = []
        self._seq_start_result = None

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self._load_existing_counts()

    def _load_existing_counts(self):
        if not JSONL_PATH.exists():
            return
        with open(JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._counts[obj.get("label", "")] += 1
                except json.JSONDecodeError:
                    pass

    def set_label(self, word: str):
        self.current_label = word.strip()

    def toggle_mode(self):
        self.mode = "sequence" if self.mode == "static" else "static"

    def stats(self) -> dict:
        return dict(self._counts)

    def _preview_dir(self, label: str) -> Path:
        d = DATASET_DIR / "preview" / label
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_record(self, record: dict):
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._counts[record["label"]] += 1

    def _write_preview(self, result, label: str, sample_id: str):
        if result is None:
            return
        frame, _ = draw_gesture_for_fixed_frame(result)
        preview_path = self._preview_dir(label) / f"{sample_id}.png"
        save_photo_with_metadata(frame, {}, str(preview_path))

    def add_static(self, result):
        if not self.current_label:
            print("No label set — skipping static capture")
            return
        sample_id = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        frame_dict = result_to_frame_dict(result)
        handedness = list(frame_dict.keys())
        cam_w = cam_h = 0
        record = {
            "id": sample_id,
            "label": self.current_label,
            "type": "static",
            "fps": None,
            "num_frames": 1,
            "handedness": handedness,
            "frames": [frame_dict],
            "meta": {"signer": self.signer, "source": "live", "cam_w": cam_w, "cam_h": cam_h},
        }
        self._write_record(record)
        self._write_preview(result, self.current_label, sample_id)
        print(f"Static sample saved: {self.current_label} ({self._counts[self.current_label]} total)")

    def begin_sequence(self):
        self._seq_frames = []
        self._seq_start_result = None

    def add_sequence_frame(self, result):
        self._seq_frames.append(result)
        if self._seq_start_result is None and result.multi_hand_landmarks:
            self._seq_start_result = result

    def end_sequence(self):
        if not self.current_label:
            print("No label set — discarding sequence")
            self._seq_frames = []
            return
        if not self._seq_frames:
            print("Empty sequence — skipping")
            return

        frames_data = [result_to_frame_dict(r) for r in self._seq_frames]
        all_hands: set = set()
        for fd in frames_data:
            all_hands.update(fd.keys())

        mid_result = self._seq_frames[len(self._seq_frames) // 2]
        sample_id = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        record = {
            "id": sample_id,
            "label": self.current_label,
            "type": "sequence",
            "fps": 20,
            "num_frames": len(frames_data),
            "handedness": sorted(all_hands),
            "frames": frames_data,
            "meta": {"signer": self.signer, "source": "live", "cam_w": 0, "cam_h": 0},
        }
        self._write_record(record)
        self._write_preview(mid_result, self.current_label, sample_id)
        print(f"Sequence saved: {self.current_label}, {len(frames_data)} frames ({self._counts[self.current_label]} total)")
        self._seq_frames = []
        self._seq_start_result = None
