# Plan: ASL Labeled Dataset Collection Tooling

## Context

Goal: build a **labeled ASL dataset** (sign → word) from the existing `hand_detection/`
webcam pipeline, as the foundation for two future custom models:
- ASL → speech
- speech → ASL

**This phase = dataset tooling only.** No model training yet. You cannot train models
with no data; first make it fast to *collect labeled samples*.

### What exists today (verified)
- `media_capture.py` `VideoCaptureThread` runs MediaPipe `Hands` every frame, produces a
  `result` with `multi_hand_landmarks` (21 normalized x/y/z) + `multi_handedness` (L/R).
- Static single-frame capture already works: `take_photo` event → `PhotoQueue` → main loop
  (`detect_hands.py:131`) → `save_take_photo` → `save_photo()` (`detect_hands_helper.py:263`)
  renders + writes a PNG with landmarks base64'd into PNG text metadata.
- `HandLandmark` enum + `HandLandmarks.from_hand_landmarks` (`detect_hands_helper.py:29-85`)
  already extract the 21 landmarks in canonical order — **reuse this**.
- `draw_gesture_for_fixed_frame` + `save_photo_with_metadata` (`detect_hands_helper.py:136-212`)
  already render + persist a hand image — **reuse for PNG preview**.

### The core gap
There is **no label field** anywhere, and capture is **single-frame only**. Most ASL words
are *motion*, so we also need variable-length **sequence** capture.

## Decisions (locked with user)
- **Switchable** static (fingerspelling / static signs) + sequence (motion words).
- **Labeling:** type the word once → it becomes the "current label" shown on-screen →
  every capture tags with it until changed (set-and-hammer for bulk reps).
- **Sequence trigger:** hold-to-record (press+hold → release ends the sample).
- **Hands:** store up to 2 hands per sample (L/R), whatever MediaPipe finds.
- **Format:** **JSONL = system of record.** PNG-metadata path kept as a **preview** so you
  can eyeball samples.

## Data model

`dataset/landmarks.jsonl` — one JSON object per line (append-only):

```json
{
  "id": "2026-07-14T15:32:45.123456Z",
  "label": "hello",
  "type": "sequence",            // "static" | "sequence"
  "fps": 20,
  "num_frames": 24,              // 1 for static
  "handedness": ["Right", "Left"],
  "frames": [
    { "Right": {"norm": [[x,y,z], ...21], "world": [[x,y,z], ...21]},
      "Left":  {"norm": [...], "world": [...]} }
    // ...one entry per recorded frame; static = length 1
  ],
  "meta": {"signer": "unknown", "source": "live", "cam_w": 640, "cam_h": 480}
}
```

Notes:
- Store **both** `norm` (MediaPipe normalized 0–1, already available) and `world`
  (`result.multi_hand_world_landmarks`, wrist-origin meters — scale/position invariant,
  cheap to add, valuable for training). Any per-sample normalization is a training-time
  transform later; keep raw here.
- `signer` from a `--signer` flag (default `"unknown"`) — needed for clean signer-disjoint
  train/test splits later.

`dataset/preview/<label>/<id>.png` — one representative frame per sample (static = the frame;
sequence = middle frame), rendered via existing `draw_gesture_for_fixed_frame`. For review only.

Directory `dataset/` is **tracked** (landmarks are the asset). `dataset/preview/` may be
gitignored if it grows large — decide at commit time.

## Implementation

### 1. New: `hand_detection/dataset_collector.py`
`DatasetCollector` class — owns dataset state, no cv2/threading:
- `current_label`, `mode` (`"static"`/`"sequence"`), per-label counts (`Counter`).
- `set_label(word)`, `toggle_mode()`.
- `add_static(result)` → build one-frame sample, append JSONL line, write preview PNG.
- `begin_sequence()` / `add_sequence_frame(result)` / `end_sequence()` → buffer frames while
  held, on end write one JSONL line (skip if buffer empty / no hand), write mid-frame preview.
- `stats()` → dict label→count (for the HUD + verification).
- Opens `dataset/landmarks.jsonl` in append mode; creates `dataset/` + `preview/<label>/`
  on demand (mirror the mkdir guard in `save_photo`).

### 2. New helper in `detect_hands_helper.py`
`result_to_frame_dict(result) -> dict` — returns `{ "Right": {"norm":[...], "world":[...]}, ... }`
for one frame, reusing the `HandLandmark` enum ordering (same loop as
`HandLandmarks.from_hand_landmarks:70-76`) and pulling `multi_hand_world_landmarks` in parallel
with `multi_hand_landmarks`, keyed by `multi_handedness` label. This is the single extraction
point shared by static + sequence.

### 3. `media_capture.py` — `VideoCaptureThread`
- Add `SequenceRecording` flag + `SequenceQueue` (`queue.Queue`, generous maxsize). In
  `start_loop` (after `result` is computed, ~line 122), when `SequenceRecording` push `result`
  onto `SequenceQueue` every frame.
- Keep existing `TakePhoto`/`PhotoQueue` for static — no change to photo mode.
- Add `start_sequence()` / `stop_sequence()` methods toggling the flag.

### 4. `detect_hands.py` — new `--collect` mode
New CLI flags: `--collect`, `--label <word>` (optional initial), `--signer <id>`.
Reuse the existing `--live` capture setup (`VideoCaptureThread`), then a collect-specific loop:
- Instantiate `DatasetCollector(signer=args.signer, initial_label=args.label)`.
- Each iteration: draw display frame (existing `FrameQueue`), then overlay a HUD via
  `cv2.putText` — current label, mode, and `label: N samples` from `collector.stats()`.
- **Keyboard (cv2 `waitKey` for discrete keys):**
  - `n` → `input()` in terminal to set a new label (brief blocking prompt; acceptable).
  - `m` → `collector.toggle_mode()` (static ↔ sequence).
  - `p` / space → static capture: trigger existing `take_photo`; drain `PhotoQueue` →
    `collector.add_static(result)`.
  - `q` → quit.
- **Hold-to-record (sequence mode):** cv2 gives no key-up event, so poll the `keyboard` lib
  (already imported, `media_capture.py:11`) in the loop: on `keyboard.is_pressed('r')` rising
  edge → `video_thread.start_sequence()` + `collector.begin_sequence()`; while held drain
  `SequenceQueue` → `collector.add_sequence_frame`; on release → `stop_sequence()` +
  `collector.end_sequence()`. Guard the label is set before recording.
- Voice (`--voice`) is **not** wired to labeling (unreliable STT); leave voice for the future
  speech→ASL model.

### 5. New: `hand_detection/dataset_stats.py`
Small standalone reader: loads `dataset/landmarks.jsonl`, prints total samples,
per-label counts, static-vs-sequence split, avg sequence length. Doubles as the verification
tool and as the seed of the future training dataloader.

### 6. Docs
- Add a short `dataset/README.md` documenting the JSONL schema (the block above).
- Update root `CLAUDE.md` "Architecture (hand_detection)" + run-commands with the
  `--collect` workflow and keyboard map.

## Files
- New: `hand_detection/dataset_collector.py`, `hand_detection/dataset_stats.py`,
  `dataset/README.md`
- Edit: `hand_detection/detect_hands.py` (flags + collect loop + HUD),
  `hand_detection/media_capture.py` (sequence flag/queue),
  `hand_detection/detect_hands_helper.py` (`result_to_frame_dict`), `CLAUDE.md`
- Reuse (no change): `HandLandmark`, `draw_gesture_for_fixed_frame`,
  `save_photo_with_metadata`, `event_manager`.

## Verification (end-to-end)
1. `cd hand_detection && python detect_hands.py --collect --label hello --signer sahil`
2. Static: press `m` to ensure static mode; press `p` a few times → HUD count rises.
3. Sequence: press `m` to switch; hold `r` while signing "hello", release → one sample added.
4. Set new label: press `n`, type `thanks`, collect a few.
5. `python dataset_stats.py` → confirm totals + per-label counts match what you captured;
   confirm static vs sequence split and avg sequence length look right.
6. Open `dataset/landmarks.jsonl`: confirm each line has `label`, correct `type`,
   `frames` length == `num_frames`, and each hand has 21 `norm` + 21 `world` points.
7. Open a `dataset/preview/<label>/*.png` to eyeball the rendered hand.
