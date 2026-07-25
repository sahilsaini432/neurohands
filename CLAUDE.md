# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Neurohands recognizes ASL hand gestures from a webcam using MediaPipe. There are two mostly independent subsystems that were built at different times and are run separately:

- `hand_detection/` — the primary, current app. Live capture, landmark drawing, photo/video saving with embedded landmark metadata, keyboard shortcuts, and optional voice commands.
- `gesture/` — an earlier experiment using MediaPipe's prebuilt `GestureRecognizer` (`.task` model) to detect canned gesture categories and log them to `gestures.csv`.
- `open-ai/` — a standalone script that sends a saved hand image to the OpenAI vision API to guess its ASL meaning.

There is no build system, test suite, or package manifest. It's a set of directly-run Python scripts.

## Setup & Running

Install dependencies (uses pip, with a Homebrew fallback path for M1/M2 Macs):

```bash
python install_libraries.py
```

Run the main app. **Scripts use sibling imports with no package structure**, so they must be run from inside their own directory:

```bash
cd hand_detection
python detect_hands.py --live              # live webcam detection
python detect_hands.py --live --voice      # + voice commands
python detect_hands.py --live --saveVideo  # + video recording controls
python detect_hands.py --live --time 30    # auto-stop after 30s
python detect_hands.py --photo --input path/to/img.jpg   # single image
python detect_hands.py --dir --input path/to/folder/     # batch a directory
```

Live keyboard controls (video window must have focus): `q` quit, `p` photo, `r`/`s` start/stop recording (only with `--saveVideo`), `h` help.

Voice commands (with `--voice`): "start/stop processing", "start/stop recording", "take/save photo", "exit/quit program".

Dataset collection:

```bash
cd hand_detection
python detect_hands.py --collect --label hello --signer sahil   # collect ASL samples
python dataset_stats.py                                          # print dataset stats
```

Collect mode keyboard controls (video window must have focus):
- `n` — set new label (prompts in terminal)
- `m` — toggle static / sequence mode
- `p` or `space` — capture static sample
- `r` (hold) — record sequence; release to save
- `q` — quit

Samples written to `dataset/landmarks.jsonl` (JSONL, one record per line). Preview PNGs in `dataset/preview/<label>/`. See `dataset/README.md` for schema.

Gesture experiment:

```bash
cd gesture
python gesture_recognition.py --cgd 1   # collect gesture data into gestures.csv
# flags: --si save image, --shpi show processed image (matplotlib),
#        --chd handedness, --chld landmarks, --chwld world landmarks
```

OpenAI script: needs `OPENAI_API_KEY` in a `.env` at repo root; the target `image_path` is hardcoded in `open-ai/analyze_image.py` and must be edited before running.

## Architecture (hand_detection)

The live pipeline is multi-threaded and coordinated by a global event bus, not direct calls:

- **`event_manager.py`** — a tiny global `event_manager` singleton (publish/subscribe). Threads communicate by triggering named events (`take_photo`, `start_recording`, `stop`, `speak`, etc.) rather than holding references to each other. This is the seam that decouples the voice thread, video thread, and main loop.
- **`detect_hands.py`** — entry point and main loop. Owns the cv2 window and keyboard input. Pulls frames off `VideoCaptureThread.FrameQueue` to display, and pulls results off `PhotoQueue` to save. Registers `on_stop_event` and `save_photo` as event listeners. Also hosts `--collect` mode via `_run_collect_mode`.
- **`dataset_collector.py`** — `DatasetCollector` class. Manages current label, static/sequence mode, per-label counts. Writes JSONL records to `dataset/landmarks.jsonl` and preview PNGs to `dataset/preview/<label>/`. No cv2/threading dependencies.
- **`dataset_stats.py`** — standalone reader: prints total samples, per-label counts, static/sequence split, avg sequence length.
- **`media_capture.py`** — two threads:
  - `VideoCaptureThread`: reads the camera, runs MediaPipe `Hands`, draws landmarks, and pushes frames into bounded queues. Frames are flipped horizontally (mirror). A separate `original_frame` copy is kept for the fixed-size saved output.
  - `VoiceCommandThread`: `speech_recognition` (Google) → normalizes text to `snake_case` → matches substrings → triggers events. Speaks back via gTTS.
- **`detect_hands_helper.py`** — all drawing/encoding/persistence. Key idea: saved images encode the 21 hand landmarks into PNG text metadata (base64 JSON via `HandLandmarks`), so a saved photo carries its own gesture data and can be decoded later (`get_exif_data` / `decode_hand_landmarks`). Left and right hands are drawn on separate centered sub-frames and `hstack`ed side by side.

Data flow for a photo: keypress/voice → `take_photo` event → `VideoCaptureThread` puts result on `PhotoQueue` → main loop → `save_take_photo` event → `save_photo()` renders landmarks on the current frame and writes PNG + metadata to `output_data/`.

## Gotchas

- **Cross-platform:** parts are Mac-specific and will break on Windows/Linux. `gesture_helper.py` forces `matplotlib.use("MacOSX")`. `media_capture.py` uses `sr.Microphone(2)` (hardcoded device index). `install_libraries.py` includes `portaudio` (a Homebrew package, not pip). The OpenAI script has a hardcoded `/Users/sahilsaini/...` path.
- **Output dirs:** `save_photo` creates `output_data/` on demand, but recording (`start_recording`) and `process_frame_from_filepath` write to `./output_data/` relative to CWD without creating it first — the directory must exist.
- **Two `HandLandmarks` classes** exist (`detect_hands_helper.py` and `gesture_recognition_result.py`) with different fields; they are not interchangeable.
- The `gesture/` code expects a `gesture_recognizer.task` model file in `gesture/` that is not committed.
- `gitignore` excludes `output_data/`, `.env`, and model/media artifacts.
