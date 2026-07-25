# Neurohands

Real-time ASL hand gesture recognition using MediaPipe, with tooling to collect a labeled landmark dataset for training custom models.

## Subsystems

- **`hand_detection/`** — primary app. Live webcam capture, MediaPipe landmark detection, photo/video saving with embedded landmark metadata, keyboard shortcuts, voice commands, and dataset collection mode.
- **`gesture/`** — earlier experiment using MediaPipe's prebuilt `GestureRecognizer` to classify canned gesture categories.
- **`open-ai/`** — standalone script that sends a saved hand image to OpenAI Vision to guess its ASL meaning.

## Setup

```bash
python install_libraries.py
```

## Running

```bash
cd hand_detection

# Live detection
python detect_hands.py --live
python detect_hands.py --live --voice       # + voice commands
python detect_hands.py --live --saveVideo   # + video recording
python detect_hands.py --live --time 30     # auto-stop after 30s

# Static / batch
python detect_hands.py --photo --input path/to/img.jpg
python detect_hands.py --dir --input path/to/folder/

# Dataset collection
python detect_hands.py --collect --label hello --signer yourname
python dataset_stats.py   # print dataset stats
```

## Dataset Collection

`--collect` mode captures labeled ASL samples into `dataset/landmarks.jsonl` (JSONL, one record per line).

**Keyboard controls** (video window must have focus):

| Key | Action |
|-----|--------|
| `n` | Set new label (type in terminal) |
| `m` | Toggle static / sequence mode |
| `p` or `space` | Capture static sample |
| `r` (hold) | Record sequence; release to save |
| `q` | Quit |

Each sample stores 21 normalized + world-space landmarks per hand. Preview PNGs are written to `dataset/preview/<label>/`. See [`dataset/README.md`](dataset/README.md) for the full schema.

## Architecture

Multi-threaded, coordinated via a global event bus (`event_manager.py`). The video thread runs MediaPipe every frame and pushes results onto queues; the main loop reads queues and handles persistence and UI.

See [`CLAUDE.md`](CLAUDE.md) for detailed architecture notes.
