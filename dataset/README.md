# ASL Dataset

## Format

`landmarks.jsonl` — one JSON object per line (append-only). Each line is one sample.

```json
{
  "id": "2026-07-14T15:32:45.123456+00:00Z",
  "label": "hello",
  "type": "static",
  "fps": null,
  "num_frames": 1,
  "handedness": ["Right"],
  "frames": [
    {
      "Right": {
        "norm": [[x, y, z], ...21 landmarks],
        "world": [[x, y, z], ...21 landmarks]
      }
    }
  ],
  "meta": {"signer": "unknown", "source": "live", "cam_w": 0, "cam_h": 0}
}
```

### Fields

- `id` — ISO 8601 UTC timestamp (used as unique ID and filename for preview PNG)
- `label` — the ASL word or phrase this sample represents
- `type` — `"static"` (single frame, fingerspelling / static signs) or `"sequence"` (motion word)
- `fps` — frame rate for sequences (20); `null` for static
- `num_frames` — 1 for static; N for sequence
- `handedness` — which hands appear across all frames
- `frames` — list of per-frame dicts keyed by `"Right"` / `"Left"`
  - `norm` — MediaPipe normalized coords (0–1), 21 × [x, y, z]
  - `world` — MediaPipe world coords (wrist-origin, meters), 21 × [x, y, z]
- `meta.signer` — signer ID (from `--signer` flag; default `"unknown"`)

Landmark ordering follows `HandLandmark` enum (wrist=0, thumb_cmc=1, … pinky_tip=20).

## Preview images

`preview/<label>/<id>.png` — rendered hand image for eyeballing samples.  
Static: the captured frame. Sequence: middle frame of the recording.

## Collection

```bash
cd hand_detection
python detect_hands.py --collect --label hello --signer sahil
```

Keyboard controls while the video window is open:
- `n` — set new label (type in terminal)
- `m` — toggle static / sequence mode
- `p` or `space` — capture static sample
- `r` (hold) — record sequence; release to save
- `q` — quit

## Stats

```bash
cd hand_detection
python dataset_stats.py
```
