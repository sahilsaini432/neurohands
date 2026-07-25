import json
import sys
from collections import Counter
from pathlib import Path

JSONL_PATH = Path(__file__).parent.parent / "dataset" / "landmarks.jsonl"


def main():
    if not JSONL_PATH.exists():
        print(f"No dataset found at {JSONL_PATH}")
        sys.exit(0)

    total = 0
    label_counts: Counter = Counter()
    type_counts: Counter = Counter()
    seq_lengths: list = []

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            label_counts[obj.get("label", "(unknown)")] += 1
            sample_type = obj.get("type", "static")
            type_counts[sample_type] += 1
            if sample_type == "sequence":
                seq_lengths.append(obj.get("num_frames", 0))

    print(f"\nDataset: {JSONL_PATH}")
    print(f"Total samples: {total}")
    print(f"  static:   {type_counts['static']}")
    print(f"  sequence: {type_counts['sequence']}")
    if seq_lengths:
        print(f"  avg sequence length: {sum(seq_lengths) / len(seq_lengths):.1f} frames")
    print("\nPer-label counts:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
