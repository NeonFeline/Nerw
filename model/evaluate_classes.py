"""Score a fine-tuned segmenter on a held-out split.

Reports per-class precision, recall, F1 and IoU over labelled pixels, the
confusion matrix, and -- because the point of the exercise is triage -- how
often an event of one class is confused for another.  Ignore pixels never
enter any count: the boundary band, ambiguous overlaps and unlabelled
transients are exactly the cells whose true class nobody knows.

    uv run python -m model.evaluate_classes \
        --segmenter runs/v3_segmenter/segmenter.pt \
        --input data/synthetic/v3/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.data import WindowDataset, discover_recordings
from model.segmenter import IGNORE_INDEX, ConfusionMatrix, load_segmenter, segmentation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segmenter", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_segmenter(args.segmenter, device=device)
    model = model.to(device).eval()

    class_ids = checkpoint.get("class_ids", {})
    names = {value: key for key, value in class_ids.items()}
    num_classes = model.config.num_classes

    dataset = WindowDataset.from_paths(
        discover_recordings(args.input),
        window_size=model.backbone.config.window_size,
        stride=args.stride,
        return_labels=True,
        highpass_hz=checkpoint.get("highpass_hz"),
        channel_mean=checkpoint["channel_mean"],
        channel_std=checkpoint["channel_std"],
        normalize=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    confusion = ConfusionMatrix(num_classes)
    loss, batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += float(segmentation_loss(logits, y))
            batches += 1
            confusion.update(logits.argmax(dim=1), y)

    ignored = sum(int((np.asarray(m) == IGNORE_INDEX).sum()) for m in dataset.labels if m is not None)
    total = sum(int(np.asarray(m).size) for m in dataset.labels if m is not None)
    print(f"[evaluate] {args.input}")
    print(f"[evaluate] windows {len(dataset)} | cross-entropy {loss / max(batches, 1):.4f} "
          f"| ignored pixels {100 * ignored / max(total, 1):.1f}%")
    print()
    print(confusion.report(names))

    matrix = confusion.matrix
    present = [c for c in range(num_classes) if matrix[c].sum() > 0]
    print("\nconfusion (rows = truth, % of that class's pixels):")
    header = "".join(f"{names.get(c, c)[:9]:>10s}" for c in present)
    print(f"{'':14s}{header}")
    for row in present:
        share = 100.0 * matrix[row, present] / max(matrix[row].sum(), 1)
        print(f"{names.get(row, row):14s}" + "".join(f"{v:10.1f}" for v in share))

    events = [c for c in present if c != 0]
    if events:
        event_pixels = matrix[events][:, :].sum()
        as_background = matrix[events][:, 0].sum()
        wrong_event = sum(
            matrix[r, c] for r in events for c in events if r != c
        )
        print(f"\nof all labelled event pixels: {100 * as_background / event_pixels:.1f}% missed "
              f"as background, {100 * wrong_event / event_pixels:.1f}% given the wrong event class")

    if args.out:
        import json

        np.savez(
            args.out,
            confusion=matrix,
            class_ids=np.array(json.dumps(class_ids)),
            **{k: v for k, v in confusion.metrics().items() if isinstance(v, np.ndarray)},
        )
        print(f"[evaluate] wrote {args.out}")


if __name__ == "__main__":
    main()
