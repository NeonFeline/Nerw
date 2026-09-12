"""Evaluate anomaly scores against miniDAS class labels.

Scores every window of a labeled recording, then reports ROC-AUC of the
residual / embedding / combined scores, the detection rate per event class at
the calibrated threshold, and the background false-positive rate.

Example:

    uv run python -m model.evaluate \
        --checkpoint runs/das_jepa/checkpoint.pt \
        --anomaly-stats runs/das_jepa/anomaly_stats.pt \
        --input data/synthetic/run01/das.h5 --stride 256 \
        --out runs/das_jepa/run01_scores.npz
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

from model.anomaly import AnomalyDetector
from model.backbone import ModelConfig, VibrationTransformer
from model.data import IGNORE_CLASS, WindowDataset, discover_recordings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--anomaly-stats", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--max-ignore-fraction", type=float, default=0.0,
        help="drop event-free windows with more than this fraction of ignore pixels; "
             "their true class is unknown, so scoring them as negatives is a guess",
    )
    return parser.parse_args()


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC with tie handling."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    num_pos = int(labels.sum())
    num_neg = int(len(labels) - num_pos)
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    for index in range(1, len(scores) + 1):
        if index == len(scores) or sorted_scores[index] != sorted_scores[start]:
            average_rank = 0.5 * (start + index - 1) + 1.0
            ranks[order[start:index]] = average_rank
            start = index

    return (ranks[labels].sum() - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)


def window_rms_scores(
    dataset: WindowDataset,
    sampling_rate: float,
    fmin: float = 15.0,
    fmax: float = 150.0,
) -> np.ndarray:
    """Trivial energy-detector baseline: band-passed window RMS."""
    from scipy.signal import butter, filtfilt

    b, a = butter(4, (fmin / (sampling_rate / 2), fmax / (sampling_rate / 2)), btype="band")
    filtered = [
        filtfilt(b, a, np.asarray(array, dtype=np.float32), axis=1)
        for array in dataset.arrays
    ]
    scores = np.empty(len(dataset), dtype=np.float64)
    for item, (file_index, start) in enumerate(dataset.index):
        window = filtered[file_index][:, start : start + dataset.window_size]
        scores[item] = np.sqrt(np.square(window, dtype=np.float64).mean())
    return scores


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = VibrationTransformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = WindowDataset.from_paths(
        discover_recordings(args.input),
        window_size=model.config.window_size,
        stride=args.stride,
        key=args.key,
        channel_mean=checkpoint.get("channel_mean"),
        channel_std=checkpoint.get("channel_std"),
        normalize=True,
    )
    if all(label is None for label in dataset.labels):
        raise SystemExit(f"{args.input} has no class labels to evaluate against")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    detector = AnomalyDetector.load(args.anomaly_stats, model)
    scores = detector.score_loader(loader)

    class_ids: dict[str, int] = {}
    for meta in dataset.metadata:
        class_ids.update(meta.get("class_ids", {}))
    names = {value: key for key, value in class_ids.items()}

    fractions = np.array(
        [dataset.window_label_fractions(i) for i in range(len(dataset))]
    )
    event_fraction, ignore_fraction = fractions[:, 0], fractions[:, 1]
    has_event = event_fraction > 0.0
    # An event-free window carrying ignore pixels may hold an unlabelled
    # transient, so it is neither a positive nor a trustworthy negative.
    scored = has_event | (ignore_fraction <= args.max_ignore_fraction)

    class_presence = {class_id: np.zeros(len(dataset), dtype=bool) for class_id in names}
    for item, (file_index, start) in enumerate(dataset.index):
        label = dataset.labels[file_index]
        window = label[:, start : start + dataset.window_size]
        for class_id in np.unique(window):
            if int(class_id) not in (0, IGNORE_CLASS) and int(class_id) in class_presence:
                class_presence[int(class_id)][item] = True

    score = scores["score"].numpy()
    is_anomaly = scores["is_anomaly"].numpy()
    threshold = float(scores["threshold"][0])
    sampling_rate = next(
        (meta.get("sampling_rate") for meta in dataset.metadata if meta.get("sampling_rate")),
        None,
    )

    print(f"[evaluate] windows: {len(dataset)} | scored: {int(scored.sum())} "
          f"| background: {int((scored & ~has_event).sum())} "
          f"| event: {int(has_event.sum())} "
          f"| dropped (ignore): {int((~scored).sum())} | threshold: {threshold:.3f}")
    if sampling_rate:
        baseline = window_rms_scores(dataset, float(sampling_rate))
        print(f"[evaluate] AUC baseline_rms: {roc_auc(baseline[scored], has_event[scored]):.4f}  "
              f"(trivial band-pass energy detector)")
    for key in ("score", "residual_z", "embedding_z"):
        auc = roc_auc(scores[key].numpy()[scored], has_event[scored])
        print(f"[evaluate] AUC {key:12s}: {auc:.4f}")

    true_positive = int((is_anomaly & has_event & scored).sum())
    false_positive = int((is_anomaly & ~has_event & scored).sum())
    false_negative = int((~is_anomaly & has_event & scored).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    print(
        f"[evaluate] at threshold: precision {precision:.3f} recall {recall:.3f} "
        f"f1 {f1:.3f} | false positives {false_positive}/{int((scored & ~has_event).sum())}"
    )

    if names:
        print("[evaluate] per-class detection rate:")
        for class_id in sorted(names):
            present = class_presence[class_id]
            if not present.any():
                continue
            detected = int((is_anomaly & present).sum())
            mean_score = float(score[present].mean())
            print(
                f"  {names[class_id]:12s} windows {int(present.sum()):4d} "
                f"detected {detected:4d} ({detected / present.sum():5.1%}) "
                f"mean score {mean_score:7.2f}"
            )
        background_score = score[scored & ~has_event]
        if background_score.size:
            print(f"  {'background':12s} windows {background_score.size:4d} "
                  f"mean score {background_score.mean():7.2f}")

    if args.out:
        payload = {
            "file_index": np.array([index[0] for index in dataset.index]),
            "start": np.array([index[1] for index in dataset.index]),
            "event_fraction": event_fraction,
            "ignore_fraction": ignore_fraction,
            "has_event": has_event,
            "scored": scored,
        }
        payload.update({key: value.numpy() for key, value in scores.items()})
        np.savez(args.out, **payload)
        print(f"[evaluate] wrote {args.out}")


if __name__ == "__main__":
    main()
