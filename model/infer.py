"""Score optical-cable recordings for anomalies with a trained checkpoint.

Example:

    python -m model.infer --checkpoint runs/das_jepa/checkpoint.pt \
        --anomaly-stats runs/das_jepa/anomaly_stats.pt \
        --input /path/to/das_recording.npy --out scores.npz
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
from model.data import WindowDataset, discover_recordings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--anomaly-stats", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help=".npy/.npz file or directory")
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="scores.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = VibrationTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    paths = discover_recordings(args.input)
    dataset = WindowDataset.from_paths(
        paths,
        window_size=model_config.window_size,
        stride=args.stride,
        key=args.key,
        channel_mean=checkpoint.get("channel_mean"),
        channel_std=checkpoint.get("channel_std"),
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    detector = AnomalyDetector.load(args.anomaly_stats, model)
    scores = detector.score_loader(loader)

    file_indices = np.asarray([index[0] for index in dataset.index], dtype=np.int64)
    starts = np.asarray([index[1] for index in dataset.index], dtype=np.int64)
    arrays = {
        "file_index": file_indices,
        "start": starts,
        **{key: value.numpy() for key, value in scores.items()},
    }
    np.savez(args.out, **arrays)

    is_anomaly = scores["is_anomaly"].numpy()
    print(
        f"[infer] scored {len(starts)} windows from {len(paths)} file(s) | "
        f"anomalies: {int(is_anomaly.sum())} | "
        f"score mean {float(scores['score'].mean()):.3f} "
        f"max {float(scores['score'].max()):.3f} | "
        f"threshold {float(scores['threshold'][0]):.3f}"
    )
    detected = np.flatnonzero(is_anomaly)[:10]
    for index in detected:
        print(
            f"  window {int(index)}: file {int(file_indices[index])} "
            f"start {int(starts[index])} score {float(scores['score'][index]):.3f} "
            f"residual_z {float(scores['residual_z'][index]):.3f} "
            f"embedding_z {float(scores['embedding_z'][index]):.3f}"
        )
    print(f"[infer] wrote {args.out}")


if __name__ == "__main__":
    main()
