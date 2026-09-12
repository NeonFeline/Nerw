"""Train the DAS vibration transformer with masked reconstruction + SIGReg.

Example (synthetic smoke run):

    python -m model.train --synthetic --epochs 3 --batch-size 8 \
        --window-size 256 --patch-size 16 --d-model 64 --n-layers 2 \
        --n-heads 4 --dim-feedforward 128 --proj-dim 64 --proj-hidden 64 \
        --out /tmp/run

Example (miniDAS recording, HDF5 traces [time, channels] with class labels):

    uv run python -m model.train --data data/synthetic/run01/das.h5 \
        --background-only --window-size 1024 --stride 256 \
        --epochs 30 --out runs/das_jepa
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.anomaly import AnomalyConfig, AnomalyDetector
from model.backbone import ModelConfig, VibrationTransformer
from model.data import (
    WindowDataset,
    discover_recordings,
    estimate_channel_stats,
    make_synthetic_das,
)
from model.losses import LeJEPALoss
from model.masking import MaskConfig, generate_span_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("data")
    data.add_argument("--data", type=str, default=None, help=".npy/.npz/.h5 file or directory")
    data.add_argument("--key", type=str, default=None, help="array key for .npz files")
    data.add_argument("--synthetic", action="store_true", help="use synthetic DAS data")
    data.add_argument(
        "--background-only",
        action="store_true",
        help="keep only unlabeled background windows (miniDAS labels required)",
    )
    data.add_argument(
        "--max-event-fraction",
        type=float,
        default=0.0,
        help="max labeled fraction for a window to count as background; ignore "
             "pixels (class 255) count against a window just like event pixels, "
             "so unlabeled transients stay out of the calibration set",
    )
    data.add_argument(
        "--highpass-hz", type=float, default=2.0,
        help="strip sub-Hz drift before windowing. Raw DAS puts most of its "
             "variance below 1 Hz, where no event lives: unfiltered, a window-RMS "
             "detector scores chance on the same data a 15-150 Hz one scores 0.78 "
             "on. Must match --highpass-hz at fine-tuning time. 0 disables.",
    )
    data.add_argument("--synthetic-channels", type=int, default=32)
    data.add_argument("--synthetic-samples", type=int, default=60000)
    data.add_argument("--window-size", type=int, default=1024)
    data.add_argument("--patch-size", type=int, default=16)
    data.add_argument("--stride", type=int, default=None, help="default: window_size // 2")
    data.add_argument("--val-fraction", type=float, default=0.1)
    data.add_argument("--num-workers", type=int, default=0)

    model = parser.add_argument_group("model")
    model.add_argument("--d-model", type=int, default=256)
    model.add_argument("--n-heads", type=int, default=8)
    model.add_argument("--n-layers", type=int, default=6)
    model.add_argument("--decoder-layers", type=int, default=2)
    model.add_argument("--dim-feedforward", type=int, default=1024)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--proj-dim", type=int, default=256)
    model.add_argument("--proj-hidden", type=int, default=512)

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--sigreg-weight", type=float, default=0.05)
    train.add_argument("--num-slices", type=int, default=256)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--warmup-fraction", type=float, default=0.05)
    train.add_argument(
        "--gain-jitter",
        type=float,
        default=0.0,
        help="std of log-normal per-channel gain augmentation (helps cross-recording transfer)",
    )
    train.add_argument("--amp", action="store_true")
    train.add_argument("--device", type=str, default=None)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--log-every", type=int, default=50)

    masking = parser.add_argument_group("masking")
    masking.add_argument("--num-spans", type=int, default=4)
    masking.add_argument("--min-span", type=int, default=2)
    masking.add_argument("--max-span", type=int, default=8)
    masking.add_argument("--mask-ratio", type=float, default=None)

    anomaly = parser.add_argument_group("anomaly calibration")
    anomaly.add_argument("--calib-windows", type=int, default=1024)
    anomaly.add_argument("--num-eval-masks", type=int, default=4)
    anomaly.add_argument("--residual-weight", type=float, default=0.6)
    anomaly.add_argument("--embedding-weight", type=float, default=0.4)
    anomaly.add_argument("--shrinkage", type=float, default=0.1)
    anomaly.add_argument("--threshold-quantile", type=float, default=0.995)

    parser.add_argument("--out", type=str, default="runs/das_jepa")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(args: argparse.Namespace) -> WindowDataset:
    stride = args.stride or max(1, args.window_size // 2)
    if args.synthetic:
        array = make_synthetic_das(
            num_channels=args.synthetic_channels,
            num_samples=args.synthetic_samples,
            seed=args.seed,
        )
        return WindowDataset([array], args.window_size, stride, normalize=True)

    if not args.data:
        raise SystemExit("provide --data or --synthetic")
    paths = discover_recordings(args.data)
    return WindowDataset.from_paths(
        paths, args.window_size, stride, key=args.key, normalize=True,
        highpass_hz=args.highpass_hz,
    )


def evaluate(
    model: VibrationTransformer,
    loader: DataLoader,
    loss_fn: LeJEPALoss,
    mask_config: MaskConfig,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "mse": 0.0, "sigreg": 0.0}
    batches = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            mask = generate_span_mask(
                x.shape[0], model.num_patches, mask_config, generator=generator,
                device=device,
            )
            output = model(x, mask)
            if x.shape[0] < 2:
                mse = (output.pred - output.patches).square()[mask].mean()
                totals["total"] += float(mse.detach())
                totals["mse"] += float(mse.detach())
                batches += 1
                continue
            losses = loss_fn(output, mask)
            totals["total"] += float(losses.total.detach())
            totals["mse"] += float(losses.mse.detach())
            totals["sigreg"] += float(losses.sigreg.detach())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


class JitteredWindowDataset(torch.utils.data.Dataset):
    """Wrap a window dataset with deterministic log-normal gain jitter."""

    def __init__(self, dataset: torch.utils.data.Dataset, std: float, seed: int = 0):
        self.dataset = dataset
        self.std = float(std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> torch.Tensor:
        x = self.dataset[item]
        generator = torch.Generator().manual_seed(self.seed + item)
        gains = torch.exp(self.std * torch.randn(x.shape[0], 1, generator=generator))
        return x * gains


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_datasets(args)
    if args.background_only:
        dataset = dataset.filter_background(
            max_event_fraction=args.max_event_fraction
        )
    stride = args.stride or max(1, args.window_size // 2)
    gap = math.ceil(args.window_size / stride)
    train_dataset, val_dataset = dataset.split(args.val_fraction, gap=gap)

    channel_mean, channel_std = estimate_channel_stats(
        train_dataset, max_windows=512, seed=args.seed
    )
    train_dataset.set_channel_stats(channel_mean, channel_std)
    val_dataset.set_channel_stats(channel_mean, channel_std)

    model_config = ModelConfig(
        num_channels=dataset.num_channels,
        window_size=args.window_size,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
    )
    mask_config = MaskConfig(
        num_spans=args.num_spans,
        min_span=args.min_span,
        max_span=args.max_span,
        mask_ratio=args.mask_ratio,
    )

    model = VibrationTransformer(model_config).to(device)
    loss_fn = LeJEPALoss(
        sigreg_weight=args.sigreg_weight, num_slices=args.num_slices
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_fraction * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps)
        * 0.5
        * (1.0 + math.cos(math.pi * min(1.0, step / total_steps))),
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mask_generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    metadata = dataset.metadata[0] if dataset.metadata else {}
    sampling_rate = metadata.get("sampling_rate")
    if sampling_rate:
        window_seconds = args.window_size / float(sampling_rate)
        rate_text = f"{float(sampling_rate):g} Hz"
    else:
        window_seconds = None
        rate_text = "n/a"
    print(
        f"[train] windows: {len(train_dataset)} train / {len(val_dataset)} val | "
        f"channels: {dataset.num_channels} | window: {args.window_size} samples"
        + (f" ({window_seconds:.2f}s)" if window_seconds else "")
        + f" | fs: {rate_text} | device: {device}"
    )

    best_val = float("inf")
    checkpoint_path = out_dir / "checkpoint.pt"
    for epoch in range(args.epochs):
        model.train()
        running = {"total": 0.0, "mse": 0.0, "sigreg": 0.0}
        start = time.time()
        for step, x in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            if args.gain_jitter > 0:
                gains = torch.exp(
                    args.gain_jitter
                    * torch.randn(x.shape[0], x.shape[1], 1, device=device)
                )
                x = x * gains
            mask = generate_span_mask(
                x.shape[0],
                model.num_patches,
                mask_config,
                generator=mask_generator,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(x, mask)
                losses = loss_fn(output, mask)
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["total"] += float(losses.total.detach())
            running["mse"] += float(losses.mse.detach())
            running["sigreg"] += float(losses.sigreg.detach())
            if (step + 1) % args.log_every == 0:
                divisor = step + 1
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[epoch {epoch + 1}/{args.epochs}] step {step + 1}/{steps_per_epoch} "
                    f"total {running['total'] / divisor:.4f} "
                    f"mse {running['mse'] / divisor:.4f} "
                    f"sigreg {running['sigreg'] / divisor:.4f} lr {lr:.2e}"
                )

        val_metrics = evaluate(
            model, val_loader, loss_fn, mask_config, args.seed + 999, device
        )
        elapsed = time.time() - start
        print(
            f"[epoch {epoch + 1}/{args.epochs}] val total {val_metrics['total']:.4f} "
            f"mse {val_metrics['mse']:.4f} sigreg {val_metrics['sigreg']:.4f} "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(
                {
                    "model_config": asdict(model_config),
                    "mask_config": asdict(mask_config),
                    "model_state": model.state_dict(),
                    "channel_mean": torch.as_tensor(channel_mean),
                    "channel_std": torch.as_tensor(channel_std),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "val_mse": best_val,
                },
                checkpoint_path,
            )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)

    anomaly_config = AnomalyConfig(
        residual_weight=args.residual_weight,
        embedding_weight=args.embedding_weight,
        shrinkage=args.shrinkage,
        num_eval_masks=args.num_eval_masks,
        threshold_quantile=args.threshold_quantile,
        mask_seed=args.seed + 1234,
    )
    detector = AnomalyDetector(model, anomaly_config, mask_config)
    calibration_dataset = train_dataset
    if any(label is not None for label in train_dataset.labels):
        calibration_dataset = train_dataset.filter_background(
            max_event_fraction=args.max_event_fraction
        )
        print(
            f"[calib] {len(calibration_dataset)} background windows "
            f"(from {len(train_dataset)} train windows)"
        )
    if args.gain_jitter > 0:
        calibration_dataset = JitteredWindowDataset(
            calibration_dataset, args.gain_jitter, seed=args.seed + 31
        )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    detector.fit(calibration_loader, max_windows=args.calib_windows)
    anomaly_path = out_dir / "anomaly_stats.pt"
    detector.save(anomaly_path)

    print(
        f"[done] best val mse {best_val:.4f} | checkpoint: {checkpoint_path} | "
        f"anomaly stats: {anomaly_path} | threshold: {float(detector.threshold):.4f}"
    )


if __name__ == "__main__":
    main()
