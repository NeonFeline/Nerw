"""Fine-tune the pretrained backbone into a per-pixel event classifier.

Stage 1 (``model.train``) is label-free: masked reconstruction plus SIGReg over
background windows.  Stage 2 is this: attach a classification head, train on
the generator's class masks, and report what the model can actually tell apart.

    uv run python -m model.finetune \
        --pretrained runs/v3_pretrain/checkpoint.pt \
        --train data/synthetic/v3/train --val data/synthetic/v3/val \
        --epochs 40 --out runs/v3_segmenter

``--freeze-epochs`` trains the head alone first, so the randomly initialised
head cannot wreck the pretrained features with its early gradients, then
unfreezes the backbone at a lower learning rate.  Pass ``--scratch`` to skip
the pretrained weights entirely, which is the comparison that says whether the
LeJEPA stage bought anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.backbone import ModelConfig, VibrationTransformer
from model.data import WindowDataset, discover_recordings, estimate_channel_stats
from model.segmenter import (
    ConfusionMatrix,
    DASSegmenter,
    SegmenterConfig,
    class_frequencies,
    inverse_frequency_weights,
    save_segmenter,
    segmentation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="stage-1 checkpoint; omit (or --scratch) to start cold")
    parser.add_argument("--scratch", action="store_true",
                        help="ignore --pretrained: the ablation for the LeJEPA stage")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--highpass-hz", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--class-weight-floor", type=float, default=0.02)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    # architecture, used only when there is no pretrained checkpoint
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    return parser.parse_args()


def build_dataset(source, args, channel_stats=None, shuffle_stats=False):
    paths = discover_recordings(source)
    dataset = WindowDataset.from_paths(
        paths,
        window_size=args.window_size,
        stride=args.stride,
        return_labels=True,
        highpass_hz=args.highpass_hz,
        normalize=True,
    )
    if channel_stats is None:
        channel_stats = estimate_channel_stats(dataset, max_windows=512, seed=args.seed)
    dataset.set_channel_stats(*channel_stats)
    return dataset, channel_stats


def class_table(paths):
    """Class ids and names, taken from the recordings themselves."""
    ids: dict[str, int] = {}
    for path in paths:
        from model.data import read_h5_metadata

        if Path(path).suffix in (".h5", ".hdf5"):
            ids.update(read_h5_metadata(path).get("class_ids", {}))
    if not ids:
        raise SystemExit("recordings carry no class_ids attribute")
    return ids, {value: key for key, value in ids.items()}


@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_weight=None):
    model.eval()
    confusion = ConfusionMatrix(num_classes)
    total, batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += float(segmentation_loss(logits, y, class_weight))
        batches += 1
        confusion.update(logits.argmax(dim=1), y)
    return total / max(batches, 1), confusion


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths = discover_recordings(args.train)
    class_ids, class_names = class_table(train_paths)
    num_classes = max(class_ids.values()) + 1

    train_set, channel_stats = build_dataset(args.train, args)
    val_set = None
    if args.val:
        val_set, _ = build_dataset(args.val, args, channel_stats=channel_stats)
    print(f"[data] train windows {len(train_set)}"
          + (f" | val windows {len(val_set)}" if val_set else "")
          + f" | classes {num_classes} | high-pass {args.highpass_hz} Hz")

    counts = class_frequencies(train_set.labels, num_classes)
    labelled = counts.sum()
    print("[data] labelled pixels per class:")
    for class_id in range(num_classes):
        if counts[class_id]:
            print(f"    {class_names.get(class_id, class_id):14s} {counts[class_id]:14,d}"
                  f"  {100 * counts[class_id] / labelled:6.3f}%")
    class_weight = None
    if not args.no_class_weights:
        class_weight = inverse_frequency_weights(counts, args.class_weight_floor).to(device)

    if args.pretrained and not args.scratch:
        checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        backbone_config = ModelConfig(**checkpoint["model_config"])
        if backbone_config.window_size != args.window_size:
            raise SystemExit(
                f"--window-size {args.window_size} does not match the pretrained "
                f"backbone's {backbone_config.window_size}"
            )
        backbone = VibrationTransformer(backbone_config)
        backbone.load_state_dict(checkpoint["model_state"])
        print(f"[init] loaded pretrained backbone from {args.pretrained}")
    else:
        backbone = VibrationTransformer(ModelConfig(
            num_channels=train_set.num_channels, window_size=args.window_size,
            patch_size=args.patch_size, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ))
        print("[init] random initialisation (no pretrained backbone)")

    model = DASSegmenter(
        backbone,
        SegmenterConfig(num_classes=num_classes, hidden=args.hidden, dropout=args.dropout,
                        class_names=tuple(class_names.get(i, str(i)) for i in range(num_classes))),
    ).to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = (DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if val_set else None)

    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.lr},
        {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_scale},
    ], weight_decay=args.weight_decay)
    steps = max(1, args.epochs * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.lr * args.backbone_lr_scale],
        total_steps=steps, pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        model.freeze_backbone(frozen)
        model.train()
        started, running, batches = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss = segmentation_loss(model(x), y, class_weight)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss)
            batches += 1

        line = (f"[epoch {epoch:3d}/{args.epochs}] "
                f"{'head only' if frozen else 'full    '} "
                f"train {running / max(batches, 1):.4f}")
        record = {"epoch": epoch, "frozen": frozen, "train_loss": running / max(batches, 1)}
        if val_loader is not None:
            val_loss, confusion = evaluate(model, val_loader, device, num_classes, class_weight)
            metrics = confusion.metrics()
            present = metrics["support"] > 0
            present[0] = False
            event_iou = float(np.nanmean(metrics["iou"][present])) if present.any() else float("nan")
            line += f" | val {val_loss:.4f} | event mIoU {event_iou:.4f}"
            record.update({"val_loss": val_loss, "event_miou": event_iou,
                           "accuracy": metrics["accuracy"]})
            if event_iou > best:
                best = event_iou
                save_segmenter(out_dir / "segmenter.pt", model, {
                    "channel_mean": channel_stats[0], "channel_std": channel_stats[1],
                    "class_ids": class_ids, "epoch": epoch, "event_miou": event_iou,
                    "highpass_hz": args.highpass_hz, "args": vars(args),
                })
                line += "  *"
        line += f" ({time.time() - started:.1f}s)"
        print(line, flush=True)
        history.append(record)

    if val_loader is None:
        save_segmenter(out_dir / "segmenter.pt", model, {
            "channel_mean": channel_stats[0], "channel_std": channel_stats[1],
            "class_ids": class_ids, "epoch": args.epochs,
            "highpass_hz": args.highpass_hz, "args": vars(args),
        })
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    if val_loader is not None:
        model_state = torch.load(out_dir / "segmenter.pt", map_location=device,
                                 weights_only=False)["model_state"]
        model.load_state_dict(model_state)
        _, confusion = evaluate(model, val_loader, device, num_classes, class_weight)
        print("\n[val] best checkpoint")
        print(confusion.report(class_names))
    print(f"\n[done] {out_dir / 'segmenter.pt'}  best event mIoU {best:.4f}")


if __name__ == "__main__":
    main()
