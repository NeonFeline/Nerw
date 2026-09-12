"""Per-pixel event classification on top of the pretrained backbone.

The LeJEPA stage learns what a quiet stretch of fibre looks like without any
labels; this stage asks a much narrower question of the same representation --
for every channel and every instant, which activity is happening there.

The backbone tokenizes time only: one token spans all channels over one time
patch.  The head mirrors the reconstruction head, mapping each token to a
per-channel class distribution for its patch, which is then expanded back to
full time resolution.  Loss is cross-entropy against the generator's mask with
``ignore_index``, so the boundary band between event and background, ambiguous
overlaps and unlabelled background transients contribute nothing -- the three
places where the label is genuinely unknown.

What this can and cannot do: DAS resolves *activity* and location.  It does not
resolve intent or identity, and no amount of training makes it.  A head that
separates digging from walking from a grinder on the cable is doing real work;
one advertised as separating a threat from a passer-by is not.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

from model.backbone import ModelConfig, VibrationTransformer

IGNORE_INDEX = 255


@dataclass
class SegmenterConfig:
    num_classes: int
    hidden: int = 256
    dropout: float = 0.1
    class_names: tuple[str, ...] = ()


class DASSegmenter(nn.Module):
    """Pretrained backbone plus a per-channel, per-patch classification head."""

    def __init__(self, backbone: VibrationTransformer, config: SegmenterConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        d_model = backbone.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, backbone.config.num_channels * config.num_classes),
        )

    @property
    def patch_size(self) -> int:
        return self.backbone.config.patch_size

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` window -> ``(B, K, C, T)`` class logits."""
        tokens = self.backbone(x).tokens                      # (B, L, d_model)
        batch, length, _ = tokens.shape
        channels = self.backbone.config.num_channels
        logits = self.head(tokens).reshape(
            batch, length, channels, self.config.num_classes
        )
        logits = logits.permute(0, 3, 2, 1)                   # (B, K, C, L)
        return logits.repeat_interleave(self.patch_size, dim=3)


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over the (channel, time) grid, ignoring unknown cells."""
    return nn.functional.cross_entropy(
        logits, target.long(), weight=class_weight, ignore_index=IGNORE_INDEX
    )


def class_frequencies(
    labels, num_classes: int, max_arrays: int | None = None
) -> np.ndarray:
    """Pixel counts per class over label masks, ignoring unknown cells."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for mask in list(labels)[:max_arrays]:
        if mask is None:
            continue
        values = np.asarray(mask).ravel()
        values = values[values != IGNORE_INDEX]
        counts += np.bincount(values, minlength=num_classes)[:num_classes]
    return counts


def inverse_frequency_weights(counts: np.ndarray, floor: float = 0.02) -> torch.Tensor:
    """Balanced class weights, normalised to mean 1.

    Background outnumbers every event class by orders of magnitude and the rare
    security classes by four or five, so unweighted cross-entropy converges to
    predicting background everywhere.
    """
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    weights = 1.0 / counts
    weights = np.maximum(weights / weights.mean(), floor)
    return torch.as_tensor(weights / weights.mean(), dtype=torch.float32)


class ConfusionMatrix:
    """Streaming confusion matrix over labelled pixels."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        predicted = predicted.detach().cpu().numpy().ravel()
        target = target.detach().cpu().numpy().ravel()
        keep = target != IGNORE_INDEX
        predicted, target = predicted[keep], target[keep]
        if predicted.size:
            flat = target.astype(np.int64) * self.num_classes + predicted.astype(np.int64)
            self.matrix += np.bincount(
                flat, minlength=self.num_classes**2
            ).reshape(self.num_classes, self.num_classes)

    def metrics(self) -> dict[str, np.ndarray]:
        true_positive = np.diag(self.matrix).astype(np.float64)
        predicted = self.matrix.sum(axis=0).astype(np.float64)
        actual = self.matrix.sum(axis=1).astype(np.float64)
        union = predicted + actual - true_positive
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(predicted > 0, true_positive / predicted, np.nan)
            recall = np.where(actual > 0, true_positive / actual, np.nan)
            iou = np.where(union > 0, true_positive / union, np.nan)
            f1 = np.where(
                (precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan
            )
        total = self.matrix.sum()
        return {
            "precision": precision, "recall": recall, "iou": iou, "f1": f1,
            "support": actual,
            "accuracy": float(true_positive.sum() / total) if total else float("nan"),
        }

    def report(self, names: dict[int, str]) -> str:
        m = self.metrics()
        lines = [f"{'class':14s} {'support':>12s} {'prec':>7s} {'recall':>7s} "
                 f"{'f1':>7s} {'IoU':>7s}"]
        for class_id in range(self.num_classes):
            if m["support"][class_id] == 0:
                continue
            lines.append(
                f"{names.get(class_id, str(class_id)):14s} "
                f"{int(m['support'][class_id]):12,d} "
                f"{m['precision'][class_id]:7.3f} {m['recall'][class_id]:7.3f} "
                f"{m['f1'][class_id]:7.3f} {m['iou'][class_id]:7.3f}"
            )
        present = m["support"] > 0
        events = present.copy()
        events[0] = False
        lines.append(f"{'-' * 58}")
        lines.append(f"{'pixel accuracy':14s} {m['accuracy']:>12.4f}")
        lines.append(f"{'mean IoU':14s} {np.nanmean(m['iou'][present]):>12.4f}  (all classes)")
        if events.any():
            lines.append(
                f"{'mean IoU':14s} {np.nanmean(m['iou'][events]):>12.4f}  (events only)"
            )
            lines.append(
                f"{'macro F1':14s} {np.nanmean(m['f1'][events]):>12.4f}  (events only)"
            )
        return "\n".join(lines)


def save_segmenter(path, model: DASSegmenter, extra: dict | None = None) -> None:
    torch.save({
        "model_config": asdict(model.backbone.config),
        "segmenter_config": asdict(model.config),
        "model_state": model.state_dict(),
        **(extra or {}),
    }, path)


def load_segmenter(path, device=None) -> tuple[DASSegmenter, dict]:
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    backbone = VibrationTransformer(ModelConfig(**checkpoint["model_config"]))
    segmenter_config = dict(checkpoint["segmenter_config"])
    segmenter_config["class_names"] = tuple(segmenter_config.get("class_names", ()))
    model = DASSegmenter(backbone, SegmenterConfig(**segmenter_config))
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
