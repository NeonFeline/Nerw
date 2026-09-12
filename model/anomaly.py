"""Anomaly detection for DAS vibration windows.

Two complementary scores are calibrated on normal data:

1. **Reconstruction residual** -- per-pixel squared error on masked spans,
   standardized by the per-pixel error distribution of normal windows. This
   catches localized transients that the backbone cannot reconstruct.
2. **Embedding distance** -- Mahalanobis distance of the SIGReg projection in
   the Gaussian embedding space fitted on normal windows. This catches
   distribution-level shifts even when the reconstruction stays plausible.

The two z-scored signals are combined into a single score with configurable
weights; the decision threshold is a quantile of the combined score on the
calibration set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from model.backbone import VibrationTransformer
from model.masking import MaskConfig, generate_span_mask


@dataclass
class AnomalyConfig:
    """Scoring configuration."""

    residual_weight: float = 0.6
    embedding_weight: float = 0.4
    shrinkage: float = 0.1
    num_eval_masks: int = 4
    threshold_quantile: float = 0.995
    mask_seed: int = 1234
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.residual_weight < 0.0 or self.embedding_weight < 0.0:
            raise ValueError("score weights must be non-negative")
        if self.residual_weight + self.embedding_weight <= 0.0:
            raise ValueError("at least one score weight must be positive")
        if not 0.0 <= self.shrinkage < 1.0:
            raise ValueError("shrinkage must be in [0, 1)")
        if self.num_eval_masks < 1:
            raise ValueError("num_eval_masks must be >= 1")
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")


class AnomalyDetector:
    """Residual + embedding anomaly detector built on a trained backbone."""

    def __init__(
        self,
        model: VibrationTransformer,
        config: AnomalyConfig | None = None,
        mask_config: MaskConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or AnomalyConfig()
        self.mask_config = mask_config or MaskConfig()
        self.num_patches = model.num_patches
        self.num_channels = model.config.num_channels
        self.window_size = model.config.window_size
        self.device = next(model.parameters()).device
        self.fitted = False

        self.residual_mean: torch.Tensor | None = None
        self.residual_std: torch.Tensor | None = None
        self.residual_count: torch.Tensor | None = None
        self.embed_mean: torch.Tensor | None = None
        self.embed_precision: torch.Tensor | None = None
        self.res_calib_mean = torch.tensor(0.0)
        self.res_calib_std = torch.tensor(1.0)
        self.embed_calib_mean = torch.tensor(0.0)
        self.embed_calib_std = torch.tensor(1.0)
        self.threshold = torch.tensor(float("inf"))

    def _eval_masks(self, batch_size: int, batch_index: int) -> list[torch.Tensor]:
        generator = torch.Generator(device=self.device).manual_seed(
            self.config.mask_seed + batch_index
        )
        return [
            generate_span_mask(
                batch_size,
                self.num_patches,
                self.mask_config,
                generator=generator,
                device=self.device,
            )
            for _ in range(self.config.num_eval_masks)
        ]

    @staticmethod
    def _unpack(batch) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch

    @torch.no_grad()
    def fit(self, loader, max_windows: int | None = None) -> "AnomalyDetector":
        """Calibrate residual statistics, embedding Gaussian, and threshold."""
        was_training = self.model.training
        self.model.eval()

        residual_sum = torch.zeros(
            self.num_channels, self.window_size, dtype=torch.float64
        )
        residual_sumsq = torch.zeros_like(residual_sum)
        residual_count = torch.zeros_like(residual_sum)

        processed = 0
        for batch_index, batch in enumerate(loader):
            x = self._unpack(batch).to(self.device, dtype=torch.float32)
            for mask in self._eval_masks(x.shape[0], batch_index):
                error, pixel_mask, _ = self.model.per_pixel_error(x, mask)
                error = error.double().cpu()
                pixel_mask = pixel_mask.cpu()
                residual_sum += (error * pixel_mask).sum(dim=0)
                residual_sumsq += (error.square() * pixel_mask).sum(dim=0)
                residual_count += pixel_mask.sum(dim=0)
            processed += x.shape[0]
            if max_windows is not None and processed >= max_windows:
                break

        if processed < 2:
            raise ValueError("need at least 2 windows to calibrate the detector")

        self.residual_mean = (residual_sum / residual_count.clamp_min(1)).float()
        variance = (
            residual_sumsq / residual_count.clamp_min(1) - self.residual_mean.double() ** 2
        ).clamp_min(0.0)
        self.residual_std = variance.sqrt().float()
        self.residual_count = residual_count

        residual_scores: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []
        processed = 0
        for batch_index, batch in enumerate(loader):
            x = self._unpack(batch).to(self.device, dtype=torch.float32)
            masks = self._eval_masks(x.shape[0], batch_index)
            window_score = torch.zeros(x.shape[0], dtype=torch.float64)
            for mask_index, mask in enumerate(masks):
                error, pixel_mask, output = self.model.per_pixel_error(x, mask)
                window_score += (
                    self._window_residual(error, pixel_mask, divide_by_masks=len(masks))
                    .double()
                    .cpu()
                )
                if mask_index == 0:
                    embeddings.append(output.proj.float().cpu())
            residual_scores.append(window_score.float())
            processed += x.shape[0]
            if max_windows is not None and processed >= max_windows:
                break

        residual_scores_t = torch.cat(residual_scores)
        embeddings_t = torch.cat(embeddings)
        self.embed_mean, self.embed_precision = self._fit_gaussian(embeddings_t)
        embedding_scores = self._mahalanobis(
            embeddings_t, self.embed_mean, self.embed_precision
        )

        eps = self.config.eps
        self.res_calib_mean = residual_scores_t.mean()
        self.res_calib_std = residual_scores_t.std().clamp_min(eps)
        self.embed_calib_mean = embedding_scores.mean()
        self.embed_calib_std = embedding_scores.std().clamp_min(eps)

        residual_z = (residual_scores_t - self.res_calib_mean) / self.res_calib_std
        embedding_z = (embedding_scores - self.embed_calib_mean) / self.embed_calib_std
        combined = (
            self.config.residual_weight * residual_z
            + self.config.embedding_weight * embedding_z
        )
        self.threshold = torch.quantile(
            combined, torch.tensor(self.config.threshold_quantile)
        )
        self.fitted = True

        if was_training:
            self.model.train()
        return self

    def _window_residual(
        self,
        error: torch.Tensor,
        pixel_mask: torch.Tensor,
        divide_by_masks: int = 1,
    ) -> torch.Tensor:
        """Mean standardized masked-pixel error per window, shape ``(B,)``."""
        assert self.residual_mean is not None and self.residual_count is not None
        mean = self.residual_mean.to(error.device, error.dtype)
        count = self.residual_count.to(error.device)
        valid = pixel_mask.bool() & (count > 0)
        ratio = error / (mean + self.config.eps)
        total = torch.where(valid, ratio, torch.zeros_like(ratio)).sum(dim=(1, 2))
        denominator = valid.sum(dim=(1, 2)).clamp_min(1)
        return total / denominator / divide_by_masks

    @torch.no_grad()
    def score_windows(
        self,
        windows: torch.Tensor,
        batch_index: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Score a batch of normalized ``(B, C, T)`` windows."""
        if not self.fitted:
            raise RuntimeError("call fit() before score_windows()")
        was_training = self.model.training
        self.model.eval()

        x = windows.to(self.device, dtype=torch.float32)
        masks = self._eval_masks(x.shape[0], batch_index)
        window_score = torch.zeros(x.shape[0], dtype=torch.float64)
        projection: torch.Tensor | None = None
        for mask_index, mask in enumerate(masks):
            error, pixel_mask, output = self.model.per_pixel_error(x, mask)
            window_score += (
                self._window_residual(error, pixel_mask, divide_by_masks=len(masks))
                .double()
                .cpu()
            )
            if mask_index == 0:
                projection = output.proj.float().cpu()

        residual_score = window_score.float()
        embedding_score = self._mahalanobis(
            projection, self.embed_mean, self.embed_precision
        )
        residual_z = (residual_score - self.res_calib_mean) / self.res_calib_std
        embedding_z = (embedding_score - self.embed_calib_mean) / self.embed_calib_std
        score = (
            self.config.residual_weight * residual_z
            + self.config.embedding_weight * embedding_z
        )

        if was_training:
            self.model.train()
        return {
            "residual_score": residual_score,
            "embedding_score": embedding_score,
            "residual_z": residual_z,
            "embedding_z": embedding_z,
            "score": score,
            "threshold": self.threshold.expand_as(score),
            "is_anomaly": score > self.threshold,
        }

    @torch.no_grad()
    def score_loader(self, loader) -> dict[str, torch.Tensor]:
        """Score every window of a loader; results are concatenated."""
        outputs: dict[str, list[torch.Tensor]] = {}
        batch_index = 0
        for batch in loader:
            x = self._unpack(batch)
            scores = self.score_windows(x, batch_index=batch_index)
            for key, value in scores.items():
                outputs.setdefault(key, []).append(value)
            batch_index += 1
        return {key: torch.cat(values) for key, values in outputs.items()}

    def _fit_gaussian(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, d = embeddings.shape
        mean = embeddings.mean(dim=0)
        centered = embeddings - mean
        covariance = centered.t().mm(centered) / max(n - 1, 1)
        trace = torch.diagonal(covariance).mean().clamp_min(1e-8)
        eye = torch.eye(d, dtype=embeddings.dtype)
        covariance = (
            (1.0 - self.config.shrinkage) * covariance
            + self.config.shrinkage * trace * eye
            + 1e-6 * trace * eye
        )
        try:
            precision = torch.linalg.inv(covariance)
        except Exception:
            precision = torch.linalg.pinv(covariance)
        return mean, precision

    @staticmethod
    def _mahalanobis(
        x: torch.Tensor,
        mean: torch.Tensor,
        precision: torch.Tensor,
    ) -> torch.Tensor:
        difference = x - mean
        squared = torch.einsum("nd,de,ne->n", difference, precision, difference)
        return squared.clamp_min(0.0).sqrt()

    def save(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("call fit() before save()")
        state = {
            "config": asdict(self.config),
            "mask_config": asdict(self.mask_config),
            "num_patches": self.num_patches,
            "num_channels": self.num_channels,
            "window_size": self.window_size,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "residual_count": self.residual_count,
            "embed_mean": self.embed_mean,
            "embed_precision": self.embed_precision,
            "res_calib_mean": self.res_calib_mean,
            "res_calib_std": self.res_calib_std,
            "embed_calib_mean": self.embed_calib_mean,
            "embed_calib_std": self.embed_calib_std,
            "threshold": self.threshold,
        }
        torch.save(state, path)

    @classmethod
    def load(
        cls,
        path: str,
        model: VibrationTransformer,
    ) -> "AnomalyDetector":
        state = torch.load(path, map_location="cpu", weights_only=False)
        detector = cls(
            model,
            config=AnomalyConfig(**state["config"]),
            mask_config=MaskConfig(**state["mask_config"]),
        )
        for key in (
            "residual_mean",
            "residual_std",
            "residual_count",
            "embed_mean",
            "embed_precision",
            "res_calib_mean",
            "res_calib_std",
            "embed_calib_mean",
            "embed_calib_std",
            "threshold",
        ):
            setattr(detector, key, state[key])
        detector.fitted = True
        return detector
