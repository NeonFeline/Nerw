"""Span masking utilities for masked reconstruction of DAS windows.

DAS windows have shape ``(num_channels, window_size)`` and are tokenized into
``num_patches = window_size // patch_size`` tokens, each token covering all
channels over one time patch. A :class:`MaskConfig` describes how contiguous
time spans are sampled and masked for the reconstruction objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MaskConfig:
    """Configuration for contiguous span masking.

    If ``mask_ratio`` is set, exactly ~``mask_ratio * num_patches`` tokens are
    masked, split across ``num_spans`` spans. Otherwise each span length is
    drawn uniformly from ``[min_span, max_span]``.
    """

    num_spans: int = 4
    min_span: int = 2
    max_span: int = 8
    mask_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.num_spans < 1:
            raise ValueError("num_spans must be >= 1")
        if self.min_span < 1:
            raise ValueError("min_span must be >= 1")
        if self.max_span < self.min_span:
            raise ValueError("max_span must be >= min_span")
        if self.mask_ratio is not None and not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1) or None")


def generate_span_mask(
    batch_size: int,
    num_patches: int,
    mask_config: MaskConfig | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample contiguous span masks.

    Returns:
        Boolean tensor ``(batch_size, num_patches)`` where ``True`` marks a
        masked (to-be-reconstructed) token. Spans are truncated at the window
        boundary, so the realized ratio is approximate.
    """
    if batch_size < 1 or num_patches < 2:
        raise ValueError("batch_size must be >= 1 and num_patches must be >= 2")
    config = mask_config or MaskConfig()
    if device is None:
        device = torch.device("cpu")

    if config.mask_ratio is not None:
        target = max(1, min(num_patches - 1, round(config.mask_ratio * num_patches)))
        base = max(1, target // config.num_spans)
        lengths = torch.full(
            (batch_size, config.num_spans), base, dtype=torch.long, device=device
        )
        remainder = target - base * config.num_spans
        if remainder > 0:
            lengths[:, :remainder] += 1
        lengths = lengths.clamp_max(num_patches)
    else:
        lengths = torch.randint(
            config.min_span,
            config.max_span + 1,
            (batch_size, config.num_spans),
            generator=generator,
            device=device,
        )

    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    positions = torch.arange(num_patches, device=device).unsqueeze(0)
    for span in range(config.num_spans):
        starts = torch.randint(
            0, num_patches, (batch_size,), generator=generator, device=device
        )
        relative = positions - starts.unsqueeze(1)
        mask |= (relative >= 0) & (relative < lengths[:, span].unsqueeze(1))

    none_masked = ~mask.any(dim=1)
    if none_masked.any():
        count = int(none_masked.sum())
        picks = torch.randint(
            0, num_patches, (count,), generator=generator, device=device
        )
        mask[none_masked, picks] = True

    all_masked = mask.all(dim=1)
    if all_masked.any():
        count = int(all_masked.sum())
        picks = torch.randint(
            0, num_patches, (count,), generator=generator, device=device
        )
        mask[all_masked, picks] = False

    return mask


def expand_token_mask(
    token_mask: torch.Tensor,
    num_channels: int,
    patch_size: int,
) -> torch.Tensor:
    """Expand a ``(B, L)`` token mask to a ``(B, C, T)`` sample mask."""
    if token_mask.dim() != 2:
        raise ValueError("token_mask must have shape (B, L)")
    batch_size, num_patches = token_mask.shape
    return (
        token_mask.view(batch_size, 1, num_patches, 1)
        .expand(batch_size, num_channels, num_patches, patch_size)
        .reshape(batch_size, num_channels, num_patches * patch_size)
    )
