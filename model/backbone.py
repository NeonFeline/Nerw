"""Transformer backbone for DAS vibration reconstruction + LeJEPA embeddings.

The backbone tokenizes a ``(B, C, T)`` optical-cable window into
``L = T // patch_size`` tokens (each token covers all channels over one time
patch), replaces masked tokens with a learned mask token, and encodes the
sequence with a pre-norm Transformer encoder. A reconstruction head predicts
the raw patch values (supervised with MSE on masked spans) while a projection
head maps the pooled embedding into the space regularized by SIGReg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from model.masking import expand_token_mask


@dataclass
class ModelConfig:
    """Backbone configuration."""

    num_channels: int
    window_size: int = 1024
    patch_size: int = 16
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    n_decoder_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1
    proj_dim: int = 256
    proj_hidden: int = 512

    def __post_init__(self) -> None:
        if self.num_channels < 1:
            raise ValueError("num_channels must be >= 1")
        if self.window_size % self.patch_size != 0:
            raise ValueError("window_size must be divisible by patch_size")
        if self.window_size // self.patch_size < 2:
            raise ValueError("window_size // patch_size must be >= 2")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


class ModelOutput(NamedTuple):
    pred: torch.Tensor       # (B, L, C * P) reconstructed patches
    proj: torch.Tensor       # (B, proj_dim) SIGReg embedding
    pooled: torch.Tensor     # (B, d_model) mean-pooled encoder output
    tokens: torch.Tensor     # (B, L, d_model) encoder output
    patches: torch.Tensor    # (B, L, C * P) ground-truth patches
    mask: torch.Tensor | None  # (B, L) token mask used, if any


class VibrationTransformer(nn.Module):
    """Pre-norm Transformer encoder with reconstruction and SIGReg heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_patches = config.window_size // config.patch_size
        patch_dim = config.num_channels * config.patch_size

        self.patch_embed = nn.Linear(patch_dim, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        self.encoder = self._make_encoder(config.n_layers, config)
        self.decoder = (
            self._make_encoder(config.n_decoder_layers, config)
            if config.n_decoder_layers > 0
            else None
        )

        self.reconstruction_head = nn.Linear(config.d_model, patch_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.proj_hidden),
            nn.GELU(),
            nn.Linear(config.proj_hidden, config.proj_dim),
        )
        self._reset_parameters()

    @staticmethod
    def _make_encoder(num_layers: int, config: ModelConfig) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for module in (self.patch_embed, self.reconstruction_head):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` raw window -> ``(B, L, C * P)`` flattened patches."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T) input, got {tuple(x.shape)}")
        batch, channels, length = x.shape
        if channels != self.config.num_channels:
            raise ValueError(
                f"expected {self.config.num_channels} channels, got {channels}"
            )
        if length != self.config.window_size:
            raise ValueError(
                f"expected window_size {self.config.window_size}, got {length}"
            )
        patches = x.reshape(batch, channels, self.num_patches, self.config.patch_size)
        return patches.permute(0, 2, 1, 3).reshape(batch, self.num_patches, -1)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """``(B, L, C * P)`` flattened patches -> ``(B, C, T)`` raw window."""
        batch = patches.shape[0]
        patches = patches.reshape(
            batch, self.num_patches, self.config.num_channels, self.config.patch_size
        )
        return patches.permute(0, 2, 1, 3).reshape(
            batch, self.config.num_channels, self.config.window_size
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        """Encode a window and produce reconstructions + SIGReg embeddings.

        Args:
            x: normalized input window ``(B, C, T)``.
            mask: optional boolean ``(B, L)`` token mask; ``True`` positions
                are replaced by the learned mask token before encoding.
        """
        patches = self.patchify(x)
        tokens = self.patch_embed(patches) + self.pos_embed

        if mask is not None:
            if mask.shape != (x.shape[0], self.num_patches):
                raise ValueError(
                    f"mask must have shape {(x.shape[0], self.num_patches)}, "
                    f"got {tuple(mask.shape)}"
                )
            tokens = torch.where(
                mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens
            )

        tokens = self.encoder(tokens)
        decoded = self.decoder(tokens) if self.decoder is not None else tokens
        pred = self.reconstruction_head(decoded)
        pooled = tokens.mean(dim=1)
        proj = self.projector(pooled)

        return ModelOutput(
            pred=pred,
            proj=proj,
            pooled=pooled,
            tokens=tokens,
            patches=patches,
            mask=mask,
        )

    def per_pixel_error(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, ModelOutput]:
        """Squared reconstruction error and mask on the ``(B, C, T)`` grid.

        Returns:
            ``error`` (B, C, T) squared error at every pixel,
            ``pixel_mask`` (B, C, T) broadcast token mask,
            ``output`` the raw :class:`ModelOutput`.
        """
        output = self.forward(x, mask)
        error = (output.pred - output.patches).square()
        batch = x.shape[0]
        error = error.reshape(
            batch,
            self.num_patches,
            self.config.num_channels,
            self.config.patch_size,
        )
        error = error.permute(0, 2, 1, 3).reshape(batch, *x.shape[1:])
        pixel_mask = expand_token_mask(
            mask, self.config.num_channels, self.config.patch_size
        )
        return error, pixel_mask, output
