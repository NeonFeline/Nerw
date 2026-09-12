"""LeJEPA-style objective: masked reconstruction MSE + SIGReg.

The total loss follows the LeJEPA trade-off

    L = (1 - lamb) * MSE(masked spans) + lamb * SIGReg(projected embeddings)

where ``lamb`` is the single LeJEPA hyper-parameter (paper default ~0.02-0.05).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from model.backbone import ModelOutput
from model.sigreg import SIGReg


class LossOutput(NamedTuple):
    total: torch.Tensor
    mse: torch.Tensor
    sigreg: torch.Tensor


class LeJEPALoss(nn.Module):
    """Masked-reconstruction MSE + SIGReg regularization."""

    def __init__(
        self,
        sigreg_weight: float = 0.05,
        num_slices: int = 256,
        num_knots: int = 17,
    ) -> None:
        super().__init__()
        if not 0.0 <= sigreg_weight <= 1.0:
            raise ValueError("sigreg_weight must be in [0, 1]")
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg = SIGReg(num_slices=num_slices, num_knots=num_knots)

    def forward(
        self,
        output: ModelOutput,
        mask: torch.Tensor | None = None,
    ) -> LossOutput:
        error = (output.pred - output.patches).square()
        if mask is None or not bool(mask.any()):
            mse = error.mean()
        else:
            mse = error[mask].mean()
        sigreg = self.sigreg(output.proj)
        total = (1.0 - self.sigreg_weight) * mse + self.sigreg_weight * sigreg
        return LossOutput(total=total, mse=mse, sigreg=sigreg)
