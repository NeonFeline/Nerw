"""Sketched Isotropic Gaussian Regularization (SIGReg) from LeJEPA.

Reference: R. Balestriero and Y. LeCun, "LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics", arXiv:2511.08544.

SIGReg pushes a batch of embeddings towards the standard isotropic Gaussian
N(0, I). By the Cramer-Wold theorem, matching N(0, I) is equivalent to
matching N(0, 1) along every 1-D projection, so the loss

  1. samples ``num_slices`` random unit directions,
  2. projects the embeddings onto each direction,
  3. averages the Epps-Pulley statistic over the slices, i.e. the weighted
     L2 distance between the empirical characteristic function of each
     projection and the characteristic function of N(0, 1).

The integral in step 3 is evaluated with the symmetric trapezoidal rule used
by the official LeJEPA implementation (integrate on [0, t_max] and double the
weights since the integrand is even).
"""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sliced Epps-Pulley statistic used as the LeJEPA regularization term.

    Args:
        num_slices: number of random projection directions.
        num_knots: number of trapezoidal quadrature knots on ``[0, t_max]``.
        t_max: upper bound of the integration domain.

    The module has no parameters; quadrature buffers move with ``.to(device)``.
    """

    def __init__(
        self,
        num_slices: int = 256,
        num_knots: int = 17,
        t_max: float = 3.0,
    ) -> None:
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if num_slices < 1:
            raise ValueError("num_slices must be >= 1")
        if t_max <= 0:
            raise ValueError("t_max must be > 0")

        t = torch.linspace(0.0, t_max, num_knots, dtype=torch.float32)
        dt = t_max / (num_knots - 1)
        weights = torch.full((num_knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-0.5 * t.square())

        self.register_buffer("t", t)
        self.register_buffer("phi", window.clone())
        self.register_buffer("weights", weights * window)
        self.num_slices = int(num_slices)

    def forward(
        self,
        embeddings: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Compute the SIGReg statistic.

        Args:
            embeddings: tensor of shape ``(N, D)`` with N >= 2.
            seed: optional seed for the random slicing directions. Seeding the
                same value across ranks keeps the directions in sync under DDP.

        Returns:
            Scalar tensor, zero (up to sampling noise) iff the embeddings are
            isotropic standard Gaussian.
        """
        if embeddings.dim() != 2:
            raise ValueError(
                f"embeddings must have shape (N, D), got {tuple(embeddings.shape)}"
            )
        n, d = embeddings.shape
        if n < 2:
            raise ValueError("SIGReg requires at least 2 embeddings per batch")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=embeddings.device)
            generator.manual_seed(int(seed))

        directions = torch.randn(
            d,
            self.num_slices,
            device=embeddings.device,
            dtype=embeddings.dtype,
            generator=generator,
        )
        eps = torch.finfo(embeddings.dtype).eps
        directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(eps)

        projections = embeddings @ directions
        angles = projections.unsqueeze(-1) * self.t

        ecf_real = angles.cos().mean(dim=0)
        ecf_imag = angles.sin().mean(dim=0)
        err = (ecf_real - self.phi).square() + ecf_imag.square()

        statistic = (err @ self.weights) * n
        return statistic.mean()
