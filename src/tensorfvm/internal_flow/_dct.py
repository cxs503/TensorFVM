"""Orthonormal discrete cosine transforms of type II/III along one axis.

The staggered projection solver needs the eigenbasis of the cell-centred
Neumann Laplacian, ``phi_k(x_j) = cos(pi k (j + 1/2) / n)``, together with its
inverse.  Applying that basis as a dense matrix costs ``O(n^2 m)`` per
transform, which dominates the run time of the fine benchmark meshes (a
``1440 x 160`` backward-facing-step grid spends ~90% of a time step there).
This module implements the same transform pair through FFTs in ``O(n log n)``:

* forward (DCT-II): mirror the samples to length ``2n``, ``rfft``, and undo the
  half-sample phase ``exp(-i pi k / (2n))``,
* inverse (DCT-III, i.e. multiplication by the transpose): scale the spectrum,
  zero-pad it to length ``4n``, ``ifft``, and read the odd samples
  ``m = 2j + 1``.

Both are checked against the dense matrix in ``tests/test_internal_flow_dct.py``
to machine precision, so the FFT path cannot silently drift from the documented
basis.
"""

from __future__ import annotations

import math

import torch


def dct_matrix(n: int) -> torch.Tensor:
    """The orthonormal DCT-II matrix ``T[k, j] = a_k cos(pi k (j + 1/2) / n)``."""
    if n < 1:
        raise ValueError("n must be positive")
    k = torch.arange(n, dtype=torch.float64)[:, None]
    j = torch.arange(n, dtype=torch.float64)[None, :]
    mat = torch.cos(math.pi * k * (j + 0.5) / n)
    scale = torch.full((n, 1), math.sqrt(2.0 / n), dtype=torch.float64)
    scale[0] = math.sqrt(1.0 / n)
    return mat * scale


def _scale(n: int, like: torch.Tensor) -> torch.Tensor:
    scale = torch.full((n,), math.sqrt(2.0 / n), dtype=like.dtype, device=like.device)
    scale[0] = math.sqrt(1.0 / n)
    return scale


def dct2(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Orthonormal DCT-II of ``x`` along ``axis`` (same as ``dct_matrix(n) @ x``)."""
    n = x.shape[axis]
    moved = x.movedim(axis, -1)
    mirrored = torch.cat([moved, torch.flip(moved, dims=[-1])], dim=-1)
    spectrum = torch.fft.rfft(mirrored.double(), dim=-1)[..., :n]
    k = torch.arange(n, dtype=torch.float64, device=x.device)
    phase = torch.polar(torch.ones_like(k), -math.pi * k / (2.0 * n))
    rotated = torch.real(spectrum * phase)
    return (0.5 * rotated * _scale(n, x)).movedim(-1, axis).to(x.dtype)


def idct2(z: torch.Tensor, axis: int) -> torch.Tensor:
    """Inverse of :func:`dct2`, i.e. multiplication by its transpose."""
    n = z.shape[axis]
    weighted = z.movedim(axis, -1) * _scale(n, z)
    padded = torch.nn.functional.pad(weighted.double(), (0, 3 * n))
    spectrum = torch.fft.ifft(padded.to(torch.complex128), dim=-1) * (4.0 * n)
    index = torch.arange(1, 2 * n, 2, device=z.device)
    return torch.real(spectrum.index_select(-1, index)).movedim(-1, axis).to(z.dtype)