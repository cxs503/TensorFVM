"""Execution and decomposition primitives shared by 2-D and 3-D solvers.

The numerical kernels remain deliberately independent of a launcher.  This
module owns process/device discovery so a future distributed solver does not
spread ``torch.distributed`` conditionals through its physics implementation.
"""

from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class SlabPartition:
    """A contiguous half-open partition of a global index interval."""

    global_size: int
    rank: int
    world_size: int
    start: int
    stop: int

    @property
    def local_size(self) -> int:
        return self.stop - self.start

    @property
    def owns_lower_boundary(self) -> bool:
        return self.start == 0

    @property
    def owns_upper_boundary(self) -> bool:
        return self.stop == self.global_size


def partition_slab(global_size: int, rank: int, world_size: int) -> SlabPartition:
    """Balance ``global_size`` cells over ranks while preserving slab order."""
    if isinstance(global_size, bool) or not isinstance(global_size, int) or global_size < 1:
        raise ValueError("global_size must be a positive integer")
    if (isinstance(rank, bool) or isinstance(world_size, bool)
            or not isinstance(rank, int) or not isinstance(world_size, int)
            or world_size < 1 or not 0 <= rank < world_size):
        raise ValueError("rank must be in [0, world_size) and world_size must be positive")
    quotient, remainder = divmod(global_size, world_size)
    start = rank * quotient + min(rank, remainder)
    stop = start + quotient + (rank < remainder)
    return SlabPartition(global_size, rank, world_size, start, stop)


@dataclass(frozen=True)
class DistributedRuntime:
    """Immutable torch runtime metadata with an explicit single-rank fallback."""

    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    distributed: bool

    @classmethod
    def discover(cls, device: str = "cpu") -> "DistributedRuntime":
        """Read a launched torch process group, or create a safe local runtime.

        Initialising a process group is intentionally the launcher's task.  This
        keeps library imports side-effect free and works with ``torchrun`` as
        well as a normal Python invocation.
        """
        initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if initialized else 0
        world_size = torch.distributed.get_world_size() if initialized else 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        target = torch.device(device)
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA device requested but CUDA is unavailable")
            if target.index is None:
                target = torch.device("cuda", local_rank)
            torch.empty(0, dtype=torch.float64, device=target)
        elif target.type != "cpu":
            raise ValueError("runtime device must be cpu or cuda")
        return cls(target, rank, world_size, local_rank, initialized)

    def partition_z(self, nz: int) -> SlabPartition:
        """Return the z-slab owned by this rank for a 3-D structured mesh."""
        return partition_slab(nz, self.rank, self.world_size)

    def global_max(self, value: torch.Tensor) -> torch.Tensor:
        """Return the max across ranks without changing the input tensor."""
        result = value.clone()
        if self.distributed:
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.MAX)
        return result

    def global_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Return the sum across ranks without changing the input tensor."""
        result = value.clone()
        if self.distributed:
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result
