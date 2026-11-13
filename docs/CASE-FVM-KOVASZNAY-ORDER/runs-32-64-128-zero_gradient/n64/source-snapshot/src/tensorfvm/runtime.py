"""Execution and decomposition primitives shared by 2-D and 3-D solvers.

The numerical kernels remain deliberately independent of a launcher.  This
module owns process/device discovery, collective reductions, and the reusable
periodic z-halo primitive used by distributed structured-grid solvers.
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
    """Immutable torch runtime metadata with an explicit single-rank fallback.

    ``discover`` has no side effects: library callers must already have created
    a process group.  Executables launched through ``torchrun`` can opt into
    :meth:`initialize_from_environment` before discovery.
    """

    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    distributed: bool

    @classmethod
    def discover(cls, device: str = "cpu") -> "DistributedRuntime":
        """Read an existing torch process group, or create a safe local runtime."""
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

    @classmethod
    def initialize_from_environment(cls, device: str = "cpu") -> "DistributedRuntime":
        """Initialise a ``torchrun`` process group when multiple ranks were requested.

        This is intentionally an explicit executable-level operation.  A normal
        Python import or a library call to :meth:`discover` never creates a
        process group.  CPU launchers use Gloo; CUDA launchers use NCCL and bind
        each rank to its ``LOCAL_RANK`` device before initialisation.
        """
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
        if torch.distributed.is_initialized():
            return cls.discover(device)
        try:
            requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError as exc:
            raise ValueError("WORLD_SIZE must be an integer when set") from exc
        if requested_world_size < 1:
            raise ValueError("WORLD_SIZE must be positive when set")
        if requested_world_size == 1:
            return cls.discover(device)
        target = torch.device(device)
        if target.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA device requested but CUDA is unavailable")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            target = torch.device("cuda", target.index if target.index is not None else local_rank)
            torch.cuda.set_device(target)
            backend = "nccl"
        elif target.type == "cpu":
            backend = "gloo"
        else:
            raise ValueError("runtime device must be cpu or cuda")
        torch.distributed.init_process_group(backend=backend)
        return cls.discover(str(target))

    def partition_z(self, nz: int) -> SlabPartition:
        """Return the z-slab owned by this rank for a 3-D structured mesh."""
        return partition_slab(nz, self.rank, self.world_size)

    def global_max(self, value: torch.Tensor) -> torch.Tensor:
        """Return the maximum across ranks without changing the input tensor."""
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

    def barrier(self) -> None:
        """Synchronise ranks, or do nothing for a local runtime."""
        if self.distributed:
            torch.distributed.barrier()

    def periodic_z_halos(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the lower and upper periodic one-cell halos for a local z slab.

        ``field`` is partitioned along dimension zero.  The returned tensors
        have shape ``field.shape[1:]`` and are independent allocations, so they
        may safely be retained while the caller updates its local field.  Two
        tagged nonblocking sends and receives are used rather than a global
        gather; the same protocol supports the two-rank case where lower and
        upper neighbours are the same rank.
        """
        if field.ndim < 1 or field.shape[0] < 1:
            raise ValueError("periodic z halo exchange requires a nonempty leading dimension")
        if self.world_size == 1:
            return field[-1].clone(), field[0].clone()
        if not self.distributed:
            raise RuntimeError("a multi-rank runtime requires an initialized process group")
        lower_rank = (self.rank - 1) % self.world_size
        upper_rank = (self.rank + 1) % self.world_size
        lower_halo = torch.empty_like(field[0])
        upper_halo = torch.empty_like(field[-1])
        # A lower rank sends its final plane upward with tag 8100 and its first
        # plane downward with tag 8101.  Tags keep both directions unambiguous
        # when world_size == 2.
        requests = (
            torch.distributed.irecv(lower_halo, src=lower_rank, tag=8100),
            torch.distributed.irecv(upper_halo, src=upper_rank, tag=8101),
            torch.distributed.isend(field[-1].contiguous(), dst=upper_rank, tag=8100),
            torch.distributed.isend(field[0].contiguous(), dst=lower_rank, tag=8101),
        )
        for request in requests:
            request.wait()
        return lower_halo, upper_halo
