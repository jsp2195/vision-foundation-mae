import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_rank0() -> bool:
    return rank() == 0


def init_distributed(enabled: bool) -> None:
    if not enabled or is_dist():
        return
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")


def cleanup_distributed() -> None:
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
