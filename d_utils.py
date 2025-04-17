import torch
import random
import os
import numpy as np


def print_rank_0(rank: int, *args: str, **kw: int) -> None:
    """Print only for rank 0."""
    if rank == 0:
        print(*args, **kw, flush=True)


def seed_everything(seed: int) -> None:
    """Add manual seed to all random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
