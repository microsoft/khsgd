import torch
from d_event_timer import EventTimer
from d_algo import CD_GraB, KH_SGD, SBW, D_RR
import torch.distributed as dist
import torchopt
import torch.nn.functional as F
from algo import Sort
from tqdm.auto import tqdm


def d_HMDA_train(
    cur_rank: int,
    d_trainset_X: torch.Tensor,
    d_trainset_y: torch.Tensor,
    func_per_example_grad: callable,
    fmodel: callable,
    params: list,
    buffers: list,
    optimizer: torchopt.Optimizer,
    opt_state: torchopt.FuncOptimizer,
    sorter: Sort,
    counter: tqdm,
    eventTimer: EventTimer,
    epoch: int,
    n: int,
    microbatch: int,
    d: int,
    device: str = None,
) -> None:
    """Distributed train loop for HMDA dataset.

    Note: EventTimer is a context manager that records the time taken by the block of code inside it.

    Args:
        cur_rank (int): Rank of the current process.
        d_trainset_X (torch.Tensor): Training data.
        d_trainset_y (torch.Tensor): Training labels.
        func_per_example_grad (callable): Function to compute per example gradients.
        fmodel (callable): Model to train.
        params (list): Model parameters.
        buffers (list): Model buffers.
        optimizer (torchopt.Optimizer): Optimizer.
        opt_state (torchopt.OptimizerState): Optimizer state.
        sorter (Sorter): Gradient sorter.
        counter (Counter): Counter object.
        eventTimer (EventTimer): EventTimer object.
        epoch (int): Current epoch.
        n (int): Number of training examples.
        microbatch (int): Microbatch size.
        d (int): Total model parameters.
        device (str): torch device.

    """
    with eventTimer(f"epoch-{epoch}"):
        with eventTimer("sorter"):
            perm_list = sorter.sort()

    if isinstance(sorter, CD_GraB):
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("communication"):
                gathered_grads = torch.empty(n, microbatch, d, device=device)

    if isinstance(sorter, KH_SGD) or isinstance(sorter, SBW):
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("communication"):
                gathered_grads = torch.empty(n, microbatch, d, device=device)

    for idx in range(0, d_trainset_X.shape[1], microbatch):
        batch = torch.arange(
            idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device
        )
        # Using the obtained order, we get the training examples
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("dataset"):
                X = d_trainset_X[cur_rank][perm_list[batch]]
                y = d_trainset_y[cur_rank][perm_list[batch]]

        if isinstance(sorter, D_RR):
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("forward-backward"):
                    avg_grads = torch.autograd.grad(
                        F.binary_cross_entropy_with_logits(
                            fmodel(params, buffers, X).squeeze(), y
                        ),
                        params,
                    )
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])
                with torch.no_grad():
                    with eventTimer("communication"):
                        dist.all_reduce(avg_grads, op=dist.ReduceOp.SUM)
                        avg_grads /= n

        elif isinstance(sorter, CD_GraB):
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("forward-backward"):
                    avg_grads = torch.autograd.grad(
                        F.binary_cross_entropy_with_logits(
                            fmodel(params, buffers, X).squeeze(), y
                        ),
                        params,
                    )
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])

            with torch.no_grad():
                with eventTimer(f"epoch-{epoch}"):
                    with eventTimer("communication"):
                        dist.all_gather_into_tensor(
                            gathered_grads, avg_grads, async_op=False
                        )
                        avg_grads = gathered_grads.mean(dim=0)

                    with eventTimer("sorter"):
                        sorter.step(gathered_grads, batch)

        elif isinstance(sorter, KH_SGD) or isinstance(sorter, SBW):
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("forward-backward"):
                    avg_grads = torch.autograd.grad(
                        F.binary_cross_entropy_with_logits(
                            fmodel(params, buffers, X).squeeze(), y
                        ),
                        params,
                    )
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])

            with torch.no_grad():
                with eventTimer(f"epoch-{epoch}"):
                    with eventTimer("communication"):
                        dist.all_gather_into_tensor(
                            gathered_grads, avg_grads, async_op=False
                        )
                        avg_grads = gathered_grads.mean(dim=0)

                    with eventTimer("sorter"):
                        sorter.step(gathered_grads, batch)

        else:
            raise NotImplementedError()

        with torch.no_grad():
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("SGD"):
                    # compute gradient and do SGD step
                    avg_grad_list = []
                    grad_cnt = 0
                    for p in params:
                        avg_grad_list.append(
                            avg_grads[grad_cnt : p.numel() + grad_cnt].view(p.shape)
                        )
                        grad_cnt += p.numel()

                    updates, opt_state = optimizer.update(
                        avg_grad_list, opt_state, params=params
                    )
                    torchopt.apply_updates(params, tuple(updates), inplace=True)
            if cur_rank == 0:
                counter.update(len(batch))


def d_HMDA_train(
    cur_rank: int,
    d_trainset_X: torch.Tensor,
    d_trainset_y: torch.Tensor,
    func_per_example_grad: callable,
    fmodel: callable,
    params: list,
    buffers: list,
    optimizer: torchopt.Optimizer,
    opt_state: torchopt.FuncOptimizer,
    sorter: Sort,
    counter: tqdm,
    eventTimer: EventTimer,
    epoch: int,
    n: int,
    microbatch: int,
    d: int,
    device: str = None,
) -> None:
    """Distributed train loop for HMDA dataset.

    Note: EventTimer is a context manager that records the time taken by the block of code inside it.

    Args:
        cur_rank (int): Rank of the current process.
        d_trainset_X (torch.Tensor): Training data.
        d_trainset_y (torch.Tensor): Training labels.
        func_per_example_grad (callable): Function to compute per example gradients.
        fmodel (callable): Model to train.
        params (list): Model parameters.
        buffers (list): Model buffers.
        optimizer (torchopt.Optimizer): Optimizer.
        opt_state (torchopt.OptimizerState): Optimizer state.
        sorter (Sorter): Gradient sorter.
        counter (Counter): Counter object.
        eventTimer (EventTimer): EventTimer object.
        epoch (int): Current epoch.
        n (int): Number of training examples.
        microbatch (int): Microbatch size.
        d (int): Total model parameters.
        device (str): torch device..

    """
    with eventTimer(f"epoch-{epoch}"):
        with eventTimer("sorter"):
            perm_list = sorter.sort()

    if isinstance(sorter, CD_GraB):
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("communication"):
                gathered_grads = torch.empty(n, microbatch, d, device=device)

    if isinstance(sorter, KH_SGD) or isinstance(sorter, SBW):
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("communication"):
                gathered_grads = torch.empty(n, microbatch, d, device=device)

    for idx in range(0, d_trainset_X.shape[1], microbatch):
        batch = torch.arange(
            idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device
        )
        # Using the obtained order, we get the training examples
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("dataset"):
                X = d_trainset_X[cur_rank][perm_list[batch]]
                y = d_trainset_y[cur_rank][perm_list[batch]]

        if isinstance(sorter, D_RR):
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("forward-backward"):
                    avg_grads = torch.autograd.grad(
                        F.binary_cross_entropy_with_logits(
                            fmodel(params, buffers, X).squeeze(), y
                        ),
                        params,
                    )
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])
                with torch.no_grad():
                    with eventTimer("communication"):
                        dist.all_reduce(avg_grads, op=dist.ReduceOp.SUM)
                        avg_grads /= n

        elif isinstance(sorter, CD_GraB):
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("forward-backward"):
                    per_example_grads = func_per_example_grad(params, buffers, X, y)
                    with torch.no_grad():
                        per_example_grads = torch.hstack(
                            [
                                g.view(g.shape[0], g.numel() // g.shape[0])
                                for g in per_example_grads
                            ]
                        )

            with torch.no_grad():
                with eventTimer(f"epoch-{epoch}"):
                    with eventTimer("communication"):
                        dist.all_gather_into_tensor(
                            gathered_grads, per_example_grads, async_op=False
                        )
                        avg_grads = gathered_grads.mean(dim=(0, 1))

                    with eventTimer("sorter"):
                        sorter.step(gathered_grads, batch)

        elif isinstance(sorter, KH_SGD) or isinstance(sorter, SBW):
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("forward-backward"):
                    per_example_grads = func_per_example_grad(params, buffers, X, y)
                    with torch.no_grad():
                        per_example_grads = torch.hstack(
                            [
                                g.view(g.shape[0], g.numel() // g.shape[0])
                                for g in per_example_grads
                            ]
                        )

            with torch.no_grad():
                with eventTimer(f"epoch-{epoch}"):
                    with eventTimer("communication"):
                        dist.all_gather_into_tensor(
                            gathered_grads, per_example_grads, async_op=False
                        )
                        avg_grads = gathered_grads.mean(dim=(0, 1))

                    with eventTimer("sorter"):
                        sorter.step(gathered_grads, batch)

        else:
            raise NotImplementedError()

        with torch.no_grad():
            with eventTimer(f"epoch-{epoch}"):
                with eventTimer("SGD"):
                    # compute gradient and do SGD step
                    avg_grad_list = []
                    grad_cnt = 0
                    for p in params:
                        avg_grad_list.append(
                            avg_grads[grad_cnt : p.numel() + grad_cnt].view(p.shape)
                        )
                        grad_cnt += p.numel()

                    updates, opt_state = optimizer.update(
                        avg_grad_list, opt_state, params=params
                    )
                    torchopt.apply_updates(params, tuple(updates), inplace=True)
            if cur_rank == 0:
                counter.update(len(batch))
