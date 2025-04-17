from algo import Sort, GraB, RandomShuffle
from collections.abc import Callable
import math
import torch


class D_Sort(Sort):
    def __init__(
        self, rank: int, node_cnt: int, sort_maker: Callable[[], Sort]
    ) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.rank = rank
        self.sorter = sort_maker()

    def sort(self) -> torch.Tensor:
        """Return the sorted order of the gradient. To be implemented by child class."""
        return self.sorter.sort()

    def save_after_training(self, addr: str) -> None:
        """Save the sorted order of the gradient. To be implemented by child class."""
        pass


class CD_GraB(D_Sort):
    def __init__(
        self, rank: int, n: int, m: int, d: int, microbatch: int, device: str
    ) -> None:
        assert m % 2 == 0, "pair balance only supports even number"
        self.rank = rank
        self.n = n
        self.m = m
        self.d = d
        self.device = device
        self.microbatch = microbatch
        self.local_balance_step = microbatch // 2

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.vstack(
            [torch.arange(m, device=device) for _ in range(n)]
        )
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.m - 1

    @torch.no_grad()
    def reorder_online(self, batch_idx: int) -> None:
        """Reorders points in this batch based on gradient pair differences computed by step.

        If the inner product of the running sum of pair differences and the current pair difference 
        is negative, the pair difference is added to the running sum. Otherwise, the pair difference 
        is subtracted from the running sum.

        Args:
            batch_idx: range of indices in this node's microbatch 
              (e.g., torch.arange(idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device))
        """
        # grad at even step subtract grad at odd step
        for i, (idx_1, idx_2) in enumerate(batch_idx.view(len(batch_idx) // 2, 2)):
            for j in range(self.n):
                pair_diff = self.local_pair_diff_cache[j, i]
                if torch.inner(self.run_pair_diff_sum, pair_diff) <= 0:
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.add_(pair_diff)
                else:
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.sub_(pair_diff)
            self.left_ptr += 1
            self.right_ptr -= 1

    @torch.no_grad()
    def step(self, cur_grad: torch.Tensor, batch_idx: int) -> None:
        """Computes gradient pair differences and assigns new ordering for a dataloader batch.

        Args:
            cur_grad: microbatch gradients from each node (shape = (n, microbatch, d)) or (n, d). 
              Assumes cur_grad has even number of examples.
            batch_idx: range of indices in this node's microbatch 
              (e.g., torch.arange(idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device))
        """
        if cur_grad.dim() == 3 and cur_grad.shape[1] == self.microbatch:
            self.local_pair_diff_cache = (
                cur_grad[:, 1 : self.microbatch : 2, :] - cur_grad[:, ::2, :]
            )
        elif cur_grad.dim() == 2:
            self.local_pair_diff_cache = (
                cur_grad[1 : self.microbatch : 2, :] - cur_grad[::2, :]
            )
        else:
            raise RuntimeError(f"wrong shape of input: {cur_grad.shape}!")

        self.reorder_online(batch_idx)
        del self.local_pair_diff_cache

    @torch.no_grad()
    def sort(self) -> torch.Tensor:
        """Resets variables and returns the sorted order of the gradient."""
        self.left_ptr = 0
        self.right_ptr = self.m - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        self.run_pair_diff_sum.zero_()
        return self.orders.clone()[self.rank]


@torch.no_grad()
@torch.jit.script
def get_swap_decision(
    bsq: float, log_pen: float, twosig_sqd: float, diff_inner: float, U: float
) -> tuple[bool, float]:
    """Returns whether to add or subtract a pair difference according to the kernel halving
    rule along with an updated twosig_sqd sub-Gaussian parameter.
    
    Args:
        bsq: the squared two norm of the paired difference
        log_pen: the log penalty term for this pair (see KH_SGD.log_penalty)
        twosig_sqd: two times the squared sub-Gaussian constant of kernel halving
        diff_inner: inner product between the paired difference and the prior paired 
            difference sum
        U: a uniform[0,1] random variable
    """
    ratio = twosig_sqd * log_pen / bsq
    if ratio <= 1:
        # threshold = bsq >= sqrt(bsqd * twosig_sqd * log_pen)
        twosig_sqd = 2 * bsq
        if diff_inner <= 0:
            if U >= 0:
                add_pair_diff = True
            else:  # U < 0
                add_pair_diff = diff_inner <= U * bsq
        else:  # diff_inner > 0
            if U <= 0:
                add_pair_diff = False
            else:  # U > 0
                add_pair_diff = diff_inner <= U * bsq
        return (add_pair_diff, twosig_sqd)
    else:
        # threshold = sqrt(bsq * twosig_sqd * log_pen)
        ratio = torch.sqrt(ratio)
        # twosig_sqd += bsq * (2 + (bsq-2*thresh)*twosig_sqd/thresh^2)_+
        # <=> twosig_sqd += bsq * (2 + (bsq-2*thresh)/(bsq * log_pen))_+
        # <=> twosig_sqd += bsq * (2 + (1/log_pen-2*ratio/log_pen))_+
        # <=> twosig_sqd += 2 * bsq * (log_pen + .5 - ratio)_+ / log_pen
        update = log_pen + 0.5 - ratio
        if update > 0:
            twosig_sqd = twosig_sqd + 2 * bsq * update / log_pen
        if diff_inner <= 0:
            if U >= 0:
                add_pair_diff = True
            else:  # U < 0
                # thresh = ratio * bsq
                add_pair_diff = diff_inner <= U * ratio * bsq
        else:  # diff_inner > 0
            if U <= 0:
                add_pair_diff = False
            else:  # U > 0
                # thresh = ratio * bsq
                add_pair_diff = diff_inner <= U * ratio * bsq
        return (add_pair_diff, twosig_sqd)


class KH_SGD(D_Sort):
    """Permuted SGD with Kernel Halving (KH).

    Attributes:
        rank: the rank of the process in the network
        n: number of nodes
        m: datapoints processed per node
        d: gradient length
        microbatch: dataloader batch size // n = number of dataloader batch points per node
        device: Pytorch device
        delta: KH failure probability
        type: version of KH algorithm to run
        epochs: Total number of training epochs to perform.

    """

    def __init__(
        self,
        rank: int,
        n: int,
        m: int,
        d: int,
        microbatch: int,
        device: str,
        delta: float = 0.5,
        epochs: int = 0,
    ):
        """Initializes instance prior to training.

        NOTE: Currently assumes n = 1

        Args:
            rank: the rank of the process in the network
            n: number of nodes
            m: datapoints processed per node
            d: gradient length
            microbatch: dataloader batch size // n = number of dataloader batch points per node
            device: Pytorch device
            delta: KH failure probability
            type: version of KH algorithm to run
            epochs: Total number of training epochs to perform.

        """
        assert m % 2 == 0, "pair balance only supports even number"
        self.rank = rank
        self.n = n
        self.m = m
        self.d = d
        self.device = device
        self.delta = delta
        self.microbatch = microbatch
        self.local_balance_step = microbatch // 2
        self.epoch = 0
        self.epochs = epochs
        # Two times squared KH sub-Gaussian constant
        self.twosig_sqd = 0.0
        # Compute failure probability log penalty for initial epoch and each iterate of KH
        # Note: assumes n = 1
        m_over_2 = self.m // 2
        one_plus_log_m_over_2 = math.log(m_over_2) + 1
        self.log_penalty = torch.log(
            (4 / self.delta)
            * (math.log(self.epochs) + 1)
            * one_plus_log_m_over_2
            * torch.arange(1, m_over_2 + 1, device=self.device, dtype=torch.float32)
        )
        # Generate array of m//2 uniform[-1,1] random variables, one for each iterate in an epoch
        # Note: assumes n = 1
        self.rad = torch.empty(m_over_2, device=self.device).uniform_(-1, 1)

        # Keep track of iterate index into log_penalty array on each epoch
        self.mult_i = 0

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.vstack(
            [torch.arange(m, device=device) for _ in range(n)]
        )
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.m - 1

    @torch.no_grad()
    def reorder_online(self, batch_idx: int, sqd_pair_diff_norms: torch.Tensor) -> None:
        """Reorders points in this batch based on gradient pair differences computed by step.

        Args:
            batch_idx: range of indices in this node's microbatch 
              (e.g., torch.arange(idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device))
            sqd_pair_diff_norms: squared norm of the pair differences
        """
        # grad at even step subtract grad at odd step
        for i, (idx_1, idx_2) in enumerate(batch_idx.view(len(batch_idx) // 2, 2)):
            for j in range(self.n):
                pair_diff = self.local_pair_diff_cache[j, i]
                # Select KH threshold alpha and update sub-Gaussian parameter
                bsq = sqd_pair_diff_norms[j, i]
                log_pen = self.log_penalty[self.mult_i]

                # Compute inner product between product between pair_diff and running signed sum
                diff_inner = torch.inner(self.run_pair_diff_sum, pair_diff)
                # Get uniform[-1,1] random variable
                U = self.rad[self.mult_i]
                # Decide whether to add or subtract pair_diff from self.run_pair_diff_sum
                # and update sub-Gaussian parameter
                (add_pair_diff, self.twosig_sqd) = get_swap_decision(
                    bsq, log_pen, self.twosig_sqd, diff_inner, U
                )

                if add_pair_diff:
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.add_(pair_diff)
                else:
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.sub_(pair_diff)
                self.mult_i += 1
            self.left_ptr += 1
            self.right_ptr -= 1

    @torch.no_grad()
    def step(self, cur_grad: torch.Tensor, batch_idx: int) -> None:
        """Computes gradient pair differences and assigns new ordering for a dataloader batch.

        Args:
            cur_grad: microbatch gradients from each node (shape = (n, microbatch, d)) or (n, d). 
              Assumes cur_grad has even number of examples.
            batch_idx: range of indices in this node's microbatch 
              (e.g., torch.arange(idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device))
        """
        if cur_grad.dim() == 3 and cur_grad.shape[1] == self.microbatch:
            self.local_pair_diff_cache = (
                cur_grad[:, 1 : self.microbatch : 2, :] - cur_grad[:, ::2, :]
            )
        elif cur_grad.dim() == 2:
            self.local_pair_diff_cache = (
                cur_grad[1 : self.microbatch : 2, :] - cur_grad[::2, :]
            )
        else:
            raise RuntimeError(f"wrong shape of input: {cur_grad.shape}!")

        sqd_pair_diff_norms = self.local_pair_diff_cache.square().sum(dim=-1)
        self.reorder_online(batch_idx, sqd_pair_diff_norms)
        del self.local_pair_diff_cache

    @torch.no_grad()
    def sort(self) -> torch.Tensor:
        """Resets variables and returns the sorted order of the gradient."""
        if self.epoch > 0:
            # Reset pointers
            self.left_ptr = 0
            self.right_ptr = self.m - 1
            # Update to new ordering
            tmp = self.orders
            self.orders = self.next_orders
            self.next_orders = tmp
            # Reset running pair difference sum
            self.run_pair_diff_sum.zero_()
            # Reset sub-Gaussian parameter
            self.twosig_sqd = 0.0
            # Reset iterate index
            self.mult_i = 0
            # Generate independent uniform[-1,1] random variables for each iterate
            self.rad.uniform_(-1, 1)
            # Update penalty at the start of every epoch except the first
            self.log_penalty += math.log(1 + 1 / self.epoch)
        self.epoch += 1

        return self.orders.clone()[self.rank]


class SBW(D_Sort):
    """Self-balancing walk (SBW) for permuted SGD, (Alweiss, et al., 2021).

    Attributes:
        rank: the rank of the process in the network
        n: number of nodes
        m: datapoints processed per node
        d: gradient length
        microbatch: dataloader batch size // n = number of dataloader batch points per node
        device: Pytorch device
        delta: KH failure probability
        epochs: Total number of training epochs to perform.

    """

    def __init__(
        self,
        rank: int,
        n: int,
        m: int,
        d: int,
        microbatch: int,
        device: str,
        delta: float = 0.5,
        epochs: int = 0,
    ):
        """Initializes instance prior to training.

        NOTE: Currently assumes n = 1

        Args:
            rank: the rank of the process in the network
            n: number of nodes
            m: datapoints processed per node
            d: gradient length
            microbatch: dataloader batch size // n = number of dataloader batch points per node
            device: Pytorch device
            delta: KH failure probability
            epochs: Total number of training epochs to perform.

        """
        assert m % 2 == 0, "pair balance only supports even number"
        self.rank = rank
        self.n = n
        self.m = m
        self.d = d
        self.device = device
        self.delta = delta
        self.microbatch = microbatch
        self.local_balance_step = microbatch // 2
        self.type = type
        self.epoch = 0
        import math

        self.b_max = math.sqrt(20875)
        self.scale = self.b_max * 30 * math.log(m * epochs * d / delta / 2)

        # Generate array of uniforms size
        self.uniforms = torch.rand((m * n * epochs) // 2, device=device)
        self.rad = 1 - 2 * self.uniforms
        self.multipliers = self.scale * self.rad
        self.mult_i = 0

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.vstack(
            [torch.arange(m, device=device) for _ in range(n)]
        )
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.m - 1

    @torch.no_grad()
    def reorder_online(self, batch_idx: int) -> None:
        """Reorders points in this batch based on gradient pair differences computed by step.

        Args:
            batch_idx: range of indices in this node's microbatch 
              (e.g., torch.arange(idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device))
        """
        # grad at even step subtract grad at odd step
        for i, (idx_1, idx_2) in enumerate(batch_idx.view(len(batch_idx) // 2, 2)):
            for j in range(self.n):
                pair_diff = self.local_pair_diff_cache[j, i]
                if torch.inner(self.run_pair_diff_sum, pair_diff) <= self.get_threshold(
                    i, j
                ):
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.add_(pair_diff)
                else:
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.sub_(pair_diff)
                self.mult_i += 1
            self.left_ptr += 1
            self.right_ptr -= 1

    # we assume cur_grad has even number of examples.
    @torch.no_grad()
    # cur_grad: (n, microbatch, d) or (n, d)
    def step(self, cur_grad: torch.Tensor, batch_idx: int) -> None:
        """Computes gradient pair differences and assigns new ordering for a dataloader batch.

        Args:
            cur_grad: microbatch gradients from each node (shape = (n, microbatch, d))
            batch_idx: range of indices in this node's microbatch 
              (e.g., torch.arange(idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device))
        """
        if cur_grad.dim() == 3 and cur_grad.shape[1] == self.microbatch:
            self.local_pair_diff_cache = (
                cur_grad[:, 1 : self.microbatch : 2, :] - cur_grad[:, ::2, :]
            )
        elif cur_grad.dim() == 2:
            self.local_pair_diff_cache = (
                cur_grad[1 : self.microbatch : 2, :] - cur_grad[::2, :]
            )
        else:
            raise RuntimeError(f"wrong shape of input: {cur_grad.shape}!")

        self.reorder_online(batch_idx)
        del self.local_pair_diff_cache

    @torch.no_grad()
    def sort(self) -> torch.Tensor:
        """Resets variables and returns the sorted order of the gradient."""
        self.left_ptr = 0
        self.right_ptr = self.m - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        self.run_pair_diff_sum.zero_()
        return self.orders.clone()[self.rank]

    @torch.no_grad()
    def get_threshold(self, i: int, j: int, type: str = "constant") -> float:
        """Return the threshold for swapping points in the gradient pair difference."""
        return self.multipliers[self.mult_i]


class Independent_Balance(D_Sort):
    def __init__(self, rank: int, n: int, m: int, d: int, device: str) -> None:
        def sort_maker() -> GraB:
            return GraB(m, d, device=device)

        super().__init__(rank, n, sort_maker)

    def step(self, cur_grad: torch.Tensor, batch_idx: int) -> None:
        """Computes gradient pair differences and assigns new ordering for a dataloader batch. To be implemented by child class."""
        self.sorter.step(cur_grad, batch_idx)

    def sort(self) -> torch.Tensor:
        """Return the sorted order of the gradient and resets variables. To be implemented by child class."""
        return super().sort()


class D_RR(D_Sort):
    def __init__(self, rank: int, n: int, m: int, device: str = None) -> None:
        def sort_maker() -> RandomShuffle:
            return RandomShuffle(m, device=device)

        super().__init__(rank, n, sort_maker)
        self.num_batches = m
        self.device = device

    def step(self, *args: int, **kw: str) -> None:
        """Computes gradient pair differences and assigns new ordering for a dataloader batch."""
        pass

    def sort(self, *args: int, **kw: str) -> torch.Tensor:
        """Return the sorted order of the gradient and resets variables following D_Sort method."""
        return super().sort()

    def save_after_training(self, addr: str) -> None:
        """Save the sorted order of the gradient. To be implemented by child"""
        pass
