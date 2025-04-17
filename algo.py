import torch


class Sort:
    def sort(self) -> None:
        """Resets variables and returns the sorted order of the gradients."""
        raise NotImplementedError()


class GraB(Sort):
    def __init__(self, n: int, d: int, device: str = None) -> None:
        self.n = n
        self.d = d
        self.avg_grad = torch.zeros(d, device=device)
        self.cur_sum = torch.zeros_like(self.avg_grad)
        self.next_epoch_avg_grad = torch.zeros_like(self.avg_grad)
        self.orders = torch.arange(self.n, device=device, dtype=torch.int64)
        self.next_orders = self.orders.clone()
        self.left_ptr = 0
        self.right_ptr = self.n - 1

    @torch.no_grad()
    def sort(self) -> torch.Tensor:
        """Resets variables and returns the sorted order of the gradients."""
        self.avg_grad.copy_(self.next_epoch_avg_grad / self.n)
        self.next_epoch_avg_grad.zero_()
        self.cur_sum.zero_()
        self.left_ptr = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        return self.orders.clone()

    @torch.no_grad()
    def single_step(self, g: torch.Tensor, idx: int) -> None:
        """Updates the sorted order of the gradients with a single gradient pair."""
        self.next_epoch_avg_grad.add_(g)
        g = g - self.avg_grad
        if torch.inner(self.cur_sum, g) <= 0:
            self.next_orders[self.left_ptr] = self.orders[idx]
            self.left_ptr += 1
            self.cur_sum.add_(g)
        else:
            self.next_orders[self.right_ptr] = self.orders[idx]
            self.right_ptr -= 1
            self.cur_sum.sub_(g)

    @torch.no_grad()
    def step(self, batch_grads: torch.Tensor, batch_idx: int) -> None:
        """Updates the sorted order of the gradients with a batch of gradients."""
        for i, idx in enumerate(batch_idx):
            self.single_step(batch_grads[i], idx)


class GraB_Single(Sort):
    def __init__(self, n: int, d: int, device: str = None) -> None:
        self.n = n
        self.d = d
        self.avg_grad = torch.zeros(d, device=device)
        self.cur_sum = torch.zeros_like(self.avg_grad)
        self.next_epoch_avg_grad = torch.zeros_like(self.avg_grad)
        self.orders = torch.randperm(self.n, device=device, dtype=torch.int64)
        self.next_orders = self.orders.clone()
        self.left_ptr = 0
        self.right_ptr = self.n - 1

    @torch.no_grad()
    def sort(self) -> torch.Tensor:
        """Resets variables and returns the sorted order of the gradients."""
        self.avg_grad.copy_(self.next_epoch_avg_grad / self.n)
        self.next_epoch_avg_grad.zero_()
        self.cur_sum.zero_()
        self.left_ptr = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        return self.orders.clone()

    @torch.no_grad()
    def step(self, g: torch.Tensor, idx: int) -> None:
        """Updates the sorted order of the gradients with a single gradient pair."""
        self.next_epoch_avg_grad.add_(g)
        g = g - self.avg_grad
        if torch.inner(self.cur_sum, g) <= 0:
            self.next_orders[self.left_ptr] = self.orders[idx]
            self.left_ptr += 1
            self.cur_sum.add_(g)
        else:
            self.next_orders[self.right_ptr] = self.orders[idx]
            self.right_ptr -= 1
            self.cur_sum.sub_(g)


class RandomShuffle(Sort):
    def __init__(self, num_batches: int, device: str = None) -> None:
        super().__init__()
        self.device = device
        self.num_batches = num_batches

    def step(self, *args: str, **kw: int) -> None:
        """Random does not require sorting decisions. Do nothing."""
        pass

    def sort(self, *args: str, **kw: int) -> torch.Tensor:
        """Returns a random permutation of the batch indices."""
        return torch.randperm(self.num_batches, device=self.device)


class PairBalance_Sorter(Sort):
    def __init__(self, n: int, d: int, device: str = None) -> None:
        assert n % 2 == 0, "pair balance only supports even number"
        self.n = n
        self.d = d
        self.device = device

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.arange(n, device=device, dtype=torch.int64)
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.n - 1

    @torch.no_grad()
    # cur_grad: B, d
    def step(self, batch_grads: torch.Tensor, batch_idx: int) -> None:
        """Updates the sorted order of the gradients with a batch of gradients.

        Args:
            batch_grads: microbatch gradients from each node (shape = (B, d). Assumes batch_grads has even number of examples.
            batch_idx: range of indices in this node's microbatch (e.g., torch.arange(idx, min(idx + microbatch, d_trainset_X.shape[1]), device=device))

        """
        B = len(batch_idx)
        batch_grads = batch_grads[0:B:2] - batch_grads[1:B:2]
        for i, (idx_1, idx_2) in enumerate(batch_idx.view(B // 2, 2)):
            pair_diff = batch_grads[i]
            if torch.inner(self.run_pair_diff_sum, pair_diff) <= 0:
                self.next_orders[self.left_ptr] = self.orders[idx_1]
                self.next_orders[self.right_ptr] = self.orders[idx_2]
                self.run_pair_diff_sum.add_(pair_diff)
            else:
                self.next_orders[self.right_ptr] = self.orders[idx_1]
                self.next_orders[self.left_ptr] = self.orders[idx_2]
                self.run_pair_diff_sum.sub_(pair_diff)
            self.left_ptr += 1
            self.right_ptr -= 1

    @torch.no_grad()
    def sort(self) -> torch.Tensor:
        """Resets variables and returns the sorted order of the gradients."""
        self.left_ptr = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        self.run_pair_diff_sum.zero_()
        return self.orders.clone()


class PairBalance_Single(Sort):
    def __init__(self, n: int, d: int, device: str = None) -> None:
        assert n % 2 == 0, "pair balance only supports even number"
        self.n = n
        self.d = d
        self.device = device

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.randperm(n, device=device, dtype=torch.int64)
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.n - 1

    @torch.no_grad()
    def step(self, g: torch.Tensor, idx: int) -> None:
        """Updates the sorted order of the gradients with a single gradient pair.

        Args:
            g: gradient pair from a node (shape = (d))
            idx: index of the gradient pair in the node's microbatch

        """
        if idx % 2 == 0:
            self.cache = g.clone()
        else:
            self.cache = self.cache - g
            if torch.inner(self.run_pair_diff_sum, self.cache) <= 0:
                self.next_orders[self.left_ptr] = self.orders[idx - 1]
                self.next_orders[self.right_ptr] = self.orders[idx]
                self.run_pair_diff_sum.add_(self.cache)
            else:
                self.next_orders[self.right_ptr] = self.orders[idx - 1]
                self.next_orders[self.left_ptr] = self.orders[idx]
                self.run_pair_diff_sum.sub_(self.cache)
            self.left_ptr += 1
            self.right_ptr -= 1

    @torch.no_grad()
    def sort(self) -> torch.Tensor:
        """Resets variables and returns the sorted order of the gradients."""
        self.left_ptr = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        self.run_pair_diff_sum.zero_()
        return self.orders.clone()
