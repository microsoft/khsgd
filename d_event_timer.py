from contextlib import contextmanager
import resource
import gc
import torch


class EventTimer:
    """A timer that measures the time taken for each event, saved in nested dicts."""

    def __init__(self, device: torch.device) -> None:
        """Initialize the event timer. The timer is reset when initialized.

        Args:
            device (torch.device): The device the event timer is running on.

        """
        self.device = device
        # Warm-up GPU
        torch.randn(3, 3, device=device) @ torch.randn(3, 3, device=device)
        torch.cuda.empty_cache()
        gc.collect()
        self.reset()

    def reset(self) -> None:
        """Reset the timer"""
        self.initialized_keys = set()
        self.time_data = dict()  # the time for each occurence of each event
        self.cuda_max_mem_data = dict()
        self.cuda_allocated_mem_data = dict()
        self.ram_allocated_mem_data = dict()

    def create_label_if_not_exists(self, label: str) -> None:
        """Create a label in time, cuda max, cuda current, and ram if it does not exist."""
        # Update first and last occurrence of this label
        if label not in self.initialized_keys:
            self.time_data[label] = []
            self.cuda_max_mem_data[label] = []
            self.cuda_allocated_mem_data[label] = []
            self.ram_allocated_mem_data[label] = []
            self.initialized_keys.add(label)

    @contextmanager
    def __call__(self, label: str) -> None:
        """Measure the time taken for the code block inside the context manager.

        Args:
            label (str): The label for the event.

        """
        # Wait for everything before me to finish
        torch.cuda.current_stream().synchronize()

        # Measure the time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        cuda_mem_offset = torch.cuda.memory_allocated()
        mem_offset = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start.record()
        yield
        # Wait for operations that happen during yield to finish
        torch.cuda.current_stream().synchronize()
        end.record()

        # Need to wait once more for operations to finish
        torch.cuda.current_stream().synchronize()
        self.create_label_if_not_exists(label)

        self.time_data[label].append(start.elapsed_time(end) / 1000)  # seconds

        self.cuda_max_mem_data[label].append(
            (torch.cuda.max_memory_allocated() - cuda_mem_offset) / (1024 * 1024)
        )  # MiB
        self.cuda_allocated_mem_data[label].append(
            (torch.cuda.memory_allocated() - cuda_mem_offset) / (1024 * 1024)
        )
        self.ram_allocated_mem_data[label].append(
            (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem_offset) / 1024
        )  # MiB

        # torch.cuda.reset_max_memory_allocated()
        # torch.cuda.reset_peak_memory_stats()

    def summary(self) -> dict:
        """Return the summary of the timer.

        Returns:
            dict: A dictionary containing the time taken for each event, with:
            time: The time taken for each event.
            cuda-max: The maximum cuda memory used for each event.
            cuda-current: The current cuda memory allocated for each event.
            ram: The RAM memory allocated for each event.

        """
        return {
            "time": {k: torch.tensor(v) for k, v in self.time_data.items()},
            "cuda-max": {k: torch.tensor(v) for k, v in self.cuda_max_mem_data.items()},
            "cuda-current": {
                k: torch.tensor(v) for k, v in self.cuda_allocated_mem_data.items()
            },
            "ram": {k: torch.tensor(v) for k, v in self.ram_allocated_mem_data.items()},
        }

    def save_results(self, addr: str) -> None:
        """Save the results to a file.

        Args:
            addr (str): The address to save the results to.

        """
        ret = self.summary()
        torch.save(ret, addr)
