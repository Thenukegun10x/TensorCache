"""
Asynchronous GPU Prefetcher.
Overlaps Host-to-Device (PCIe) transfers with GPU compute using double-buffered CUDA/HIP streams.
"""

from __future__ import annotations

from typing import Iterator, Any
import torch

class AsyncGPUPrefetcher:
    """
    Wraps any PyTorch DataLoader and asynchronously prefetches batches to GPU VRAM
    on a dedicated background CUDA/HIP stream.

    Usage:
        loader = DataLoader(dataset, batch_size=256, pin_memory=True)
        prefetcher = AsyncGPUPrefetcher(loader, device="cuda")
        for batch in prefetcher:
            # batch is already in VRAM!
            ...
    """
    def __init__(self, dataloader: Any, device: str = "cuda"):
        self.loader = dataloader
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type in ("cuda", "hip") else None
        self.iterator: Optional[Iterator] = None
        self.next_data: Any = None

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> AsyncGPUPrefetcher:
        self.iterator = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self.next_data = next(self.iterator)
        except StopIteration:
            self.next_data = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_data = self._to_device_async(self.next_data)

    def _to_device_async(self, data: Any) -> Any:
        if torch.is_tensor(data):
            return data.to(self.device, non_blocking=True)
        elif isinstance(data, (list, tuple)):
            return [self._to_device_async(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._to_device_async(v) for k, v in data.items()}
        return data

    def __next__(self) -> Any:
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            
        data = self.next_data
        if data is None:
            raise StopIteration
            
        self._preload()
        return data
