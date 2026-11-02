"""Optional Torch Metal storage for finite two-stage covariance products."""
import numpy as np

from .finiteTwoStageProducts import DeviceFiniteProducts


class MpsFiniteProducts(DeviceFiniteProducts):
    """Evaluate shared device products in float32 on Apple Metal."""

    def __init__(self, backend, sampling):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('finiteApertureBackend=mps requires PyTorch with Metal support') from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError('finiteApertureBackend=mps requires available Torch Metal support')
        self.torch = torch
        super().__init__(backend, sampling)

    def upload(self, array, integer=False):
        return self.torch.tensor(np.asarray(array), device='mps',
            dtype=self.torch.int64 if integer else self.torch.float32)

    def zeros(self, shape):
        return self.torch.zeros(shape, device='mps', dtype=self.torch.float32)

    def empty(self, shape):
        return self.torch.empty(shape, device='mps', dtype=self.torch.float32)

    def to_host(self, array):
        return array.cpu().numpy().astype(float)
