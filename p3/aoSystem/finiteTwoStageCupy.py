"""Optional CuPy CUDA storage for finite two-stage covariance products."""
import numpy as np

from .finiteTwoStageProducts import DeviceFiniteProducts


class CupyFiniteProducts(DeviceFiniteProducts):
    """Evaluate shared device products in float64 on the current CUDA device.

    CuPy is imported only when selected. No global P3 backend is changed and no
    CPU fallback is made when CuPy or its CUDA device is unavailable. Keep the
    same CUDA device current while using this object and its field results.
    """

    def __init__(self, backend, sampling):
        try:
            import cupy
        except ImportError as exc:
            raise RuntimeError('finiteApertureBackend=cuda requires CuPy with CUDA support') from exc
        try:
            available = cupy.is_available()
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            raise RuntimeError('finiteApertureBackend=cuda requires an available CUDA device') from exc
        if not available:
            raise RuntimeError('finiteApertureBackend=cuda requires an available CUDA device')
        self.cupy = cupy
        super().__init__(backend, sampling)

    def upload(self, array, integer=False):
        # Blocking upload keeps host scratch arrays safe to reuse immediately.
        return self.cupy.asarray(np.asarray(array),
            dtype=self.cupy.int64 if integer else self.cupy.float64, blocking=True)

    def zeros(self, shape):
        return self.cupy.zeros(shape, dtype=self.cupy.float64)

    def empty(self, shape):
        return self.cupy.empty(shape, dtype=self.cupy.float64)

    def to_host(self, array):
        return self.cupy.asnumpy(array)
