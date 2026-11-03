"""Optional CuPy CUDA storage for finite two-stage covariance products."""
import numpy as np
from scipy import ndimage

from .finiteTwoStageProducts import DeviceFiniteProducts


class CupyFiniteProducts(DeviceFiniteProducts):
    """Evaluate shared device products in float64 on the current CUDA device.

    CuPy is imported only when selected. No global P3 backend is changed and no
    CPU fallback is made when CuPy or its CUDA device is unavailable. Keep the
    same CUDA device current while using this object and its field results.
    """

    host_covariance_dtype = np.float64
    kernel_cache_bytes = 1024*1024**2
    direction_batch_size = 2

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
        from .finiteCudaQueries import FusedCubicQuery
        self._fused_query=FusedCubicQuery(self)
        self.query_implementation="cuda_cubic"
        self.complex_feedback_supported=True
        self.pin_kernel_pairs()

    def upload(self, array, integer=False):
        # Blocking upload keeps host scratch arrays safe to reuse immediately.
        return self.cupy.asarray(np.asarray(array),
            dtype=(self.cupy.int64 if integer else
                   self.cupy.complex128 if np.iscomplexobj(array) else self.cupy.float64), blocking=True)

    def zeros(self, shape):
        return self.cupy.zeros(shape, dtype=self.cupy.float64)

    def empty(self, shape):
        return self.cupy.empty(shape, dtype=self.cupy.float64)

    def to_host(self, array):
        return self.cupy.asnumpy(array)

    def evaluate_many(self, kernels, origins, contraction, output):
        """Evaluate identical cubic stencils with a bounded direction axis."""
        self._fused_query.evaluate(kernels, origins, contraction, output)

    def prefetch(self, fields):
        """Reuse resident kernels across at most two requested directions."""
        if not 1 <= len(fields) <= self.direction_batch_size:
            raise ValueError('CUDA queries require one bounded direction batch')
        if getattr(self, '_prefetched', None):
            raise RuntimeError('Consume the previous direction batch first')
        b=self.backend; s=self.sampling
        beta=b.native.ao.src.direction[:,list(fields)].T
        stage=b.stage if b.native.twoStageEnabled else None
        streams=3 if stage is not None else 1
        result=self.zeros((len(fields),streams,len(s.xs),b.nc*b.nv))
        values=self.empty((2,len(fields),len(s.xs),b.nv))
        rate=b.native.ao.rtc.holoop['rate']
        for layer,(a,h) in enumerate(zip(b.contraction,b.heights)):
            angle=np.deg2rad(b.native.ao.atm.wDir[layer])
            shift=(b.native.ao.atm.wSpeed[layer]*stage.delay/rate*
                   np.array([np.cos(angle),np.sin(angle)])) if stage is not None else 0.
            filtered,raw=self.kernel_pair(layer)
            for guide in range(b.ng):
                origin=h*(beta-b.guide[guide])-a*b.offsets[guide]
                work=[(0,b.interpolation[layer],filtered,origin,1.)]
                if stage is not None:
                    work.extend(((1,b.raw_interpolation[layer],raw,origin-shift,stage.gain),
                                 (2,b.interpolation[layer],filtered,origin-shift,stage.gain)))
                for stream,kernel,plane,offset,gain in work:
                    self.evaluate_many(plane,offset,a,values)
                    base=np.array([offset[:,1,None]-a*b.cells[None,:,1],
                                   offset[:,0,None]-a*b.cells[None,:,0]])*b.kernel_grid/b.period
                    fit=np.array([[ndimage.map_coordinates(kernel.fit[i,axis],base,order=3,
                        mode='grid-wrap',prefilter=False) for i in range(2)] for axis in range(2)])
                    corrected=values-self.ramps.T@self.upload(fit.transpose(0,2,1,3))
                    for axis in range(2):
                        channel=slice((2*guide+axis)*b.nv,(2*guide+axis+1)*b.nv)
                        result[:,stream,:,channel]+=gain*corrected[axis]
        self._prefetched={int(field):result[i] for i,field in enumerate(fields)}

    def crosses(self, field):
        pending=getattr(self, '_prefetched', {})
        if field not in pending:
            self.prefetch((field,))
            pending=self._prefetched
        return pending.pop(field)

    def projected_atmosphere(self, projection, sampling, field):
        """Full complex128 feedback sums; physical modes stay distinct before folding."""
        b=self.backend; cp=self.cupy
        cached=getattr(self,'_projected_modes',None)
        if cached is None or cached[0] is not projection or cached[1] is not sampling:
            layers=[self.upload((b.raw[layer]*b.dk**2)[None]*z.conj())
                    for layer,z in enumerate(projection.layers)]
            kernels=b.kernels.reshape(len(b.kernels),-1)
            cached=(projection,sampling,layers,self.upload(kernels.real),self.upload(kernels.imag),
                self.upload((sampling.ys-sampling.n//2)%sampling.n,True),
                self.upload((sampling.xs-sampling.n//2)%sampling.n,True),
                self.upload((np.arange(b.count)-b.count//2)%sampling.n,True))
            self._projected_modes=cached
        _,_,layers,kr,ki,ys,xs,index=cached
        beta=b.native.ao.src.direction[:,field]
        atmospheric=self.zeros((len(sampling.xs),3*b.nc))
        for layer,height in enumerate(b.heights):
            px=self.upload(np.exp(2j*np.pi*height*b.qx[:,0]*beta[0]))
            py=self.upload(np.exp(2j*np.pi*height*b.qy[0,:]*beta[1]))
            density=layers[layer]*(px[:,None]*py[None,:])[None]
            folded=cp.zeros((len(density),sampling.n,sampling.n),dtype=cp.complex128)
            for first in range(0,b.count,sampling.n):
                last=min(first+sampling.n,b.count)
                for other in range(0,b.count,sampling.n):
                    end=min(other+sampling.n,b.count)
                    folded[:,index[None,other:end],index[first:last,None]]+=density[:,first:last,other:end]
            values=cp.fft.ifft2(folded).real*sampling.n**2
            moments=kr@density.real.reshape(len(density),-1).T-ki@density.imag.reshape(len(density),-1).T
            atmospheric+=values[:,ys,xs].T-self.ramps.T@moments
        return self.to_host(atmospheric)
