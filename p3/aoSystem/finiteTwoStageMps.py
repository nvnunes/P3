"""Optional Torch Metal storage for finite two-stage covariance products."""
import numpy as np
from scipy import ndimage

from .finiteTwoStageProducts import DeviceFiniteProducts


class MpsFiniteProducts(DeviceFiniteProducts):
    """Evaluate shared device products in float32 on Apple Metal."""

    host_covariance_dtype = np.float32
    kernel_cache_bytes = 512*1024**2
    direction_batch_size = 2

    def __init__(self, backend, sampling):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('finiteApertureBackend=mps requires PyTorch with Metal support') from exc
        if not torch.backends.mps.is_available():
            raise RuntimeError('finiteApertureBackend=mps requires available Torch Metal support')
        self.torch = torch
        super().__init__(backend, sampling)
        self._fused_query=None
        self.query_implementation='torch'
        self.query_fallback_reason=None
        if hasattr(torch.mps,'compile_shader'):
            from .finiteMpsQueries import FusedCubicQuery
            try:
                self._fused_query=FusedCubicQuery(self)
            except (RuntimeError,NotImplementedError) as exc:
                self.query_fallback_reason=str(exc)
            else:
                self.query_implementation='metal_cubic'
        try:
            probe=torch.zeros((2,2),device='mps',dtype=torch.complex64)
            torch.fft.ifft2(probe)
            torch.mps.synchronize()
        except (RuntimeError,NotImplementedError,TypeError):
            self.complex_feedback_supported=False
        else:
            self.complex_feedback_supported=True
        self.pin_kernel_pairs()

    def upload(self, array, integer=False):
        values=np.asarray(array)
        dtype=(self.torch.int64 if integer else
               self.torch.complex64 if np.iscomplexobj(values) else self.torch.float32)
        return self.torch.tensor(np.asarray(array), device='mps',
            dtype=dtype)

    def zeros(self, shape):
        return self.torch.zeros(shape, device='mps', dtype=self.torch.float32)

    def empty(self, shape):
        return self.torch.empty(shape, device='mps', dtype=self.torch.float32)

    def to_host(self, array):
        return array.cpu().numpy().astype(float)

    def copy(self,array):
        return array.clone()

    def evaluate_many(self,kernels,origins,contraction,output):
        """Evaluate unchanged cubic stencils with a bounded direction axis."""
        fused=getattr(self,'_fused_query',None)
        if fused is not None:
            fused.evaluate(kernels,origins,contraction,output)
            return
        q=self.query;torch=self.torch;batch=len(origins)
        cached=getattr(self,'_last_stencils',None)
        if cached is None or cached[1]!=contraction or not np.array_equal(cached[0],origins):
            x=(q.x[None,:,None]+origins[:,0,None,None]-contraction*q.cx[None,None])*q.scale
            ix,wx=self.stencil(x)
            ix=ix.reshape(batch,-1,4);wx=wx.reshape(batch,-1,4)
            vertical=[]
            for y,points,indices in self.blocks:
                coordinates=(y[None,:,None]+origins[:,1,None,None]-contraction*q.cy[None,None])*q.scale
                iy,wy=self.stencil(coordinates)
                vertical.append((iy.reshape(batch,-1,4),wy.reshape(batch,-1,4),points,indices))
            cached=(origins.copy(),contraction,ix,wx,vertical)
            self._last_stencils=cached
        _,_,ix,wx,vertical=cached
        axes=len(kernels);nx=ix.shape[1]
        for iy,wy,points,indices in vertical:
            intermediate=(kernels[:,iy,:]*wy[None,:,:,:,None]).sum(axis=3)
            ny=iy.shape[1]
            plane=intermediate[:,:,:,None,:].expand(axes,batch,ny,nx,q.grid)
            selectors=ix[None,:,None,:,:].expand(axes,batch,ny,nx,4)
            values=(torch.gather(plane,-1,selectors)*wx[None,:,None,:,:]).sum(axis=-1).swapaxes(-1,-2)
            output[:,:,points,:]=values.reshape(axes,batch,-1)[:,:,indices]

    def prefetch(self,fields):
        """Reuse each layer's device kernels across a bounded direction group."""
        if getattr(self,'_prefetched',None):
            raise RuntimeError('Consume the previous direction batch before preparing another')
        b=self.backend;s=self.sampling
        beta=b.native.ao.src.direction[:,list(fields)].T
        batch=len(fields)
        result=self.zeros((batch,3,len(s.xs),b.nc*b.nv))
        values=self.empty((2,batch,len(s.xs),b.nv))
        rate=b.native.ao.rtc.holoop['rate']
        for layer,(a,h) in enumerate(zip(b.contraction,b.heights)):
            angle=np.deg2rad(b.native.ao.atm.wDir[layer])
            shift=b.native.ao.atm.wSpeed[layer]*b.stage.delay/rate*np.array([np.cos(angle),np.sin(angle)])
            filtered,raw=self.kernel_pair(layer)
            for guide in range(b.ng):
                origin=h*(beta-b.guide[guide])-a*b.offsets[guide]
                delayed=origin-shift
                for stream,kernel,device_kernel,offset,gain in (
                    (0,b.interpolation[layer],filtered,origin,1.),
                    (1,b.raw_interpolation[layer],raw,delayed,b.stage.gain),
                    (2,b.interpolation[layer],filtered,delayed,b.stage.gain)):
                    self.evaluate_many(device_kernel,offset,a,values)
                    base=np.array([offset[:,1,None]-a*b.cells[None,:,1],
                                   offset[:,0,None]-a*b.cells[None,:,0]])*b.kernel_grid/b.period
                    fit=np.array([[ndimage.map_coordinates(kernel.fit[i,axis],base,order=3,
                        mode='grid-wrap',prefilter=False) for i in range(2)] for axis in range(2)])
                    fit=self.upload(fit.transpose(0,2,1,3))
                    corrected=values-self.ramps.T@fit
                    for axis in range(2):
                        channel=slice((2*guide+axis)*b.nv,(2*guide+axis+1)*b.nv)
                        result[:,stream,:,channel]+=gain*corrected[axis]
        self._prefetched={int(field):result[i] for i,field in enumerate(fields)}

    def crosses(self,field):
        pending=getattr(self,'_prefetched',{})
        if field in pending:return pending.pop(field)
        return super().crosses(field)

    def projected_atmosphere(self,projection,sampling,field):
        """Batch complete physical Fourier sums of the small feedback streams.

        Preparation and reconstruction remain float64. Separable phase vectors
        are evaluated in float64 before upload; no physical frequencies are
        identified after folding. Only the small field cross-factor returns.
        """
        b=self.backend;torch=self.torch
        cached=getattr(self,'_projected_modes',None)
        if cached is None or cached[0] is not projection or cached[1] is not sampling:
            layers=[self.upload((b.raw[layer]*b.dk**2)[None]*z.conj())
                    for layer,z in enumerate(projection.layers)]
            kernels=b.kernels.reshape(len(b.kernels),-1)
            real=self.upload(kernels.real);imag=self.upload(kernels.imag)
            ys=self.upload((sampling.ys-sampling.n//2)%sampling.n,True)
            xs=self.upload((sampling.xs-sampling.n//2)%sampling.n,True)
            index=self.upload((np.arange(b.count)-b.count//2)%sampling.n,True)
            cached=(projection,sampling,layers,real,imag,ys,xs,index)
            self._projected_modes=cached
        _,_,layers,kr,ki,ys,xs,index=cached
        beta=b.native.ao.src.direction[:,field]
        atmospheric=self.zeros((len(sampling.xs),3*b.nc))
        for layer,height in enumerate(b.heights):
            px=self.upload(np.exp(2j*np.pi*height*b.qx[:,0]*beta[0]))
            py=self.upload(np.exp(2j*np.pi*height*b.qy[0,:]*beta[1]))
            density=layers[layer]*(px[:,None]*py[None,:])[None]
            folded=torch.zeros((len(density),sampling.n,sampling.n),
                dtype=density.dtype,device=density.device)
            for first in range(0,b.count,sampling.n):
                last=min(first+sampling.n,b.count)
                for other in range(0,b.count,sampling.n):
                    end=min(other+sampling.n,b.count)
                    folded[:,index[None,other:end],index[first:last,None]]+=density[:,first:last,other:end]
            values=torch.fft.ifft2(folded).real*sampling.n**2
            dr=density.real.reshape(len(density),-1).T
            di=density.imag.reshape(len(density),-1).T
            moments=kr@dr-ki@di
            atmospheric+=values[:,ys,xs].T-self.ramps.T@moments
        return self.to_host(atmospheric)
