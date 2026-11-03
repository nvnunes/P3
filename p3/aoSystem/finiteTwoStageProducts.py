"""Shared device arithmetic for finite two-stage optical covariance."""
import numpy as np
from scipy import ndimage
from collections import OrderedDict


class DeviceFiniteProducts:
    """Resident two-stage queries and covariance products.

    Device subclasses supply allocation, upload and host transfer. Reconstruction
    and host interpolation moments remain float64. Field objects retain their
    directional arrays independently of later queries.
    """
    def __init__(self, backend, sampling):
        self.backend, self.sampling = backend, sampling
        self.query = backend._science_query
        self.common_rows = self.upload(backend.weights)
        self.common_operator = self.upload(backend.common_basis)
        self.feedback = self.upload(backend.feedback)
        self.cbb = self.upload(backend.cbb)
        self.common_cross = self.upload(backend.common_science_cross)
        self.ramps = self.upload(sampling.ramps)
        self.blocks = [(y, self.upload(points, integer=True), self.upload(indices, integer=True))
            for y,points,indices in self.query.blocks]
        self._resident_background = None
        self._kernel_cache = OrderedDict()
        self._pinned_kernel_pairs = None
        self.kernel_cache_info = dict(hits=0,misses=0,resident_bytes=0,
                                     limit_bytes=getattr(self,'kernel_cache_bytes',0))

    def kernel_pair(self,layer):
        """Reuse immutable sampling-owned kernels within an explicit device budget."""
        cache=self._kernel_cache;stats=self.kernel_cache_info
        if self._pinned_kernel_pairs is not None:
            stats['hits']+=1
            return self._pinned_kernel_pairs[layer]
        if layer in cache:
            stats['hits']+=1
            cache.move_to_end(layer)
            return cache[layer][0]
        stats['misses']+=1
        pair=(self.upload(self.backend.interpolation[layer].host_phase()),
              self.upload(self.backend.raw_interpolation[layer].host_phase()))
        size=sum(value.numel()*value.element_size() if hasattr(value,'numel') else value.nbytes for value in pair)
        limit=stats['limit_bytes']
        if size<=limit:
            while cache and stats['resident_bytes']+size>limit:
                _,(_,removed)=cache.popitem(last=False)
                stats['resident_bytes']-=removed
            cache[layer]=(pair,size)
            stats['resident_bytes']+=size
        return pair

    def pin_kernel_pairs(self):
        """Release host phase only when every layer can stay device-resident."""
        if self._pinned_kernel_pairs is not None:
            return True
        families=(self.backend.interpolation,self.backend.raw_interpolation)
        dtype=np.dtype(getattr(self,'host_covariance_dtype',np.float64))
        required=sum(layer.host_phase().size*dtype.itemsize for family in families for layer in family)
        if required>self.kernel_cache_info['limit_bytes']:
            return False
        pairs=tuple(self.kernel_pair(i) for i in range(len(self.backend.heights)))
        released=0
        for family in families:
            for layer in family:
                released+=layer.phase.nbytes
                layer.phase=None
        self._pinned_kernel_pairs=pairs
        self.kernel_cache_info.update(pinned_layers=len(pairs),released_host_bytes=released)
        return True

    def resident_covariance(self,residual):
        """Keep complete covariance tiles on this device, with owner-bound caches."""
        from .finiteResident import ResidentCovariance
        return ResidentCovariance(self,residual)

    def copy(self,array):
        return array.copy()

    def upload(self, array, integer=False):
        """Copy a host array into device-owned floating or integer storage."""
        raise NotImplementedError

    def zeros(self, shape):
        """Allocate a zeroed device floating array."""
        raise NotImplementedError

    def empty(self, shape):
        """Allocate an uninitialized device floating array."""
        raise NotImplementedError

    def to_host(self, array):
        """Synchronize and return a host float64 array."""
        raise NotImplementedError

    def stencil(self, coordinates):
        """Identical grid-wrap cubic coefficients; coordinates computed in f64."""
        c = np.remainder(coordinates.ravel(),self.query.grid)
        index = np.floor(c).astype(np.int64); t = c-index
        weights = np.stack(((1-t)**3,3*t**3-6*t*t+4,
            -3*t**3+3*t*t+3*t+1,t**3),axis=1)/6
        indices = (index[:,None]+np.arange(-1,3))%self.query.grid
        return self.upload(indices,True),self.upload(weights)

    def evaluate(self,kernels,origin,contraction,output):
        q = self.query
        ix,wx = self.stencil((q.x[:,None]+origin[0]-contraction*q.cx[None])*q.scale)
        for y,points,indices in self.blocks:
            iy,wy = self.stencil((y[:,None]+origin[1]-contraction*q.cy[None])*q.scale)
            # Separable four-point gathers avoid a pupil x lenslet x 16 cube.
            intermediate = (kernels[:,iy,:]*wy[None,:,:,None]).sum(axis=2)
            values = (intermediate[:,:,ix]*wx[None,None,:,:]).sum(axis=-1).swapaxes(1,2)
            output[:,points,:] = values.reshape(2,-1)[:,indices]

    def crosses(self,field):
        b,s = self.backend,self.sampling
        beta = b.native.ao.src.direction[:,field]
        result = self.zeros((3,len(s.xs),b.nc*b.nv))
        values = self.empty((2,len(s.xs),b.nv))
        rate = b.native.ao.rtc.holoop['rate']
        for layer,(a,h) in enumerate(zip(b.contraction,b.heights)):
            angle = np.deg2rad(b.native.ao.atm.wDir[layer])
            shift = b.native.ao.atm.wSpeed[layer]*b.stage.delay/rate*np.array([np.cos(angle),np.sin(angle)])
            # Host fit moments retain float64; device kernels are sampling-owned.
            filtered,raw = self.kernel_pair(layer)
            for guide in range(b.ng):
                origin = h*(beta-b.guide[guide])-a*b.offsets[guide]
                for stream,kernel,device_kernel,offset,gain in (
                    (0,b.interpolation[layer],filtered,origin,1.),
                    (1,b.raw_interpolation[layer],raw,origin-shift,b.stage.gain),
                    (2,b.interpolation[layer],filtered,origin-shift,b.stage.gain)):
                    self.evaluate(device_kernel,offset,a,values)
                    base = np.array([offset[1]-a*b.cells[:,1],offset[0]-a*b.cells[:,0]])*b.kernel_grid/b.period
                    fit = np.array([[ndimage.map_coordinates(kernel.fit[i,axis],base,order=3,
                        mode='grid-wrap',prefilter=False) for i in range(2)] for axis in range(2)])
                    corrected = values-(self.ramps.T@self.upload(fit))
                    for axis in range(2):
                        channel = slice((2*guide+axis)*b.nv,(2*guide+axis+1)*b.nv)
                        result[stream,:,channel] += gain*corrected[axis]
            del filtered,raw
        return result

    def context(self, crosses, science_rows, science_basis):
        """Retain exact command-space factors for both physical mirrors."""
        common = crosses[0]@self.common_rows.T
        rows = self.upload(science_rows)
        operator = self.upload(science_basis)
        command_covariance = (rows@self.cbb)@rows.T
        update = .5*(operator@command_covariance)+self.common_cross@rows.T-crosses[1]@rows.T
        update += crosses[2]@(self.feedback.T@rows.T)
        return self.to_host(common), _DeviceField(self, common, operator, update)


class _DeviceField:
    def __init__(self,owner,common,operator,update):
        self.owner,self.common,self.operator,self.update = owner,common,operator,update

    def diagonal(self):
        return self.owner.to_host(2*(self.update*self.operator).sum(axis=1))

    def reporting_factors(self):
        """Copy covariance factors to the host for optional scalar accounting."""
        return tuple(self.owner.to_host(value) for value in
                     (self.operator, self.update, self.owner.common_operator, self.common))

    def block(self,first,last):
        return self.owner.to_host(self.block_array(first,last))

    def block_array(self,first,last):
        """Return the same device tile without a host synchronization."""
        o1=self.owner.common_operator; o2=self.operator; d=self.update; x=self.common
        value = d[first:last]@o2[first:].T+o2[first:last]@d[first:].T
        value -= x[first:last]@o1[first:].T+o1[first:last]@x[first:].T
        return value
