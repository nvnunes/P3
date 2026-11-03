"""Float64 periodic cubic queries at valid pupil/lenslet pairs on CUDA."""
import numpy as np

_SOURCE = r'''
extern "C" __global__ void cubic_pairs(
 const double* kernels, const long long* ix, const double* wx,
 const long long* iy, const double* wy, const long long* gx, const long long* gy,
 double* output, int pupil, int cells, int batch, int grid,
 int xlook, int ylook, int cellx, int celly, long long total) {
 long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(index>=total)return;
 int cell=index%cells, point=(index/cells)%pupil;
 int direction=(index/((long long)cells*pupil))%batch;
 int axis=index/((long long)cells*pupil*batch);
 long long x=4*(direction*xlook+gx[point]*cellx+gx[pupil+cell]);
 long long y=4*(direction*ylook+gy[point]*celly+gy[pupil+cell]);
 const double* plane=kernels+(long long)axis*grid*grid;
 double values[4];
 for(int j=0;j<4;j++) {
   double a=plane[iy[y]*grid+ix[x+j]]*wy[y];
   double b=plane[iy[y+1]*grid+ix[x+j]]*wy[y+1];
   double c=plane[iy[y+2]*grid+ix[x+j]]*wy[y+2];
   double d=plane[iy[y+3]*grid+ix[x+j]]*wy[y+3];
   values[j]=((a+b)+(c+d))*wx[x+j];
 }
 output[index]=(values[0]+values[1])+(values[2]+values[3]);
}
'''


class FusedCubicQuery:
    """Same float64 B-splines; no rectangular pupil/cell intermediates.

    Coordinates and stencil weights retain CPU double precision. Device arrays
    and the last stencil belong to one immutable optical sampling/owner.
    """
    def __init__(self, owner):
        self.owner=owner
        self.query=owner.query
        self.kernel=owner.cupy.RawKernel(_SOURCE, 'cubic_pairs', options=('--fmad=false',))
        xy=owner.sampling.xy; cells=owner.backend.cells; q=self.query
        self.gx=owner.upload(np.concatenate((np.searchsorted(q.x,xy[:,0]),
            np.searchsorted(q.cx,cells[:,0]))),integer=True)
        self.gy=owner.upload(np.concatenate((np.searchsorted(q.y,xy[:,1]),
            np.searchsorted(q.cy,cells[:,1]))),integer=True)
        self.pupil=len(xy); self.cells=len(cells)
        self._stencils=None

    def evaluate(self, kernels, origins, contraction, output):
        q=self.query; cached=self._stencils
        if cached is None or cached[1]!=contraction or not np.array_equal(cached[0],origins):
            x=(q.x[None,:,None]+origins[:,0,None,None]-contraction*q.cx[None,None])*q.scale
            y=(q.y[None,:,None]+origins[:,1,None,None]-contraction*q.cy[None,None])*q.scale
            ix,wx=self.owner.stencil(x); iy,wy=self.owner.stencil(y)
            cached=(origins.copy(),contraction,ix,wx,iy,wy)
            self._stencils=cached
        _,_,ix,wx,iy,wy=cached
        if not kernels.flags.c_contiguous or not output.flags.c_contiguous:
            raise ValueError('CUDA cubic kernels and output must be contiguous')
        dimensions=(self.pupil,self.cells,len(origins),q.grid,len(q.x)*len(q.cx),
                    len(q.y)*len(q.cy),len(q.cx),len(q.cy))
        self.kernel(((output.size+255)//256,), (256,),
            (kernels,ix,wx,iy,wy,self.gx,self.gy,output,
             *(np.int32(x) for x in dimensions),np.int64(output.size)))
