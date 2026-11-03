"""Optional Metal cubic queries at valid pupil/lenslet pairs only."""
import numpy as np


_CUBIC_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline float vertical(device const float* plane, uint grid, uint column,
                      uint4 rows, float4 weights) {
    float a=plane[rows.x*grid+column]*weights.x;
    float b=plane[rows.y*grid+column]*weights.y;
    float c=plane[rows.z*grid+column]*weights.z;
    float d=plane[rows.w*grid+column]*weights.w;
    return (a+b)+(c+d);
}

kernel void cubic_pairs(
    device const float* kernels [[buffer(0)]],
    device const long* ix [[buffer(1)]], device const float* wx [[buffer(2)]],
    device const long* iy [[buffer(3)]], device const float* wy [[buffer(4)]],
    device const long* gx [[buffer(5)]], device const long* gy [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& pupil [[buffer(8)]], constant uint& cells [[buffer(9)]],
    constant uint& batch [[buffer(10)]], constant uint& grid [[buffer(11)]],
    constant uint& xlook [[buffer(12)]], constant uint& ylook [[buffer(13)]],
    constant uint& cellx [[buffer(14)]], constant uint& celly [[buffer(15)]],
    constant uint& total [[buffer(16)]],
    uint index [[thread_position_in_grid]]) {
    if(index>=total)return;
    uint cell=index%cells;
    uint point=(index/cells)%pupil;
    uint direction=(index/(cells*pupil))%batch;
    uint axis=index/(cells*pupil*batch);
    uint x=4*(direction*xlook+uint(gx[point])*cellx+uint(gx[pupil+cell]));
    uint y=4*(direction*ylook+uint(gy[point])*celly+uint(gy[pupil+cell]));
    uint4 rows=uint4(iy[y],iy[y+1],iy[y+2],iy[y+3]);
    float4 weights=float4(wy[y],wy[y+1],wy[y+2],wy[y+3]);
    device const float* plane=kernels+axis*grid*grid;
    float a=vertical(plane,grid,uint(ix[x]),rows,weights)*wx[x];
    float b=vertical(plane,grid,uint(ix[x+1]),rows,weights)*wx[x+1];
    float c=vertical(plane,grid,uint(ix[x+2]),rows,weights)*wx[x+2];
    float d=vertical(plane,grid,uint(ix[x+3]),rows,weights)*wx[x+3];
    output[index]=(a+b)+(c+d);
}
"""


class FusedCubicQuery:
    """Evaluate unchanged periodic B-splines without rectangular intermediates.

    The owning MPS backend supplies float64 query coordinates and the existing
    float32 stencil weights. This object keeps only geometry indices, the last
    direction-dependent stencils and a compiled kernel. It writes every valid
    pupil/lenslet pair for each slope axis; no physical input is omitted.
    """
    def __init__(self,owner):
        self.owner=owner
        self.query=owner.query
        self.library=owner.torch.mps.compile_shader(_CUBIC_SOURCE)
        xy=owner.sampling.xy
        cells=owner.backend.cells
        q=self.query
        self.gx=owner.upload(np.concatenate((np.searchsorted(q.x,xy[:,0]),
                                            np.searchsorted(q.cx,cells[:,0]))),integer=True)
        self.gy=owner.upload(np.concatenate((np.searchsorted(q.y,xy[:,1]),
                                            np.searchsorted(q.cy,cells[:,1]))),integer=True)
        self.pupil=len(xy)
        self.cells=len(cells)
        self._stencils=None

    def evaluate(self,kernels,origins,contraction,output):
        """Fill contiguous ``(axis, direction, pupil, cell)`` device output."""
        q=self.query
        cached=self._stencils
        if cached is None or cached[1]!=contraction or not np.array_equal(cached[0],origins):
            x=(q.x[None,:,None]+origins[:,0,None,None]-contraction*q.cx[None,None])*q.scale
            y=(q.y[None,:,None]+origins[:,1,None,None]-contraction*q.cy[None,None])*q.scale
            ix,wx=self.owner.stencil(x);iy,wy=self.owner.stencil(y)
            cached=(origins.copy(),contraction,ix,wx,iy,wy)
            self._stencils=cached
        _,_,ix,wx,iy,wy=cached
        if not kernels.is_contiguous() or not output.is_contiguous():
            raise ValueError('Fused cubic kernels and output must be contiguous')
        self.library.cubic_pairs(kernels,ix,wx,iy,wy,self.gx,self.gy,output,
            self.pupil,self.cells,len(origins),q.grid,len(q.x)*len(q.cx),
            len(q.y)*len(q.cy),len(q.cx),len(q.cy),output.numel(),
            threads=output.numel(),group_size=min(256,output.numel()))
