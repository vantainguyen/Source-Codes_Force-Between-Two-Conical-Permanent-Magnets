"""Reconstructed axisymmetric P1 FEM; SI units; identical repulsive cone pair.
Half-domain z>=0 uses mirror symmetry, upper tip z=g/2, upper remanence -ez.
Exact polygon overlap moments integrate material coefficients on cut triangles.
This is a NEW implementation, not the original submitted FEM code.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from time import perf_counter
MU0=4*np.pi*1e-7

def axis(core, outer, h, growth=1.12):
    a=list(np.linspace(0,core,int(np.ceil(core/h))+1)); step=h
    while a[-1]<outer:
        step*=growth; a.append(min(outer,a[-1]+step))
    return np.array(a)

def clip(poly,normal,offset):
    out=[]
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        fa=a@normal-offset; fb=b@normal-offset
        if fa>=0: out.append(a)
        if (fa>=0)!=(fb>=0): out.append(a+(b-a)*fa/(fa-fb))
    return np.array(out)

def r_moment(poly):
    if len(poly)<3:return 0.
    a=poly;b=np.roll(poly,-1,axis=0)
    return abs(np.sum((a[:,0]+b[:,0])*(a[:,0]*b[:,1]-b[:,0]*a[:,1]))/6)

def solve(alpha,g=.002,V=np.pi*(.01*np.tan(np.pi/6))**2*.01/3,mur=1.,h=.00015,outer=.12,growth=1.12):
    tic=perf_counter(); s=np.tan(np.deg2rad(alpha));L=(3*V/(np.pi*s*s))**(1/3);R=L*s; tip=g/2
    rr=axis(R+.004,outer,h,growth);zz=axis(tip+L+.004,outer,h,growth);nr=len(rr);nz=len(zz)
    rg,zg=np.meshgrid(rr,zz);xy=np.c_[rg.ravel(),zg.ravel()]
    ids=np.arange(nr*nz).reshape(nz,nr);a=ids[:-1,:-1].ravel();b=a+1;c=a+nr+1;d=a+nr
    tri=np.stack([np.c_[a,b,c],np.c_[a,c,d]],axis=1).reshape(-1,3);p=xy[tri]
    det=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1])
    grad=np.stack([np.stack([p[:,1,1]-p[:,2,1],p[:,2,0]-p[:,1,0]],1),np.stack([p[:,2,1]-p[:,0,1],p[:,0,0]-p[:,2,0]],1),np.stack([p[:,0,1]-p[:,1,1],p[:,1,0]-p[:,0,0]],1)],1)/det[:,None,None]
    integ=det/2*p[:,:,0].mean(1)
    normals=[np.array([0.,1.]),np.array([0.,-1.]),np.array([-1.,s])];offsets=[tip,-(tip+L),s*tip]
    ds=np.stack([p@n-o for n,o in zip(normals,offsets)],axis=2)
    full=np.all(ds>=-1e-15,axis=(1,2));outside=np.any(np.all(ds<0,axis=1),axis=1)
    mat=np.zeros(len(tri));mat[full]=integ[full]
    for it in np.flatnonzero(~full&~outside):
        poly=p[it]
        for norm,off in zip(normals,offsets):
            if len(poly):poly=clip(poly,norm,off)
        mat[it]=r_moment(poly)
    weight=integ+(mur-1)*mat
    ke=np.einsum('nik,njk,n->nij',grad,grad,weight)
    rows=np.repeat(tri,3,axis=1).ravel();cols=np.tile(tri,(1,3)).ravel()
    A=coo_matrix((ke.ravel(),(rows,cols)),shape=(len(xy),len(xy))).tocsr()
    rhs=np.bincount(tri.ravel(),weights=(-grad[:,:,1]*mat[:,None]).ravel(),minlength=len(xy))
    free=(xy[:,0]<outer-1e-13)&(xy[:,1]<outer-1e-13);u=np.zeros(len(xy));u[free]=spsolve(A[free][:,free],rhs[free])
    fields=-np.einsum('ni,nik->nk',u[tri],grad)
    def at(r,z):
        ix=np.clip(np.searchsorted(rr,r,side='right')-1,0,nr-2);iz=np.clip(np.searchsorted(zz,z,side='right')-1,0,nz-2)
        fx=(r-rr[ix])/(rr[ix+1]-rr[ix]);fz=(z-zz[iz])/(zz[iz+1]-zz[iz]);tid=2*(iz*(nr-1)+ix)+(fz>fx)
        return fields[tid]
    def horiz(z,rmax,nzsign):
        iz=np.clip(np.searchsorted(zz,z,side='right')-1,0,nz-2);fz=(z-zz[iz])/(zz[iz+1]-zz[iz])
        cuts=np.unique(np.r_[0,rmax,rr[(rr>0)&(rr<rmax)],(rr[:-1]+fz*np.diff(rr))[(rr[:-1]+fz*np.diff(rr)>0)&(rr[:-1]+fz*np.diff(rr)<rmax)]])
        lo=cuts[:-1];hi=cuts[1:];B=at((lo+hi)/2,np.full(len(lo),z))
        return np.sum(np.pi*(hi*hi-lo*lo)*nzsign*(B[:,1]**2-B[:,0]**2)/(2*MU0))
    def vert(r,zlo,zhi):
        ix=np.clip(np.searchsorted(rr,r,side='right')-1,0,nr-2);fx=(r-rr[ix])/(rr[ix+1]-rr[ix]);diag=zz[:-1]+fx*np.diff(zz)
        cuts=np.unique(np.r_[zlo,zhi,zz[(zz>zlo)&(zz<zhi)],diag[(diag>zlo)&(diag<zhi)]])
        lo=cuts[:-1];hi=cuts[1:];B=at(np.full(len(lo),r),(lo+hi)/2)
        return np.sum(2*np.pi*r*(hi-lo)*B[:,0]*B[:,1]/MU0)
    plane_force=horiz(0,outer,-1)+horiz(outer,outer,1)
    forces=[]
    for off in [.002,.003,.004]:
        rc=R+off;zt=tip+L+off
        forces.append(horiz(0,rc,-1)+horiz(zt,rc,1)+vert(rc,0,zt))
    # Internal opposing H component along the -ez remanence is grad(u)_z / mu0.
    # Report volume-weighted mean and percentile, not singular-apex maxima.
    hind=-fields[full,1]/MU0;weights=integ[full]
    mean_h=np.average(hind,weights=weights)
    order=np.argsort(hind);cdf=np.cumsum(weights[order])/weights.sum();h01=np.interp(.01,cdf,hind[order])
    return dict(alpha=alpha,g=g,mur=mur,h=h,outer=outer,growth=growth,L=L,R=R,nodes=len(xy),elements=len(tri),force=float(plane_force),contour_forces=[float(f) for f in forces],Hparallel_mean=float(mean_h),Hparallel_p01=float(h01),seconds=perf_counter()-tic)

if __name__=='__main__':
    import json,sys
    print(json.dumps(solve(float(sys.argv[1]) if len(sys.argv)>1 else 30,h=float(sys.argv[2]) if len(sys.argv)>2 else .0002,mur=float(sys.argv[3]) if len(sys.argv)>3 else 1.)),flush=True)
