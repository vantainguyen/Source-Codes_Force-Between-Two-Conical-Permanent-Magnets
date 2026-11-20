"""Reproducible revision calculations; SI units throughout."""
import numpy as np
from scipy.special import ellipe,ellipk,j1
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from numpy.polynomial.legendre import leggauss
from functools import lru_cache
import json,time,platform,sys
MU0=4*np.pi*1e-7
V0=np.pi*(.01*np.tan(np.pi/6))**2*.01/3
@lru_cache(None)
def rule(n):
    x,w=leggauss(n);return (x+1)/2,w/2
def cone(alpha,V=V0,L=None,J=1,eta=0):
    a=np.deg2rad(alpha);s=np.tan(a)
    if L is None:L=(3*V*(1-eta)**2/(np.pi*s*s*(1+eta+eta*eta)))**(1/3)
    R=L*s/(1-eta);b=eta*R
    return dict(kind='cone' if eta==0 else 'frustum',alpha=alpha,L=L,R=R,b=b,J=J,V=np.pi*L*(R*R+R*b+b*b)/3)
def cylinder(V=V0,aspect=1,R=None,L=None,J=1):
    if R is None:R=(V/(np.pi*aspect))**(1/3)
    if L is None:L=aspect*R
    return dict(kind='cylinder',L=L,R=R,b=R,J=J,V=np.pi*R*R*L)
def surfaces(o,n):
    x,w=rule(n);L=o['L'];R=o['R'];b=o['b'];J=o['J']
    out=[('B',R*x,np.full(n,L),-J*R*R*x*w)]
    if R>b:out.append(('C',b+(R-b)*x,L*x,J*(R-b)*(b+(R-b)*x)*w))
    if b>0:out.append(('T',b*x,np.zeros(n),J*b*b*x*w))
    return out
def kernel(r,s,d,deriv=False):
    A=d*d+(r+s)**2;B=d*d+(r-s)**2;m=4*r*s/A;E=ellipe(m)
    if deriv:return 4/(B*np.sqrt(A))*(E*(1-2*d*d/B-d*d/A)+d*d*(ellipk(m)-E)/A)
    return 4*d*E/(B*np.sqrt(A))
def force(o1,o2,g,n=160,deriv=False,raw=False,npsi=256):
    terms={}
    for name,r,z,q in surfaces(o1,n):
        for name2,s,t,p in surfaces(o2,n):
            d=g+z[:,None]+t[None,:]
            if raw:
                x,w=rule(npsi);kk=np.zeros_like(d)
                for xx,ww in zip(x,w):kk+=2*np.pi*ww*d/(d*d+r[:,None]**2+s[None,:]**2-2*r[:,None]*s[None,:]*np.cos(2*np.pi*xx))**1.5
            else:kk=kernel(r[:,None],s[None,:],d,deriv)
            terms[name+name2]=float(np.sum(q[:,None]*p[None,:]*kk)/(2*MU0))
    return float(sum(terms.values())),terms
def optimum(ghat,n=120):
    g=ghat*V0**(1/3)
    opt=minimize_scalar(lambda a:-force(cone(a),cone(a),g,n)[0],bounds=(1,89.95),method='bounded',options={'xatol':1e-8})
    f,c=force(cone(opt.x),cone(opt.x),g,160)
    return dict(ghat=ghat,g=g,alpha=float(opt.x),force=f,psi=MU0*f/V0**(2/3),components=c)
def vokoun(R,L,g,J=1):
    # Eq. (4) written with decaying exponentials to avoid overflow; repulsive magnitude.
    val,err=quad(lambda q: j1(q)**2/q*np.exp(-q*g/R)*(-np.expm1(-q*L/R))**2 if q>0 else 0,0,np.inf,epsabs=2e-12,epsrel=2e-11,limit=1000)
    return np.pi*J*J*R*R/MU0*val
def run():
    out={}; ref=cone(30);opt=optimum(.002/V0**(1/3));best=cone(opt['alpha']);out['optimum']=opt
    out['convergence']=[]
    for a in [30,opt['alpha']]:
        for g in [.0001,.0005,.002]:
            for n in [20,40,80,160,320]:
                f,c=force(cone(a),cone(a),g,n);out['convergence'].append(dict(alpha=a,g=g,n=n,force=f,**c))
    out['unequal']=[]
    for o1,o2,g in [(cone(25,L=.008,J=.9),cone(55,L=.005,J=1.1),.002),(cone(40,L=.006,J=1),cone(70,L=.003,J=.8),.001)]:
        f,c=force(o1,o2,g,160);u,v=force(o1,o2,g,100,raw=True,npsi=512)
        out['unequal'].append(dict(o1=o1,o2=o2,g=g,force=f,raw=u,terms=c,raw_terms=v))
    out['baselines']=[]
    for label,o1,o2 in [('reference cones',ref,ref),('optimal cones',best,best),('cylinders R=L',cylinder(),cylinder()),('cone-cylinder',best,cylinder())]+[(f'frustums eta={eta}',cone(opt['alpha'],eta=eta),cone(opt['alpha'],eta=eta)) for eta in [.05,.1,.25]]:
        f,c=force(o1,o2,.002);k=-force(o1,o2,.002,deriv=True)[0]
        fd=-(force(o1,o2,.002+1e-6)[0]-force(o1,o2,.002-1e-6)[0])/2e-6
        out['baselines'].append(dict(label=label,o1=o1,o2=o2,force=f,stiffness=k,fd_stiffness=fd,components=c))
    out['cylinder_benchmark']=[]
    for g in [.0005,.002,.01]:
        o=cylinder();f=force(o,o,g)[0];known=vokoun(o['R'],o['L'],g)
        out['cylinder_benchmark'].append(dict(g=g,ring=f,vokoun=known))
    out['published_cylinder_geometry']=[]
    for L in [.003,.006,.009]:
        o=cylinder(R=.0015,L=L,J=MU0*.859e6)
        for g in [.0005,.001,.003]:
            out['published_cylinder_geometry'].append(dict(R=o['R'],L=L,J=o['J'],g=g,ring=force(o,o,g)[0],vokoun=vokoun(o['R'],L,g,o['J'])))
    out['dimensionless']=[optimum(float(x)) for x in np.geomspace(.01,30,45)]
    out['dimensionless_table']=[optimum(x) for x in [.02,.05,.1,.2,.5,1,2,5,10,20]]
    out['farfield']=[]
    for g in np.geomspace(.03,1,80):
        D=g+1.5*ref['L'];f=force(ref,ref,g,160)[0];dip=3*V0*V0/(2*np.pi*MU0*D**4)
        out['farfield'].append(dict(g=g,D=D,force=f,dipole=dip,ratio=f/dip,error=abs(f/dip-1)))
    out['curves']=[]
    for g in np.geomspace(.0001,.03,80):
        row={'g':float(g)}
        for name,o in [('ref',ref),('opt',best),('cyl',cylinder()),('frustum',cone(opt['alpha'],eta=.1))]:
            row[name]=force(o,o,g)[0];row[name+'_k']=-force(o,o,g,deriv=True)[0]
        out['curves'].append(row)
    out['angle_curves']=[]
    for a in np.linspace(1,89,177):
        out['angle_curves'].append(dict(alpha=float(a),**{str(g):force(cone(a),cone(a),g,n=160)[0] for g in [.0005,.002,.005,.01]}))
    times=[]
    for _ in range(101):
        tic=time.perf_counter();force(best,best,.002,n=160);times.append(time.perf_counter()-tic)
    out['timing']=dict(n=160,median_seconds=float(np.median(times[1:])),min_seconds=min(times[1:]),max_seconds=max(times[1:]),repeats=100)
    import scipy,matplotlib
    out['environment']=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,platform=platform.platform(),processor=platform.processor())
    with open('results.json','w') as f:json.dump(out,f,indent=2)
    print(json.dumps({k:out[k] for k in ['optimum','unequal','baselines','cylinder_benchmark','timing']},indent=2),flush=True)
if __name__=='__main__':run()
