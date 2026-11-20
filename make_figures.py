import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from fem_check import axis
r=json.load(open('results.json'));f=json.load(open('fem_results.json'))
Path('figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'savefig.dpi':200})
def save(n):
 plt.tight_layout();plt.savefig(f'figures/figure_{n}.png',bbox_inches='tight');plt.close()
ff=r['farfield'];D=np.array([x['D'] for x in ff]);F=np.array([x['force'] for x in ff]);dip=np.array([x['dipole'] for x in ff]);err=np.abs(F/dip-1)
fig,ax=plt.subplots(1,2,figsize=(7,2.8));ax[0].loglog(D,F,label='Cone model');ax[0].loglog(D,dip,'--',label='Centroid dipoles');ax[0].set(xlabel='Centroid separation D (m)',ylabel='Force (N)');ax[0].legend(fontsize=8)
ax[1].loglog(D,err,label='|F / Fdip − 1|');ax[1].loglog(D,2.5e-5/D**2,'--',label='2.5 × 10⁻⁵ m² / D²');ax[1].set(xlabel='Centroid separation D (m)',ylabel='Relative dipole error');ax[1].legend(fontsize=8);save(4)
fig,ax=plt.subplots(figsize=(6,3));aa=r['angle_curves']
for g in [.0005,.002,.005,.01]:ax.plot([x['alpha'] for x in aa],[x[str(g)] for x in aa],label=f'g = {g*1000:g} mm')
ax.set(xlabel='Cone half-angle α (degrees)',ylabel='Force (N)',xlim=(1,89));ax.legend(fontsize=8);save(5)
fig,ax=plt.subplots(1,2,figsize=(7,2.8));di=r['dimensionless'];x=[z['ghat'] for z in di]
ax[0].semilogx(x,[z['alpha'] for z in di]);ax[0].set(xlabel='Dimensionless gap ĝ = g / V¹ᐟ³',ylabel='Optimal half-angle (degrees)')
ax[1].loglog(x,[z['psi'] for z in di]);ax[1].set(xlabel='Dimensionless gap ĝ = g / V¹ᐟ³',ylabel='Maximum dimensionless force Ψmax');save(6)
fig,ax=plt.subplots(figsize=(5.8,3));a=np.deg2rad(68.877724);L=.00367779;R=.0095202;tip=.001;rr=axis(R+.004,.12,.0004);zz=axis(tip+L+.004,.12,.0004)
for x in rr[rr<.015]:ax.plot([x*1000]*2,[0,9],color='.80',lw=.4)
for z in zz[zz<.009]:ax.plot([0,15],[z*1000]*2,color='.80',lw=.4)
for i in range(len(rr)-1):
 for j in range(len(zz)-1):
  if rr[i]<.015 and zz[j]<.009:ax.plot(rr[i:i+2]*1000,zz[j:j+2]*1000,color='.86',lw=.3)
ax.fill([0,R*1000,0],[tip*1000,(tip+L)*1000,(tip+L)*1000],color='#a7bbcb',alpha=.5)
ax.plot([0,R*1000,0,0],[tip*1000,(tip+L)*1000,(tip+L)*1000,tip*1000],color='#1f4e79',lw=1.5)
ax.set(xlabel='r (mm)',ylabel='z from gap midplane (mm)',xlim=(0,15),ylim=(0,9));ax.set_aspect('equal');save(7)
fig,ax=plt.subplots(figsize=(6,3))
for a,label in [(30,'30° reference'),(68.877724,'68.878° cone')]:
 rows=[z for z in f['convergence'] if z['alpha']==a];ax.semilogx([z['h']*1000 for z in rows],[100*(z['force']/z['semi_analytical']-1) for z in rows],'o-',label=label)
ax.axhline(0,color='.5',lw=.6);ax.set(xlabel='Core mesh spacing h (mm)',ylabel='FEM difference (%)');ax.legend(fontsize=8);save(8)
ref=json.load(open('fem_reference_gaps.json'));ref+=[z for z in f['convergence'] if z['alpha']==30 and z['h']==.00005]
op=f['gap_checks']+[z for z in f['convergence'] if z['alpha']>60 and z['h']==.00005]
fig,ax=plt.subplots(figsize=(6,3));cu=r['curves']
for key,rows,label in [('ref',ref,'30°'),('opt',op,'68.878°')]:
 ax.plot([z['g']*1000 for z in cu],[z[key] for z in cu],label=f'{label} quadrature');ax.scatter([z['g']*1000 for z in rows],[z['force'] for z in rows],s=18,marker='x',label=f'{label} FEM')
ax.set(xlabel='Minimum gap (mm)',ylabel='Force (N)',xlim=(.5,20));ax.legend(fontsize=8,ncol=2);save(9)
from numerics import force,cone
fig,ax=plt.subplots(figsize=(6,3))
for rows,a,label in [(ref,30,'30°'),(op,68.877724,'68.878°')]:
 rows=sorted(rows,key=lambda z:z['g']);ax.plot([z['g']*1000 for z in rows],[100*(z['force']/force(cone(a),cone(a),z['g'])[0]-1) for z in rows],'o-',label=label)
ax.axhline(0,color='.5',lw=.6);ax.set(xlabel='Minimum gap (mm)',ylabel='FEM difference (%)');ax.legend(fontsize=8);save(10)
fig,ax=plt.subplots(1,2,figsize=(7,2.8))
for key,label in [('ref','30° cones'),('opt','68.878° cones'),('cyl','Cylinders R = L'),('frustum','Frustums η = 0.10')]:
 ax[0].plot([z['g']*1000 for z in cu],[z[key] for z in cu],label=label);ax[1].plot([z['g']*1000 for z in cu],[z[key+'_k'] for z in cu],label=label)
ax[0].set(xlabel='Minimum gap (mm)',ylabel='Force (N)',xlim=(.5,10),ylim=(0,14));ax[1].set(xlabel='Minimum gap (mm)',ylabel='Axial stiffness (N/m)',xlim=(.5,10),ylim=(0,6500));ax[0].legend(fontsize=7);save(11)
