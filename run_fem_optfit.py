import json,numpy as np
from fem_check import solve
out=[]
for mu in [1.,1.05,1.1]:
 rows=[solve(a,mur=mu,h=.00005) for a in [67.5,68,68.5,69,69.5,70]]
 coef=np.polyfit([r['alpha']-68.5 for r in rows],[r['force'] for r in rows],3)
 roots=np.roots(np.polyder(coef));a=[float(x.real+68.5) for x in roots if abs(x.imag)<1e-8 and 67.5<x.real+68.5<70][0]
 fine=[solve(af,mur=mu,h=.000025) for af in [a-.3,a,a+.3]]
 c=np.polyfit([r['alpha']-a for r in fine],[r['force'] for r in fine],2);astar=float(a-c[1]/(2*c[0]))
 out.append(dict(mur=mu,alpha_coarse=a,alpha_fine=astar,force_fine=float(np.polyval(c,astar-a)),coarse=rows,fine=fine))
 json.dump(out,open('fem_optfit.json','w'),indent=2)
 print(mu,a,astar,flush=True)
