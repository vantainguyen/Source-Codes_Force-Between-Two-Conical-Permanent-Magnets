import json,numpy as np
from fem_check import solve
from numerics import cone,force
from scipy.optimize import minimize_scalar
out={'convergence':[],'gap_checks':[],'sensitivity':[],'domain_checks':[]}
def save():json.dump(out,open('fem_results.json','w'),indent=2)
for a in [30,68.877724]:
 for h in [.0002,.0001,.00005,.000025]:
  r=solve(a,h=h);r['semi_analytical']=force(cone(a),cone(a),.002)[0];out['convergence'].append(r);save();print('mesh',a,h,r['force'],flush=True)
for g in [.0005,.001,.005,.01,.02]:
 r=solve(68.877724,g=g,h=.00005);r['semi_analytical']=force(cone(68.877724),cone(68.877724),g)[0];out['gap_checks'].append(r);save();print('gap',g,r['force'],flush=True)
for a in [30,68.877724]:
 for mu in [1.,1.05,1.1]:
  for h in [.0001,.00005]:
   r=solve(a,mur=mu,h=h);out['sensitivity'].append(r);save();print('mu',a,mu,h,r['force'],flush=True)
for a in [30,68.877724]:
 for outer,growth in [(.16,1.12),(.12,1.06)]:
  r=solve(a,h=.00005,outer=outer,growth=growth);out['domain_checks'].append(r);save();print('domain',a,outer,growth,r['force'],flush=True)
print('DONE',flush=True)
