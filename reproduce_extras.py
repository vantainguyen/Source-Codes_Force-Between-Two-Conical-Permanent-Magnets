import json,time,numpy as np
from fem_check import solve
from numerics import cone,force
rows=[solve(30,g=g,h=.00005) for g in [.0005,.001,.003,.005,.01,.02]]
json.dump(rows,open('fem_reference_gaps.json','w'),indent=2)
ref=cone(30);f,c=force(ref,ref,.002,160);u,v=force(ref,ref,.002,100,raw=True,npsi=512)
json.dump({'reference_unreduced':dict(reduced=f,unreduced=u,reduced_components=c,unreduced_components=v)},open('additional_checks.json','w'),indent=2)
force(cone(68.877724),cone(68.877724),.002)
a=[]
for _ in range(100):
 tic=time.perf_counter();force(cone(68.877724),cone(68.877724),.002);a.append(time.perf_counter()-tic)
b=[solve(68.877724,h=.00005)['seconds'] for _ in range(5)]
json.dump({'semi_median_s':float(np.median(a)),'semi_range_s':[min(a),max(a)],'fem_median_s':float(np.median(b)),'fem_range_s':[min(b),max(b)],'repeats_semi':100,'repeats_fem':5,'hardware':'Timing values depend on host hardware and load'},open('timing.json','w'),indent=2)
