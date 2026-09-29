import math,json,copy
from pathlib import Path
import closed_support as c
import qualify as f
rows=[]
def rejected(name,mutate):
 m=f.retained();mutate(m);r=c.solve(m)
 assert r['status']=='invalid' and not r['accepted'] and r['outer_nodes']==0
 assert 'global_nonpenetration_exact' not in r and 'normal_closing_exact' not in r
 rows.append({'case':name,'result':r})
def translate(m,offset):
 for poses in (m.start,m.end):
  for owner in poses:owner[2]+=offset
rejected('unrepresented_owner_translation_4096',lambda m:translate(m,4096))
rejected('unrepresented_owner_translation_1',lambda m:translate(m,1))
rejected('unrepresented_owner_double_ULP',lambda m:m.start[0].__setitem__(0,math.nextafter(m.start[0][0],math.inf)))
rejected('unrepresented_radius_double_ULP',lambda m:setattr(m,'radius',math.nextafter(m.radius,math.inf)))
rejected('bad_owner_count',lambda m:m.start.pop())
rejected('bad_component_count',lambda m:m.end[2].append(0))
rejected('nonfinite_radius',lambda m:setattr(m,'radius',math.nan))
rejected('nonfinite_owner',lambda m:m.end[4].__setitem__(1,math.inf))
r=c.solve({});assert r['status']=='invalid' and not r['accepted'];rows.append({'case':'wrong_motion_type','result':r})
m=f.retained()
for poses in (m.start,m.end):
 for i in range(2):poses[i][0]=c.f32(poses[i][0]+.0625)
r=c.solve(m);assert r['status']=='grazing' and r['accepted'];rows.append({'case':'valid_represented_owner_mutation','result':r})
m=f.retained();m.radius=f.nextf(m.radius,False);r=c.solve(m);assert r['status']=='clear' and r['accepted'];rows.append({'case':'valid_represented_radius_mutation','result':r})
for radius in (.004,5e-8,1e-15,2**-149):
 m=f.retained();m.radius=c.f32(radius)
 for poses in (m.start,m.end):
  for i in range(2):poses[i][2]=m.radius
 r=c.solve(m);assert r['status']=='grazing' and r['accepted'] and r['root_exact']=='1/4' and r['outer_nodes']<=128
 assert m.gap(.25)==0
 rows.append({'case':'exact_closed_plane_radius_'+str(m.radius),'result':r})
result={'source_binding_controls':len(rows),'failures':0,'cases':rows,'no_gpu':True}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
