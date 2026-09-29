import hashlib,json,math
from pathlib import Path
root=Path('/Users/home/numi-solver');out=Path(__file__).resolve().parent
source=root/'build/loaded-drop-investigation-corrected-launch/cloth_bag.cpp'
manifest=root/'build/loaded-drop-investigation-corrected-launch/manifest.json'
frozen=json.loads(manifest.read_text())
for name,h in frozen['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h

def idx(r,c):
    assert 1<=r<=11 and 1<=c<=11
    return 1344+(r-1)*11+c-1

def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def angle(a,b):return math.atan2(math.hypot(*cross(a,b)),sum(x*y for x,y in zip(a,b)))
rows=[];digests={};triangle=[1433,1444,1445]
for frame in (0,620):
    path=root/f'build/loaded-drop-investigation-corrected-96-{frame}.obj';data=path.read_bytes();digests[str(path)]=hashlib.sha256(data).hexdigest()
    p=[tuple(map(float,l.split()[1:])) for l in data.decode().splitlines() if l.startswith('v ')]
    assert len(p)==1465 and all(math.isfinite(v) for point in p for v in point)
    r,c=10,2;center=p[idx(r,c)];n,s,w,e=[p[idx(a,b)] for a,b in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]]
    a,b,d=[p[i] for i in triangle]
    rows.append({'frame':frame,'center_owner':idx(r,c),'centered_knot_owners':[idx(r-1,c),idx(r+1,c),idx(r,c-1),idx(r,c+1)],
        'centered_knot_angle_rad':angle(sub(s,n),sub(e,w)),
        'outgoing_arm_angles_rad':{name:angle(sub(aa,center),sub(bb,center)) for name,aa,bb in [('north_east',n,e),('north_west',n,w),('south_east',s,e),('south_west',s,w)]},
        'triangle':2813,'triangle_owners':triangle,'triangle_area_fp64_m2':.5*math.hypot(*cross(sub(b,a),sub(d,a))),
        'triangle_edge_lengths_m':[math.dist(a,b),math.dist(b,d),math.dist(d,a)]})
assert rows[1]['triangle_area_fp64_m2']<1e-8
result={'schema':'numi.loaded-drop.retained-local-cell-shape-diagnostic.v1',
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'frozen_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'all_seven_frozen_hashes_verified':True,'snapshot_sha256':digests,'rows':rows,
    'centered_knot_angle_change_rad':abs(rows[1]['centered_knot_angle_rad']-rows[0]['centered_knot_angle_rad']),
    'outgoing_north_east_angle_change_rad':abs(rows[1]['outgoing_arm_angles_rad']['north_east']-rows[0]['outgoing_arm_angles_rad']['north_east']),
    'retained_triangle_area_ratio':rows[1]['triangle_area_fp64_m2']/rows[0]['triangle_area_fp64_m2'],
    'diagnostic_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'boundary':'Read-only two-state geometric diagnostic. Frame0 is makeCloth rest geometry before simulation, verified in frozen source. The production centered knot measure uses opposite-neighbor axes, not all outgoing-arm angles or individual face area. A stable centered angle therefore does not certify local triangle noncollapse. No unique causal solver operator, material calibration, new energy/force model or successful repair is established.'}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
