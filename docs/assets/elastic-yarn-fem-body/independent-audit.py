#!/usr/bin/env python3
"""Read-only full pack audit; exact-rational geometry + independent pivoted LU."""
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal as D,localcontext
from collections import defaultdict,Counter
import argparse,csv,hashlib,itertools,json,math,struct,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PACK=ROOT/'build/elastic-yarn-fem-step-preparation'
N=2057;E=10240
PERMS=[(p,(-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))) for p in itertools.permutations(range(3))]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def bits(x):return struct.pack('<f',float(x))
def load(name,fmt,count):
 data=(PACK/name).read_bytes();assert len(data)==struct.calcsize(fmt)*count,(name,len(data),'bad byte count')
 return list(struct.iter_unpack(fmt,data))
def det_exact(a):return sum((s*math.prod(a[i][p[i]] for i in range(3)) for p,s in PERMS),Q(0))
def det_lu(a):
 m=[list(row) for row in a];answer=1.
 for k in range(3):
  pivot=max(range(k,3),key=lambda i:abs(m[i][k]));assert m[pivot][k]!=0
  if pivot!=k:m[pivot],m[k]=m[k],m[pivot];answer=-answer
  q=m[k][k];answer*=q
  for i in range(k+1,3):
   factor=m[i][k]/q
   for j in range(k+1,3):m[i][j]-=factor*m[k][j]
 return answer

def edge(v,ids):return [[v[ids[j+1]][k]-v[ids[0]][k] for j in range(3)] for k in range(3)]
def mul(a,b):return [[math.fsum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def inverse_residual(a,b):return max(abs(sum(a[i][k]*b[k][j] for k in range(3))-int(i==j)) for i in range(3) for j in range(3))
def parity(v):return (-1)**sum(v[i]>v[j] for i in range(len(v)) for j in range(i+1,len(v)))
def q32(value):return Q.from_float(float(value))
def recover(token):
 value=f32(token);assert math.isfinite(value)
 if D(token)==0:
  # defaultfloat significant precision prints no nonzero FP32 as decimal0.
  assert float(token)==0;return value
 integer=struct.unpack('<I',bits(value))[0]
 previous=struct.unpack('<f',struct.pack('<I',integer-1 if value>0 else integer+1))[0]
 following=struct.unpack('<f',struct.pack('<I',integer+1 if value>0 else integer-1))[0]
 cell=D(5)*(D(10)**(D(token).adjusted()-12))
 lower=(D.from_float(previous)+D.from_float(value))/2;upper=(D.from_float(value)+D.from_float(following))/2
 assert lower<D(token)-cell and D(token)+cell<upper,(token,'not uniquely represented')
 return value

def negative(mode):
 raw=load('elements.abi1.bin','<4I16f4I',E);rest=load('rest-positions.f32x4.bin','<4f',N)
 if mode=='inverse-shear':
  e=raw[0];ids=e[:4];inv=[list(e[4+4*i:7+4*i]) for i in range(3)];dm=edge(rest,ids);changed=[r[:] for r in inv]
  changed[0]=[x+.01*y for x,y in zip(changed[0],changed[1])]
  determinant_only=abs(6*e[7]*det_lu(changed)-1)<=2e-4
  residual=inverse_residual(dm,changed);proper_reject=residual>.001
  print(json.dumps({'negative':mode,'determinant_only_rest_gate_would_admit':determinant_only,'actual_matrix_inverse_residual':residual,'independent_geometry_rejects':proper_reject}));return 1 if determinant_only and proper_reject else 2
 if mode=='mass':
  vm=load('velocity-mass-frame38.f32x4.bin','<4f',N);old=vm[5][3];u=struct.unpack('<I',bits(old))[0];new=struct.unpack('<f',struct.pack('<I',u+1))[0]
  false_admit=math.isfinite(new) and new>0;rejected=bits(old)!=bits(new)
  print(json.dumps({'negative':mode,'positive_mass_only_would_admit':false_admit,'captured_source_bits_changed':rejected,'old_mass_kg':old,'new_mass_kg':new}));return 1 if false_admit and rejected else 2
 if mode=='truncate':
  actual=len((PACK/'elements.abi1.bin').read_bytes())-1;rejected=actual!=96*E
  print(json.dumps({'negative':mode,'truncated_bytes':actual,'exact_ABI_size_check_rejects':rejected}));return 1 if rejected else 2
 raise AssertionError('unknown negative')

def main():
 begin=time.monotonic();producer=json.loads((PACK/'evidence.json').read_text());receipt=json.loads((PACK/'receipt.json').read_text());source=json.loads((PACK/'source-manifest.json').read_text());body=json.loads((PACK/'body.json').read_text())
 receipt_checks={p:sha(ROOT/p)==s for p,s in receipt['sha256'].items()};input_checks={p:sha(p)==s for p,s in producer['input_sha256'].items()};output_checks={p:sha(p)==s for p,s in producer['outputs_sha256'].items()};assert all(receipt_checks.values()) and all(input_checks.values()) and all(output_checks.values())
 git_bindings={}
 for name,s in source['source_blob_sha256'].items():
  r=subprocess.run(['git','show',source['physics_source_commit']+':'+name],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5);assert r.returncode==0
  git_bindings[name]={'git_blob_sha256':hashlib.sha256(r.stdout).hexdigest(),'frozen_sha256':sha(PACK/'frozen'/name),'recorded_sha256':s};assert len(set(git_bindings[name].values()))==1
 current=load('positions-frame38.f32x4.bin','<4f',N);vm=load('velocity-mass-frame38.f32x4.bin','<4f',N);rest=load('rest-positions.f32x4.bin','<4f',N);elements=load('elements.abi1.bin','<4I16f4I',E);faces=load('boundary-faces.u32x3.bin','<3I',1280);offsets=[v[0] for v in load('incidence-offsets.u32.bin','<I',N+1)];incidence=load('incidence.u32x4.bin','<4I',4*E)
 assert all(math.isfinite(x) for row in current+vm+rest for x in row);assert all(bits(row[3])==bits(0) for row in current+rest);assert min(row[3] for row in vm)>0
 assert len(body['nodes'])==N and len(body['elements'])==E
 for i,node in enumerate(body['nodes']):
  assert struct.pack('<4f',*node['position_f32'],0)==struct.pack('<4f',*current[i]);assert struct.pack('<4f',*node['velocity_f32'],node['lumped_mass_kg_f32'])==struct.pack('<4f',*vm[i]);assert struct.pack('<4f',*node['rest_position_f32'],0)==struct.pack('<4f',*rest[i])
 for i,e in enumerate(body['elements']):
  values=[*e['nodes']]
  for row,values3 in enumerate(e['inverse_rest_rows_f32']):values.extend([*values3,e['rest_volume_m3_f32'] if row==0 else 0])
  values.extend(e['material_f32']);values.extend(e['control_u32']);assert struct.pack('<4I16f4I',*values)==struct.pack('<4I16f4I',*elements[i])
 assert body['boundary_faces']==[list(f) for f in faces] and body['node_incidence_offsets']==offsets and body['node_incidence_entries']==[list(e) for e in incidence]
 selected={0:[],38:[]};serialized={0:[],38:[]};fieldcount=0;framecounts=Counter()
 with localcontext() as ctx:
  ctx.prec=80
  with (ROOT/'docs/assets/deformable-support-10240-trajectory.csv').open() as stream:
   for row in csv.DictReader(stream):
    frame=int(row['frame']);assert int(row['node'])==framecounts[frame];framecounts[frame]+=1
    if frame not in selected:continue
    assert float(row['time_s'])==frame*.005
    values=[row[k] for k in ('x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','mass_kg')];selected[frame].append([recover(x) for x in values]);serialized[frame].append([float(x) for x in values]);fieldcount+=7
 assert len(framecounts)==101 and set(framecounts.values())=={N}
 for i in range(N):
  assert all(bits(a)==bits(b) for a,b in zip(selected[38][i],list(current[i][:3])+list(vm[i])))
  assert all(bits(a)==bits(b) for a,b in zip(selected[0][i][:3],rest[i][:3]));assert selected[0][i][3:6]==[0,0,0] and bits(selected[0][i][6])==bits(vm[i][3])
 topology=list(csv.DictReader((ROOT/'docs/assets/deformable-support-10240-topology.csv').open()));assert len(topology)==E
 qrest=[tuple(q32(x) for x in p[:3]) for p in rest];qcurrent=[tuple(q32(x) for x in p[:3]) for p in current];qcache={};qt=lambda x:qcache.setdefault(x,q32(x))
 ownership=defaultdict(list);expected_incidence=[[] for _ in range(N)];mass=[0.]*N;total_qvolume=Q(0);total_volume=math.fsum(e[7] for e in elements);peak_matrix=Q(0);peak_consistency=Q(0);peak_lu_error=0.;peak_geometric_error=0.;min_j=math.inf;max_j=-math.inf;minimum=None;min_serialized=math.inf;all_tets=set();det_rows=[]
 for i,e in enumerate(elements):
  ids=e[:4];assert len(set(ids))==4 and min(ids)>=0 and max(ids)<N;key=tuple(sorted(ids));assert key not in all_tets;all_tets.add(key)
  inv=[e[4+4*r:7+4*r] for r in range(3)];volume=e[7];material=e[16:20];control=e[20:24]
  assert all(math.isfinite(x) for x in e[4:20]) and volume>0 and e[11]==e[15]==0 and material==(3000,6000,0,0) and control==(1,0,0,0)
  row=topology[i];assert int(row['element'])==i and tuple(int(row['node'+str(k)]) for k in range(4))==ids;assert bits(row['rest_volume_m3'])==bits(volume) and bits(row['mu_Pa'])==bits(material[0]) and bits(row['lambda_Pa'])==bits(material[1])
  dm=edge(qrest,ids);ds=edge(qcurrent,ids);qi=[[qt(x) for x in row] for row in inv]
  dr=det_exact(dm);dc=det_exact(ds);di=det_exact(qi);assert dr>0 and dc>0 and di>0
  total_qvolume+=dr/6;exactj=dc*di;geometric=dc/dr;consistency=abs(6*Q.from_float(volume)*di-1);matrix_error=inverse_residual(dm,qi)
  assert exactj>q32(f32(1e-6)) and consistency<=q32(f32(2e-4));assert matrix_error<Q(1,1000000)
  peak_matrix=max(peak_matrix,matrix_error);peak_consistency=max(peak_consistency,consistency)
  fpdm=[[float(x) for x in r] for r in dm];fpds=[[float(x) for x in r] for r in ds];fpj=det_lu(mul(fpds,inv));peak_lu_error=max(peak_lu_error,abs(fpj-float(exactj)));peak_geometric_error=max(peak_geometric_error,abs(float(exactj-geometric)))
  if float(exactj)<min_j:min_j=float(exactj);minimum={'element':i,'nodes':list(ids),'exact_detF':str(exactj),'detF_FP64':float(exactj),'geometric_J_FP64':float(geometric)}
  max_j=max(max_j,float(exactj));serialized_j=det_lu(edge(serialized[38],ids))/(6*float(row['rest_volume_m3']));min_serialized=min(min_serialized,serialized_j)
  det_rows.append([i,*ids,float(exactj),fpj,float(geometric),float(matrix_error),float(consistency)])
  contribution=f32(f32(1000*volume)/4)
  for corner,n in enumerate(ids):mass[n]+=contribution;expected_incidence[n].append((i,corner,n,0))
  a,b,c,d=ids
  for face in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):ownership[tuple(sorted(face))].append((i,face,parity(face)))
 assert offsets[0]==0 and offsets[-1]==4*E and all(offsets[i]<=offsets[i+1] for i in range(N))
 for i in range(N):assert incidence[offsets[i]:offsets[i+1]]==expected_incidence[i] and bits(f32(mass[i]))==bits(vm[i][3])
 assert all(len(owners) in (1,2) for owners in ownership.values());assert all(owners[0][2]==-owners[1][2] for owners in ownership.values() if len(owners)==2)
 boundary={key:owners[0] for key,owners in ownership.items() if len(owners)==1};assert len(boundary)==1280 and len(set(tuple(sorted(f)) for f in faces))==1280 and set(boundary)=={tuple(sorted(f)) for f in faces}
 boundary_edges=defaultdict(list);boundary_nodes=set();boundary_qvolume=Q(0)
 for f in faces:
  key=tuple(sorted(f));assert parity(f)==boundary[key][2];boundary_nodes.update(f);boundary_qvolume+=det_exact([qrest[i] for i in f])/6
  for k in range(3):a,b=f[k],f[(k+1)%3];boundary_edges[tuple(sorted((a,b)))].append(1 if a<b else -1)
 assert all(sorted(directions)==[-1,1] for directions in boundary_edges.values());assert boundary_qvolume==total_qvolume
 euler=len(boundary_nodes)-len(boundary_edges)+len(faces);assert euler==2
 seen={0};adj=[set() for _ in range(N)]
 for e in elements:
  for a in e[:4]:adj[a].update(e[:4])
 frontier=[0]
 while frontier:
  a=frontier.pop()
  for b in adj[a]-seen:seen.add(b);frontier.append(b)
 assert len(seen)==N
 tet_neighbors=[[] for _ in range(E)]
 for owners in ownership.values():
  if len(owners)==2:
   a,b=owners[0][0],owners[1][0];tet_neighbors[a].append(b);tet_neighbors[b].append(a)
 seen_tets={0};frontier=[0]
 while frontier:
  for b in tet_neighbors[frontier.pop()]:
   if b not in seen_tets:seen_tets.add(b);frontier.append(b)
 assert len(seen_tets)==E
 selected_face=tuple(body['selected_boundary_face']);owners=ownership[tuple(sorted(selected_face))];assert len(owners)==1 and owners[0][0]==8768 and tuple(body['selected_face_owner'][0]['oriented_nodes'])==owners[0][1]
 provenance=json.loads((ROOT/'docs/assets/elastic-yarn-fem-frame38-provenance.json').read_text());assert selected_face==tuple(provenance['triangle_owning_nodes'])
 for n,state in zip(selected_face,provenance['states'][2:]):assert all(bits(a)==bits(b) for a,b in zip(list(current[n][:3])+list(vm[n]),state['position']+state['velocity']+[state['mass']]))
 energy=list(csv.DictReader((ROOT/'docs/assets/deformable-support-verlet-level3.csv').open()));captured=next(x for x in energy if int(x['frame'])==38);assert int(captured['accepted_steps'])==1900 and int(captured['rejected_steps'])==0 and int(captured['integrator'])==1
 assert abs(min_j-producer['minimum_detF_FP64_source_reconstructed_inverse'])<1e-12 and abs(max_j-producer['maximum_detF_FP64_source_reconstructed_inverse'])<1e-12
 assert abs(min_serialized-producer['independently_reproduced_serialized_volume_ratio_minimum'])<1e-12
 total_mass=math.fsum(r[3] for r in vm);mu=3000.;lam=6000.;rho=1000.;bulk=lam+2*mu/3;young=mu*(3*lam+2*mu)/(lam+mu);poisson=lam/(2*(lam+mu));assert mu>0 and bulk>0 and rho>0 and -1<poisson<.5
 audit={'schema':'numi.elastic-yarn.full-fem-pack.independent-audit.v1','source_commit':source['physics_source_commit'],'source_binding':git_bindings,'receipt_bindings_verified':receipt_checks,'retained_input_bindings_verified':input_checks,'prepared_output_bindings_verified':output_checks,'node_count':N,'tet_count':E,'boundary_faces':len(faces),'boundary_nodes':len(boundary_nodes),'boundary_edges':len(boundary_edges),'boundary_euler_characteristic':euler,'volume_graph_connected_nodes':len(seen),'tet_faces_unique':len(ownership),'tet_face_adjacency_connected_elements':len(seen_tets),'raw_layout_bytes':{name:(PACK/name).stat().st_size for name in ('positions-frame38.f32x4.bin','velocity-mass-frame38.f32x4.bin','rest-positions.f32x4.bin','elements.abi1.bin','boundary-faces.u32x3.bin','incidence-offsets.u32.bin','incidence.u32x4.bin')},'incidence_entries':len(incidence),'represented_trace_fields_uniquely_recovered':fieldcount,'raw_JSON_and_trace_fields_bit_exact':True,'owning_mass_reconstruction_bit_exact':True,'total_lumped_mass_kg':total_mass,'represented_rest_volume_m3':total_volume,'exact_geometric_rest_volume_m3':str(total_qvolume),'geometric_rest_volume_m3_FP64':float(total_qvolume),'boundary_volume_exactly_matches_all_tetrahedra':True,'mass_minus_density_times_represented_volume_kg':total_mass-rho*total_volume,'minimum_detF':min_j,'maximum_detF':max_j,'minimum_detF_witness':minimum,'maximum_exact_rest_matrix_inverse_residual':float(peak_matrix),'maximum_exact_rest_volume_inverse_determinant_consistency_error':float(peak_consistency),'maximum_independent_FP64_LU_vs_exact_detF_error':peak_lu_error,'maximum_source_inverse_vs_exact_geometric_J_difference':peak_geometric_error,'minimum_serialized_volume_ratio':min_serialized,'selected_face':selected_face,'selected_face_owner':owners[0][0],'selected_nodal_masses_kg':[vm[n][3] for n in selected_face],'material':{'mu_Pa':mu,'lambda_Pa':lam,'density_kg_m3':rho,'small_strain_Young_modulus_Pa':young,'small_strain_Poisson_ratio':poisson,'bulk_modulus_Pa':bulk,'compressional_wave_speed_m_s':math.sqrt((lam+2*mu)/rho),'shear_wave_speed_m_s':math.sqrt(mu/rho),'calibrated':False},'captured_frame':38,'nominal_recorded_time_s':.19,'represented_timestep_product_s':1900*f32(1e-4),'accepted_steps':1900,'rejected_steps':0,'integrator':'support_verlet','capture_limits':['Historical uploaded inverse-rest element buffer has no retained raw dump/hash; this is reconstructed frozen-source data.','Native frame38 per-element detF/forces/stress/energy outputs were not exported.','Frame38 current/cumulative failure bits, support-impulse/minJ/net-force status entries and full96-byte native status were not exported.','Previous/free/candidate element outputs and buffers are not a bit-exact continuation checkpoint.'],'no_step_advance':True,'no_gpu':True,'full_scene_qualified':False,'elapsed_s':time.monotonic()-begin}
 (HERE/'audit-result.json').write_text(json.dumps(audit,indent=2)+'\n')
 with (HERE/'determinants.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['element','node0','node1','node2','node3','exact_detF_rounded_FP64','independent_pivoted_LU_detF','exact_geometric_J_rounded_FP64','exact_matrix_inverse_residual','exact_volume_inverse_consistency_error']);w.writerows(det_rows)
 print(json.dumps({k:v for k,v in audit.items() if k in ('node_count','tet_count','boundary_faces','boundary_euler_characteristic','represented_trace_fields_uniquely_recovered','total_lumped_mass_kg','geometric_rest_volume_m3_FP64','minimum_detF','maximum_detF','maximum_exact_rest_matrix_inverse_residual','maximum_independent_FP64_LU_vs_exact_detF_error','accepted_steps','rejected_steps','elapsed_s')},sort_keys=True));return 0

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--negative',choices=['mass','inverse-shear','truncate']);args=p.parse_args()
 sys.exit(negative(args.negative) if args.negative else main())
