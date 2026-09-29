#!/usr/bin/env python3
"""Freeze the complete represented frame38 FEM body; no time advancement."""
import csv,hashlib,json,math,pathlib,shutil,struct
from collections import Counter,defaultdict
from decimal import Decimal as D,localcontext

P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[1]
N=2057;E=10240
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def bits(x):return struct.pack('<f',float(x))
def records(path,fmt):return list(struct.iter_unpack(fmt,path.read_bytes()))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def determinant(rows):return dot(rows[0],cross(rows[1],rows[2]))
def edge_matrix(v,nodes):
    edges=[sub(v[nodes[k]][:3],v[nodes[0]][:3]) for k in (1,2,3)]
    return [[edges[c][r] for c in range(3)] for r in range(3)]
def matmul(a,b):return [[math.fsum(a[r][k]*b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
def inverse_independent(a):
    # Pivoted Gauss-Jordan, independent from the frozen cross/determinant authoring.
    m=[list(row)+[float(r==c) for c in range(3)] for r,row in enumerate(a)]
    for k in range(3):
        pivot=max(range(k,3),key=lambda r:abs(m[r][k]));m[k],m[pivot]=m[pivot],m[k]
        q=m[k][k];assert q!=0
        m[k]=[x/q for x in m[k]]
        for r in range(3):
            if r==k:continue
            q=m[r][k];m[r]=[m[r][c]-q*m[k][c] for c in range(6)]
    return [row[3:] for row in m]
def quantized_trace_value(token):
    value=f32(token);assert math.isfinite(value)
    d=D(token)
    if d==0:return value
    # The owning exporter uses defaultfloat precision12. Prove its whole
    # printed decimal rounding cell selects exactly this represented FP32.
    u=struct.unpack('<I',bits(value))[0]
    prev=struct.unpack('<f',struct.pack('<I',u-1 if value>0 else u+1))[0]
    nex=struct.unpack('<f',struct.pack('<I',u+1 if value>0 else u-1))[0]
    radius=D(5).scaleb(d.adjusted()-12)
    lo=(D.from_float(prev)+D.from_float(value))/2
    hi=(D.from_float(value)+D.from_float(nex))/2
    assert lo<d-radius and d+radius<hi,(token,value,'ambiguous FP32 recovery')
    return value

def main():
    provenance_path=ROOT/'docs/assets/elastic-yarn-fem-frame38-provenance.json'
    provenance=json.loads(provenance_path.read_text())
    published=json.loads((ROOT/'docs/assets/deformable-support-evidence.json').read_text())
    source=json.loads((P/'source-manifest.json').read_text())
    for name,expected in source['source_blob_sha256'].items():
        assert sha(P/'frozen'/name)==expected==published['frozen_source_binary_sha256'][name]
    owning=(P/'frozen/tools/deformable_mesh.mm').read_text()
    owning_authoring=owning[owning.index('namespace {'):owning.index('struct Frame {')]
    assert owning_authoring in (P/'authoring.cpp').read_text(), 'owning authoring extraction changed'
    inputs={str(provenance_path):sha(provenance_path)}
    for name,expected in provenance['source_sha256'].items():
        q=ROOT/name;assert sha(q)==expected,(q,'published input changed');inputs[str(q)]=expected
    for name in ['docs/assets/deformable-support-verlet-level3.log','docs/assets/deformable-support-verlet-level3.csv',
        'docs/assets/deformable-support-10240-geometry.json','docs/assets/deformable-support-evidence.json',
        'docs/assets/elastic-yarn-fem-frame38-five-nodes.txt']:
        inputs[str(ROOT/name)]=sha(ROOT/name)
    runs=json.loads((P/'authoring-runs.json').read_text())
    assert all(r['actual_exit']==0 and r['outputs_sha256']==runs[0]['outputs_sha256'] for r in runs)
    authored=P/'authoring-off-output'
    rest=records(authored/'rest-positions.f32x4.bin','<4f')
    rest_vm=records(authored/'rest-velocity-mass.f32x4.bin','<4f')
    raw_elements=records(authored/'elements.abi1.bin','<4I16f4I')
    faces=records(authored/'boundary-faces.u32x3.bin','<3I')
    offsets=[x[0] for x in records(authored/'incidence-offsets.u32.bin','<I')]
    incidence=records(authored/'incidence.u32x4.bin','<4I')
    assert len(rest)==len(rest_vm)==N and len(raw_elements)==E and len(faces)==1280 and len(incidence)==4*E
    assert len(offsets)==N+1 and offsets[0]==0 and offsets[-1]==4*E
    trace_path=ROOT/'docs/assets/deformable-support-10240-trajectory.csv'
    selected={0:[],38:[]};serialized={0:[],38:[]};counts=Counter();roundtrip_values=0
    with localcontext() as ctx:
        ctx.prec=60
        for row in csv.DictReader(trace_path.open()):
            frame=int(row['frame']);node=int(row['node'])
            assert node==counts[frame];counts[frame]+=1
            if frame not in selected:continue
            assert float(row['time_s'])==frame*.005
            fields=['x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','mass_kg']
            selected[frame].append(tuple(quantized_trace_value(row[k]) for k in fields))
            serialized[frame].append(tuple(float(row[k]) for k in fields))
            roundtrip_values+=len(fields)
    assert len(counts)==101 and all(c==N for c in counts.values())
    assert len(selected[0])==len(selected[38])==N
    current=selected[38]
    for n in range(N):
        assert all(bits(selected[0][n][k])==bits(rest[n][k]) for k in range(3)),(n,'rest bits')
        assert all(bits(selected[0][n][k+3])==bits(rest_vm[n][k]) for k in range(3))
        assert bits(selected[0][n][6])==bits(current[n][6])==bits(rest_vm[n][3]),(n,'mass bits')
    topology=list(csv.DictReader((ROOT/'docs/assets/deformable-support-10240-topology.csv').open()))
    assert len(topology)==E
    elements=[];owners=defaultdict(list);mass=[0.]*N;mass_fp64=[0.]*N
    exact_volume=0.;represented_volume=0.;min_j=math.inf;max_j=-math.inf;max_rest_error=0.;max_j_difference=0.
    max_inverse_reconstruction_error=0.;min_serialized=math.inf;witness=None;det_rows=[]
    for e,(raw,row) in enumerate(zip(raw_elements,topology)):
        nodes=raw[:4];floats=raw[4:20];control=raw[20:24]
        inverse=[list(floats[4*r:4*r+3]) for r in range(3)];volume=floats[3];material=floats[12:16]
        assert int(row['element'])==e and tuple(int(row['node'+str(k)]) for k in range(4))==nodes
        assert bits(row['rest_volume_m3'])==bits(volume) and f32(row['mu_Pa'])==material[0]==3000 and f32(row['lambda_Pa'])==material[1]==6000
        assert material[2:]==(0.,0.) and control==(1,0,0,0)
        assert floats[7]==floats[11]==0
        # Owning source's integer literals promote to the float volume type:
        # FP32 density*volume/4 contribution, then += into a double accumulator.
        contribution=f32(f32(1000*volume)/4)
        for n in nodes:
            mass[n]+=contribution
            mass_fp64[n]+=1000*volume/4
        dm=edge_matrix(rest,nodes);ds=edge_matrix(current,nodes)
        true_volume=determinant(dm)/6;assert true_volume>0 and volume>0
        inverse64=inverse_independent(dm)
        max_inverse_reconstruction_error=max(max_inverse_reconstruction_error,max(abs(inverse[r][c]-inverse64[r][c]) for r in range(3) for c in range(3)))
        residual=abs(6*volume*determinant(inverse)-1);max_rest_error=max(max_rest_error,residual)
        j=determinant(matmul(ds,inverse));geometric=determinant(ds)/determinant(dm)
        exported=determinant(edge_matrix(serialized[38],nodes))/(6*float(row['rest_volume_m3']))
        assert j>f32(1e-6) and residual<=f32(2e-4)
        min_serialized=min(min_serialized,exported);max_j=max(max_j,j)
        max_j_difference=max(max_j_difference,abs(j-geometric))
        if j<min_j:min_j=j;witness={'element':e,'nodes':list(nodes),'detF_FP64':j,'geometric_current_over_rest_determinant':geometric,'serialized_volume_ratio':exported}
        exact_volume+=true_volume;represented_volume+=volume
        det_rows.append([e,*nodes,j,geometric,exported,residual])
        elements.append({'nodes':list(nodes),'inverse_rest_rows_f32':inverse,'rest_volume_m3_f32':volume,'material_f32':list(material),'control_u32':list(control)})
        a,b,c,d=nodes
        for face in [(b,c,d),(a,d,c),(a,b,d),(a,c,b)]:owners[tuple(sorted(face))].append({'element':e,'oriented_nodes':face})
    assert all(len(v) in (1,2) for v in owners.values())
    boundary=[key for key,v in owners.items() if len(v)==1]
    assert len(boundary)==len(faces)==1280 and set(boundary)=={tuple(sorted(f)) for f in faces}
    for n in range(N):
        assert bits(f32(mass[n]))==bits(current[n][6])
        for entry in incidence[offsets[n]:offsets[n+1]]:
            element,corner,owner,reserved=entry
            assert owner==n and reserved==0 and elements[element]['nodes'][corner]==n
    selected_face=tuple(provenance['triangle_owning_nodes']);selected_owners=owners[tuple(sorted(selected_face))]
    assert len(selected_owners)==1
    fixture_checks=[]
    for n,state in zip(selected_face,provenance['states'][2:]):
        expected=tuple(state['position']+state['velocity']+[state['mass']])
        assert all(bits(a)==bits(b) for a,b in zip(current[n],expected)),(n,'fixture owning state changed')
        fixture_checks.append({'node':n,'position':list(current[n][:3]),'velocity':list(current[n][3:6]),'mass':current[n][6],'all_seven_fields_bit_exact':True})
    total_mass=math.fsum(row[6] for row in current)
    energy_rows=list(csv.DictReader((ROOT/'docs/assets/deformable-support-verlet-level3.csv').open()))
    energy38=next(r for r in energy_rows if int(r['frame'])==38)
    assert int(energy38['accepted_steps'])==1900 and int(energy38['rejected_steps'])==0
    geometry=json.loads((ROOT/'docs/assets/deformable-support-10240-geometry.json').read_text())
    retained_ratio=next(row['minimum_volume_ratio'] for row in geometry['snapshots'] if row['frame']==38)
    assert abs(retained_ratio-min_serialized)<1e-12
    # A single file contains all physical owners and elements. Source authored
    # inverse arrays are reconstructed; every exported rest/volume/mass bit agrees.
    pack={'schema':'numi.elastic-yarn.complete-fem-body.cpu-ready.v1','physics_source_revision':source['physics_source_commit'],
        'recorded_frame':38,'recorded_time_s':.19,'nodes':[
            {'rest_position_f32':list(rest[n][:3]),'position_f32':list(current[n][:3]),
             'velocity_f32':list(current[n][3:6]),'lumped_mass_kg_f32':current[n][6]} for n in range(N)],
        'elements':elements,'boundary_faces':list(faces),'node_incidence_offsets':offsets,'node_incidence_entries':list(incidence),
        'material':{'constitutive_model':'compressible Neo-Hookean as frozen evaluateTet','mu_Pa':3000,'lambda_Pa':6000,'density_kg_m3':1000,'calibrated':False},
        'selected_boundary_face':list(selected_face),'selected_face_owner':selected_owners,
        'state_kind':'native accepted frame38 body before authored yarn-contact fixture response',
        'boundary':'Complete physical body state prepared for CPU math; no FEM time advancement or coupled contact performed. Cumulative native status and work ledgers are not a byte-exact checkpoint.'}
    (P/'body.json').write_text(json.dumps(pack,separators=(',',':'),allow_nan=False)+'\n')
    current_positions=[(*row[:3],0.) for row in current]
    velocity_mass=[(*row[3:6],row[6]) for row in current]
    (P/'positions-frame38.f32x4.bin').write_bytes(b''.join(struct.pack('<4f',*v) for v in current_positions))
    (P/'velocity-mass-frame38.f32x4.bin').write_bytes(b''.join(struct.pack('<4f',*v) for v in velocity_mass))
    for name in ['rest-positions.f32x4.bin','elements.abi1.bin','boundary-faces.u32x3.bin','incidence-offsets.u32.bin','incidence.u32x4.bin']:
        shutil.copyfile(authored/name,P/name)
    with (P/'detF-frame38.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['element','node0','node1','node2','node3','detF_FP64_source_inverse','geometric_J_FP64','serialized_volume_ratio_FP64','rest_consistency_error']);writer.writerows(det_rows)
    output_files=['body.json','positions-frame38.f32x4.bin','velocity-mass-frame38.f32x4.bin','rest-positions.f32x4.bin','elements.abi1.bin','boundary-faces.u32x3.bin','incidence-offsets.u32.bin','incidence.u32x4.bin','detF-frame38.csv']
    report={'schema':'numi.elastic-yarn.complete-fem-preparation-evidence.v1','actual_preparation_result':'PASS',
        'physics_source_revision':source['physics_source_commit'],'source_manifest_sha256':sha(P/'source-manifest.json'),
        'prepare_script_sha256':sha(pathlib.Path(__file__).resolve()),'input_sha256':inputs,'outputs_sha256':{str(P/name):sha(P/name) for name in output_files},
        'nodes':N,'elements':E,'boundary_triangles':1280,'incidence_entries':4*E,'frame':38,'recorded_time_s':.19,
        'represented_FP32_states_uniquely_recovered_from_precision12_trace_fields':roundtrip_values,
        'all_generated_rest_positions_and_masses_bit_exact_to_trace':True,'all_generated_connectivity_volumes_material_bit_exact_to_topology':True,
        'owning_authoring_extracted_without_formula_edits':True,'all_authoring_output_buffers_bit_identical_contraction_off_on_fast':True,
        'inverse_rest_binding':'Reconstructed by the exact frozen owning makeMesh function; all output buffers identical across off/on/fast contraction builds. All rest, volume, mass and topology buffers match retained trace bits. Original uploaded inverse-rest buffer was not separately exported, so this is source-bound deterministic reconstruction rather than a direct historical buffer hash.',
        'total_lumped_mass_kg':total_mass,'total_represented_rest_volume_m3':represented_volume,'total_geometric_rest_volume_m3_FP64':exact_volume,
        'lumped_mass_minus_density_times_represented_volume_kg':total_mass-1000*represented_volume,
        'maximum_node_mass_difference_owning_FP32_contribution_vs_FP64_volume_sum_kg':max(abs(mass[n]-mass_fp64[n]) for n in range(N)),
        'mass_binding_arithmetic':'FP32(1000*represented_volume)/4 contribution accumulated inFP64, then owning nodal mass cast toFP32; matches every trace mass bit.',
        'selected_face_owner':selected_owners,'selected_face_three_nodal_states':fixture_checks,
        'minimum_detF_FP64_source_reconstructed_inverse':min_j,'maximum_detF_FP64_source_reconstructed_inverse':max_j,
        'minimum_detF_witness':witness,'maximum_FP64_J_difference_source_inverse_vs_geometric_rest':max_j_difference,
        'maximum_inverse_row_difference_F32_vs_independent_FP64':max_inverse_reconstruction_error,
        'maximum_rest_consistency_error_FP64':max_rest_error,
        'frozen_native_limits':{'detF_strictly_greater_than':float(f32(1e-6)),'rest_consistency_abs_error_at_most':float(f32(2e-4)),
            'native_all_5000_steps_minimum_accepted_detF_reported':.21475289762,'frame38_native_elementwise_detF_exported':False},
        'retained_frame38_serialized_geometric_volume_ratio_minimum':retained_ratio,'independently_reproduced_serialized_volume_ratio_minimum':min_serialized,
        'frame38_all_elements_pass_reconstructed_native_detF_and_rest_limits':True,
        'accepted_steps_at_captured_frame':1900,'rejected_steps_at_captured_frame':0,
        'native_elapsed_timestep_product_s':1900*float(f32(1e-4)),
        'binary_layout':{'byte_order':'little','positions_and_velocity_mass':'2057 records of4 float32; position.w reserved0; velocity.w owning mass','element':'10240 96-byte NumiDeformableMeshElement records ABI1; nodes uint4,3 inverse float4 rows(volume in row0.w),material float4,control uint4','incidence':'40960 uint4(element,localCorner,owningNode,0);2058 uint offsets'},
        'gaps':['No direct historical inverse-rest-buffer dump/hash; exact owning-source reconstruction is supplied and independently cross-bound to all retained authoring outputs.','No captured per-element native frame38 detF values; FP64 results are compared to unchanged native limits and retained geometric ratio instead.','No full cumulative native status/work checkpoint; this pack contains complete physical owners/material/topology for a new coupled CPU step.'],
        'boundary':'Preparation only. No toy tetrahedron, invented material/state, GPU/new trajectory/production edit, elastic advance, yarn-FEM coupled step, contact-work closure or full scene qualification.'}
    (P/'evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,expected in inputs.items():assert sha(pathlib.Path(name))==expected
    print('prepared_complete_fem_body=PASS nodes='+str(N)+' elements='+str(E)+' boundary_faces=1280 incidence=40960')
    print('total_mass_kg='+repr(total_mass)+' represented_rest_volume_m3='+repr(represented_volume))
    print('frame38_minimum_detF_FP64='+repr(min_j)+' maximum='+repr(max_j)+' rest_consistency_max='+repr(max_rest_error))
    print('selected_face='+str(selected_face)+' owner='+str(selected_owners)+' all_three_nodal_states_bit_exact=1')
    print('all_authoring_buffers_contraction_modes_exact=1 historical_inverse_buffer_dump=ABSENT no_step_advance=1 no_gpu=1')

if __name__=='__main__':main()
