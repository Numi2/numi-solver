#!/usr/bin/env python3
"""Compact repeated full-body/yarn source-frozen normal-block trajectory.

Each accepted step computes all source stresses/J/forces, full owner state,
all boundary distances and global normal velocity. Keep full states at only
initial/matched/terminal points; retain per-step hashes and physics ledgers.
"""
import argparse,copy,gzip,hashlib,importlib.util,json,math,pathlib,struct,sys,time
P=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('endpoint_multistep',P/'endpoint_step.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)
B=N.B
SOURCE_SHA='57115c3a0c82241d0567f237d937864f1d0e7734e580e126cc95aa5f35e75241'
NORMAL_SHA='dcfaeca294146fbec3e161dce05a7583d0ffdf34e8e181e2a008c92a0dc22176'
CENTROID_SHA='b249e25faaeba30e486e8c2b70d76ab875b69d3a716518f9a62858a6c5b4379b'
assert B.sha(P/'endpoint_step.py')==SOURCE_SHA and B.sha(P/'step.py')==NORMAL_SHA and B.sha(P/'centroid_step.py')==CENTROID_SHA
def h(payload):return hashlib.sha256(payload).hexdigest()
def save_full_state(prefix,state):
    raw=b''.join(struct.pack('<6d',*(state['x'][i]+state['v'][i])) for i in range(len(state['x'])))
    metadata={k:v for k,v in state.items() if k not in ('x','v')}
    zipped=gzip.compress(json.dumps(metadata,sort_keys=True,allow_nan=False,separators=(',',':')).encode(),mtime=0)
    position_file=pathlib.Path(str(prefix)+'.state.f64x6.bin');meta_file=pathlib.Path(str(prefix)+'.metadata.json.gz')
    position_file.write_bytes(raw);meta_file.write_bytes(zipped)
    # A retained checkpoint must reconstruct the COMPLETE accepted state,
    # including clock/warm/cache/ledger, byte-identically before publication.
    reconstructed=copy.deepcopy(metadata)
    records=list(struct.iter_unpack('<6d',position_file.read_bytes()))
    reconstructed['x']=[list(q[:3]) for q in records]
    reconstructed['v']=[list(q[3:]) for q in records]
    retained=json.loads(gzip.decompress(meta_file.read_bytes()))
    reconstructed.update(retained)
    assert len(records)==2059 and B.state_sha(reconstructed)==B.state_sha(state)
    return {'position_file':position_file.name,'position_sha256':B.sha(position_file),'position_bytes':len(raw),
      'metadata_file':meta_file.name,'metadata_sha256':B.sha(meta_file),'metadata_bytes':len(zipped),
      'complete_state_sha256':B.state_sha(state),'elapsed_s':state['elapsed'],'steps':state['steps']}
def record_step(m,state,returned,index,dt):
    e=returned['final_eval'];ledger=returned['ledger'];scan=ledger['final_global_velocity_scan']
    assert len(state['x'])==len(state['v'])==len(m.mass)==2059
    assert len(e['outputs'])==10240 and len(e['force'])==2059
    assert e['minJ']>B.J_GATE and math.isfinite(e['maxJ'])
    assert all(B.finite(q) for q in state['x']+state['v']+e['force'])
    assert all(B.finite(q) for q in e['outputs'])
    assert ledger['final_areas']['faces']==scan['faces_checked']==1280
    assert scan['minimum']['gap_m']>=-B.GAP_GATE
    assert scan['minimum']['closing_m_s']>=-512*B.EPS*max(1.,max(B.norm(state['v'][i]) for i in m.row))
    assert all(mass>0 and math.isfinite(mass) for mass in m.mass)
    stress_sha=h(b''.join(struct.pack('<11d',*q) for q in e['outputs']))
    force_sha=h(b''.join(struct.pack('<3d',*q) for q in e['force']))
    normal=ledger['contact_rows']
    assert len(normal)==2 and all(q['normal_block'] and q['kinetic_delta_J']<=0 for q in normal)
    assert all(len(q['row_impulses_Ns'])==2 and min(q['row_impulses_Ns'])>=0 for q in normal)
    gap=scan['minimum']['gap_m'];rate=scan['minimum']['closing_m_s']
    return {'step':index,'elapsed_s':state['elapsed'],'dt_s':dt,'state_sha256':B.state_sha(state),
      'source_step_after_sha256':returned['after_sha'],'owners_checked':2059,'elements_checked':10240,
      'stress_J_U_sha256':stress_sha,'force_sha256':force_sha,
      'minJ':e['minJ'],'maxJ':e['maxJ'],'minimum_element':e['min_element'],
      'minimum_face':scan['minimum']['face'],'minimum_face_nodes':scan['minimum']['nodes'],
      'minimum_gap_m':gap,'minimum_closing_m_s':rate,
      'endpoint_gaps_m':[q['gap'] for q in normal[-1]['geometry']['endpoint_rows']],
      'row_active_masks':[q['LCP_active_mask'] for q in normal],
      'row_impulses_Ns':[q['row_impulses_Ns'] for q in normal],
      'contact_dissipation_J':ledger['contact_dissipation_J'],
      'plane_kinetic_removal_J':ledger['plane_kinetic_removal_J'],
      'plane_impulse_Ns':ledger['plane_impulse_Ns'],
      'gravity_impulse_Ns':ledger['gravity_impulse_Ns'],
      'support_events':len(ledger['plane_events']),'position_projections':len(ledger['position_projections']),
      'gravity_work_J':ledger['gravity_work_J'],
      'position_potential_work_J':ledger['position_potential_work_J'],
      'plane_force_response_kinetic_J':ledger['plane_force_response_kinetic_J'],
      'free_split_integration_residual_J':ledger['free_split_integration_residual_J'],
      'physical_energy_balance_residual_J':ledger['physical_energy_balance_residual_J'],
      'bookkeeping_identity_error_J':ledger['bookkeeping_identity_error_J'],
      'start_energy_J':ledger['start']['total_J'],'end_energy_J':ledger['end']['total_J'],
      'start_momentum_kg_m_s':ledger['start']['P_kg_m_s'],'end_momentum_kg_m_s':ledger['end']['P_kg_m_s'],
      'start_angular_kg_m2_s':ledger['start']['L_kg_m2_s'],'end_angular_kg_m2_s':ledger['end']['L_kg_m2_s'],
      'momentum_balance_residual_Ns':ledger['momentum_balance_residual_Ns'],
      'angular_balance_residual_Nms':ledger['angular_balance_residual_Nms']}
def run_pass(label,model,dt,max_steps,output,interval_steps=0,expected=None):
    state=model.initial();initial_sha=B.state_sha(state);records=[];snapshot=None;frontier=None
    stream=output/(label+'.jsonl')
    with stream.open('w') as f:
        for i in range(max_steps):
            before=B.state_sha(state);before_clock=state['elapsed'];before_count=state['steps']
            returned=N.advance(model,state,dt)
            if not returned['accepted']:
                assert returned['rollback_exact'] and B.state_sha(state)==before
                assert state['elapsed']==before_clock and state['steps']==before_count
                frontier={'attempted_step':i+1,'reason':returned['reason'],'detail':returned.get('detail'),
                  'accepted_elapsed_s':state['elapsed'],'rollback_exact':True,'state_sha256':before}
                f.write(json.dumps({'frontier':frontier},sort_keys=True)+'\n');f.flush()
                print(json.dumps({'pass':label,'frontier_step':i+1,'reason':returned['reason'],'accepted_steps':i,'clock_s':state['elapsed']}),flush=True)
                break
            rec=record_step(model,state,returned,i+1,dt)
            if expected is not None:
                if i>=len(expected) or rec!=expected[i]:
                    raise AssertionError(('exact replay diverged',label,i+1,rec,expected[i] if i<len(expected) else None))
            records.append(rec);f.write(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n');f.flush()
            if interval_steps and i+1==interval_steps:snapshot=copy.deepcopy(state)
            if i<3 or (i+1)%8==0:
                print(json.dumps({'pass':label,'step':i+1,'elapsed_us':state['elapsed']*1e6,
                  'gap_um':rec['minimum_gap_m']*1e6,'minJ':rec['minJ'],
                  'normal_m_s':rec['minimum_closing_m_s']}),flush=True)
    return {'state':state,'initial_sha256':initial_sha,'records':records,'frontier':frontier,
      'interval_state':snapshot,'log_sha256':B.sha(stream),'log_bytes':stream.stat().st_size}
def summarize(records):
    if not records:return None
    first=records[0];last=records[-1]
    contact=math.fsum(q['contact_dissipation_J'] for q in records)
    plane_loss=math.fsum(q['plane_kinetic_removal_J'] for q in records)
    work=math.fsum(q['gravity_work_J'] for q in records)
    residual=math.fsum(q['physical_energy_balance_residual_J'] for q in records)
    return {'accepted_steps':len(records),'elapsed_s':last['elapsed_s'],'initial_energy_J':first['start_energy_J'],
      'final_energy_J':last['end_energy_J'],'contact_dissipation_J':contact,
      'plane_kinetic_removal_J':plane_loss,'gravity_work_J':work,
      'cumulative_physical_energy_residual_J':residual,
      'direct_physical_energy_residual_J':last['end_energy_J']-first['start_energy_J']+contact+plane_loss,
      'maximum_absolute_per_step_energy_residual_J':max(abs(q['physical_energy_balance_residual_J']) for q in records),
      'minimum_J':min(q['minJ'] for q in records),
      'minimum_gap_m':min(q['minimum_gap_m'] for q in records),
      'maximum_speed_of_yarn_globally_not_recorded':True,
      'support_impulse_Ns':B.sumv([q['plane_impulse_Ns'] for q in records]),
      'gravity_impulse_Ns':B.sumv([q['gravity_impulse_Ns'] for q in records]),
      'support_events':sum(q['support_events'] for q in records),
      'position_projections':sum(q['position_projections'] for q in records),
      'all_steps_full_body_J_stress_force_owner_and_boundary_checked':True}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dt-us',type=float,default=20.0)
    parser.add_argument('--max-steps',type=int,default=128)
    parser.add_argument('--interval-us',type=float,default=320.0)
    parser.add_argument('--output',type=pathlib.Path,default=P/'candidate-20us')
    args=parser.parse_args();start=time.time()
    assert args.dt_us>0 and args.max_steps>=2 and args.interval_us>0
    dt=args.dt_us*1e-6;half=dt/2
    assert math.isfinite(dt) and half+half==dt
    interval_steps=round(args.interval_us/args.dt_us)
    assert interval_steps>=2 and abs(interval_steps*dt-args.interval_us*1e-6)<1e-18
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    model=N.Model();source_inputs=model.inputs.copy()
    primary=run_pass('primary',model,dt,args.max_steps,output,interval_steps)
    accepted=len(primary['records']);terminal=primary['state'];terminal_binding=save_full_state(output/'terminal',terminal)
    same_length=accepted+(primary['frontier'] is not None)
    replay=run_pass('replay',N.Model(),dt,same_length,output,expected=primary['records'])
    replay_exact=(len(replay['records'])==accepted and replay['frontier']==primary['frontier'] and
      B.state_sha(replay['state'])==B.state_sha(terminal))
    assert replay_exact
    matched=None
    if primary['interval_state'] is not None:
        match_steps=2*interval_steps
        finer=run_pass('half',N.Model(),half,match_steps,output)
        if finer['frontier'] is None and len(finer['records'])==match_steps:
            full=primary['interval_state'];fine=finer['state']
            full_binding=save_full_state(output/'interval-full',full)
            fine_binding=save_full_state(output/'interval-half',fine)
            full_obs=B.observables(model,full,model.evaluate(full['x'],False))
            fine_obs=B.observables(model,fine,model.evaluate(fine['x'],False))
            matched={'passed_admission_both':True,'interval_s':interval_steps*dt,
              'full_steps':interval_steps,'half_steps':match_steps,'full_checkpoint':full_binding,
              'half_checkpoint':fine_binding,'maximum_position_difference_m':max(B.norm(B.sub(a,b)) for a,b in zip(full['x'],fine['x'])),
              'maximum_velocity_difference_m_s':max(B.norm(B.sub(a,b)) for a,b in zip(full['v'],fine['v'])),
              'energy_difference_J':fine_obs['total_J']-full_obs['total_J'],
              'final_minimum_J_difference':fine_obs['minJ']-full_obs['minJ'],
              'full_global_gap_m':primary['records'][interval_steps-1]['minimum_gap_m'],
              'half_global_gap_m':finer['records'][-1]['minimum_gap_m'],
              'full_cumulative_energy_residual_J':summarize(primary['records'][:interval_steps])['cumulative_physical_energy_residual_J'],
              'half_cumulative_energy_residual_J':summarize(finer['records'])['cumulative_physical_energy_residual_J'],
              'not_temporal_convergence':True}
        else:
            matched={'passed_admission_both':False,'interval_s':interval_steps*dt,'full_steps':interval_steps,
              'half_attempts':len(finer['records'])+(finer['frontier'] is not None),'half_frontier':finer['frontier'],
              'not_temporal_convergence':True}
    else:
        matched={'passed_admission_both':False,'reason':'Primary frontier before requested meaningful interval',
          'requested_interval_s':interval_steps*dt,'not_temporal_convergence':True}
    result={'schema':'numi.elastic-yarn.complete-FEM-endpoint-normal-block.multistep-short.v1',
      'dt_s':dt,'maximum_steps':args.max_steps,'interval_requested_s':interval_steps*dt,
      'source_sha256':SOURCE_SHA,'normal_block_source_sha256':NORMAL_SHA,'centroid_source_sha256':CENTROID_SHA,
      'trajectory_source_sha256':B.sha(pathlib.Path(__file__)),
      'input_sha256':source_inputs,'initial_state_sha256':primary['initial_sha256'],
      'primary_accepted_steps':accepted,'first_unsupported_frontier':primary['frontier'],
      'right_censored_at_max_steps':primary['frontier'] is None,
      'exact_replay':replay_exact,'replay_frontier':replay['frontier'],
      'full_body_summary':summarize(primary['records']),
      'primary_records':primary['records'],'primary_log_sha256':primary['log_sha256'],
      'replay_log_sha256':replay['log_sha256'],'terminal_checkpoint':terminal_binding,
      'matched_interval':matched,'runtime_s':time.time()-start,
      'scope':['Full actual 2059-owner positions/velocities and 10240 tetrahedral Piola/J/U/force evaluations checked and SHA-bound at each accepted step.',
        'All1280 boundary faces and global closest-feature normal velocity checked at each accepted step by the same numerical FP64 oracle.',
        'No GPU, native continuation, internal yarn elasticity, friction, moving CCD, whole bag/fruit scene or temporal convergence claim.',
        'Unchanged 2um contact, native J/rest/material, nonnegative shared normal impulses and actual kinetic passivity gates; new dt explicitly reported.',
        'Selected triangle interior and both yarn endpoints only; global closest segment/triangle witness may move to a yarn endpoint. Unsupported triangle features still reject.',
        'Source has no energy-closure gate: physical energy residual remains measured and open.']}
    result_path=output/'result.json';result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    receipt={'schema':'numi.elastic-yarn.endpoint-multistep-short.source-bound-receipt.v1',
      'actual_program_exit':0,'result_sha256':B.sha(result_path),'trajectory_source_sha256':result['trajectory_source_sha256'],
      'endpoint_source_sha256':SOURCE_SHA,'frozen_normal_source_sha256':NORMAL_SHA,'frozen_centroid_source_sha256':CENTROID_SHA,
      'input_sha256':source_inputs,'retained_outputs':{q.name:{'sha256':B.sha(q),'bytes':q.stat().st_size} for q in output.iterdir() if q.is_file() and q.name!='receipt.json'}}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'accepted':accepted,'frontier':primary['frontier']['reason'] if primary['frontier'] else None,
      'frontier_step':primary['frontier']['attempted_step'] if primary['frontier'] else None,
      'right_censored':primary['frontier'] is None,'exact_replay':replay_exact,
      'matched_interval':matched['passed_admission_both'],'runtime_s':result['runtime_s']}),flush=True)
    return 0
if __name__=='__main__':sys.exit(main())
