#!/usr/bin/env python3
"""Reproduce unchanged centroid and shared normal-block CPU steps from public inputs."""
import gzip,hashlib,json,pathlib,shutil,subprocess,sys,tempfile,time
HERE=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(q):return json.dumps(q,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
def metric_digest(q,keys,baseline_key):
    physical={k:q[k] for k in keys}
    physical[baseline_key]={k:v for k,v in q[baseline_key].items() if k!='independent_force_path'}
    return hashlib.sha256(canonical(physical)).hexdigest()
def check_outputs(folder,expected):
    identities={}
    for name,item in expected.items():
        target=folder/name; actual=sha(target)
        assert target.stat().st_size==item['bytes'] and actual==item['sha256'],('physical output changed',target)
        identities[name]={'sha256':actual,'bytes':item['bytes'],'byte_identical_to_owner':True}
    assert len(identities)==9
    return identities
def run_source(path,work):
    with (path.parent/'run.log').open('wb') as output:
        result=subprocess.run([sys.executable,str(path)],cwd=work,stdout=output,stderr=subprocess.STDOUT,check=False)
    (path.parent/'actual.exit').write_text(str(result.returncode)+'\n')
    assert result.returncode==0,('unchanged source rejected',path.parent/'run.log')
    return result.returncode
def main():
    begin=time.time()
    manifest_path=HERE/'reproduction-manifest.json'; manifest=json.loads(manifest_path.read_text())
    roots=[p for p in [HERE,*HERE.parents] if (p/'docs/assets/elastic-yarn-fem-body/payload-manifest.json').is_file()]
    if not roots:raise RuntimeError('Run within a repository containing the public full FEM pack.')
    repository=roots[0]; public=repository/'docs/assets/elastic-yarn-fem-body'
    assert sha(HERE/'step.py')==manifest['source_sha256']
    assert sha(HERE/'centroid_step.py')==manifest['centroid_source_sha256']
    assert sha(public/'payload-manifest.json')==manifest['public_payload_manifest_sha256']
    fixture=repository/'docs/assets/elastic-yarn-fem-frame38-provenance.json'
    assert sha(fixture)==manifest['public_fixture_sha256']
    review=HERE/manifest['independent_review_result_file']
    assert sha(review)==manifest['independent_review_result_sha256'] and manifest['independent_review_actual_exit']==0
    reviewed=json.loads(review.read_text())
    assert reviewed['candidate_source_sha256']==manifest['source_sha256']
    assert {q['name']:{'sha256':q['sha256'],'bytes':q['bytes']} for q in reviewed['physical_output_bindings']}==manifest['physical_outputs']
    decoded={}; payload=json.loads((public/'payload-manifest.json').read_text())
    for name,item in payload['payloads'].items():
        packed=(public/item['compressed_file']).read_bytes();assert hashlib.sha256(packed).hexdigest()==item['compressed_sha256']
        raw=gzip.decompress(packed);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        decoded[name]=raw
    assert len(decoded)==9
    frozen={}
    for name,expected in manifest['frozen_physical_source_sha256'].items():
        path=public/'frozen'/name; assert sha(path)==expected; frozen[name]=path.read_bytes()
    assert len(frozen)==7
    packed=(HERE/manifest['baseline_compressed_file']).read_bytes()
    assert hashlib.sha256(packed).hexdigest()==manifest['baseline_compressed_sha256']
    baseline=gzip.decompress(packed)
    assert len(baseline)==manifest['baseline_raw_bytes'] and hashlib.sha256(baseline).hexdigest()==manifest['baseline_raw_sha256']
    build=repository/'build'; build.mkdir(exist_ok=True)
    work=pathlib.Path(tempfile.mkdtemp(prefix='elastic-yarn-normal-block-public-replay-',dir=build))
    # Preserve the owning source's expected relative layout; source text is unchanged.
    pack=work/'build/elastic-yarn-fem-step-preparation'; pack.mkdir(parents=True)
    centroid=work/'build/elastic-yarn-fem-common-step'; centroid.mkdir(parents=True)
    block=work/'build/elastic-yarn-fem-normal-block-step'; block.mkdir(parents=True)
    reference=work/'build/elastic-yarn-common-step-independent-review'; reference.mkdir(parents=True)
    docs=work/'docs/assets'; docs.mkdir(parents=True)
    for name,data in decoded.items():(pack/name).write_bytes(data)
    for name,data in frozen.items():
        target=pack/'frozen'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    shutil.copyfile(HERE/'centroid_step.py',centroid/'step.py')
    shutil.copyfile(HERE/'centroid_step.py',block/'centroid_step.py')
    shutil.copyfile(HERE/'step.py',block/'step.py')
    shutil.copyfile(fixture,docs/'elastic-yarn-fem-frame38-provenance.json')
    (reference/'constitutive-frame38-forces.f64x3.bin').write_bytes(baseline)
    supplied={str(path.relative_to(work)):sha(path) for path in pack.rglob('*') if path.is_file()}
    # NEW provenance receipts, with only real available files. No native binary
    # or historical authoring executable is fabricated or implicitly checked.
    pack_receipt={'schema':'numi.elastic-yarn.public-physical-input-preparation.v1','sha256':supplied,
      'historical_native_checkpoint_identity_claimed':False,'authoring_binary_rebuilt_here':False,
      'inputs_are_reviewed_lossless_public_payloads':True,'source_revision':'69736313414d3afd81f3b181116870143c13966e'}
    (pack/'receipt.json').write_text(json.dumps(pack_receipt,indent=2)+'\n')
    evidence={'schema':'numi.elastic-yarn.public-physical-input-evidence.v1',
      'outputs_sha256':{str(pack/name):hashlib.sha256(raw).hexdigest() for name,raw in decoded.items()},
      'compressed_payloads_verified':9,'frozen_physical_sources_verified':7,
      'provenance_receipt_changed_physical_values_unchanged':True}
    (pack/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    launch={'centroid_source_sha256':sha(centroid/'step.py'),'normal_block_source_sha256':sha(block/'step.py'),
      'reproducer_sha256':sha(pathlib.Path(__file__)),'manifest_sha256':sha(manifest_path),
      'fresh_root':str(work),'new_pack_receipt_sha256':sha(pack/'receipt.json'),'new_pack_evidence_sha256':sha(pack/'evidence.json'),
      'actually_supplied_input_sha256':{str(path.relative_to(work)):sha(path) for path in work.rglob('*') if path.is_file()},
      'public_payload_manifest_sha256':sha(public/'payload-manifest.json'),'historical_native_checkpoint_identity_claimed':False}
    (work/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    centroid_exit=run_source(centroid/'step.py',work)
    centroid_outputs=check_outputs(centroid,manifest['centroid_physical_outputs'])
    cq=json.loads((centroid/'evidence.json').read_text())
    centroid_digest=metric_digest(cq,manifest['centroid_physical_metric_keys'],'independent_constitutive_baseline')
    assert centroid_digest==manifest['centroid_canonical_physical_metrics_sha256']
    assert cq['source_script_sha256']==manifest['centroid_source_sha256'] and cq['exact_replay'] and cq['passed_bounded_controls']
    assert all(sha(pathlib.Path(path))==expected for path,expected in cq['input_sha256'].items())
    block_exit=run_source(block/'step.py',work)
    outputs=check_outputs(block,manifest['physical_outputs'])
    q=json.loads((block/'evidence.json').read_text())
    digest=metric_digest(q,manifest['block_physical_metric_keys'],'baseline')
    assert digest==manifest['block_canonical_physical_metrics_sha256']
    assert q['source_sha256']==manifest['source_sha256'] and q['copied_split_source_sha256']==manifest['centroid_source_sha256']
    assert q['passed_bounded_controls'] and q['exact_replay'] and q['centroid_full_manifold_negative_reproduced']
    assert all(sha(pathlib.Path(path))==expected for path,expected in q['input_sha256'].items())
    scan=q['one_step']['ledger']['final_global_velocity_scan']['minimum']
    assert scan['closing_m_s']>=0 and scan['gap_m']>=-2e-6 and scan['nodes']==[5,1814,1347]
    assert q['one_step']['ledger']['energy_closure_claim'] is False
    rejections=[z['reason'] for z in q['negative_controls']]
    assert rejections==['contact-nonreciprocal','contact-energy-positive','invalid-J','invalid-boundary-area']
    assert all(z['rollback_exact'] and z['unchanged_full_state'] for z in q['negative_controls'])
    receipt={'schema':'numi.elastic-yarn.normal-block.portable-public-reproduction.v1',
      'actual_reproduction_exit':0,'centroid_source_exit':centroid_exit,'normal_block_source_exit':block_exit,
      'fresh_root':str(work),'frozen_centroid_source_sha256':manifest['centroid_source_sha256'],
      'frozen_normal_block_source_sha256':manifest['source_sha256'],
      'owner_centroid_evidence_sha256':manifest['centroid_owner_evidence_sha256'],
      'fresh_centroid_evidence_sha256':sha(centroid/'evidence.json'),
      'owner_normal_block_evidence_sha256':manifest['owner_evidence_sha256'],
      'owner_normal_block_receipt_sha256':manifest['owner_receipt_sha256'],
      'new_input_provenance_receipt':True,'historical_native_checkpoint_identity_claimed':False,
      'decoded_payloads':9,'frozen_physical_sources':7,
      'centroid_runtime_input_hashes_verified':len(cq['input_sha256']),
      'normal_block_runtime_input_hashes_verified':len(q['input_sha256']),
      'centroid_nine_outputs_byte_identical':True,'normal_block_nine_outputs_byte_identical':True,
      'centroid_physical_outputs':centroid_outputs,'normal_block_physical_outputs':outputs,
      'centroid_physical_metric_digest':centroid_digest,'normal_block_physical_metric_digest':digest,
      'retained_independent_review_result_sha256':sha(review),'retained_independent_review_actual_exit':0,
      'exact_replay':q['exact_replay'],'half_steps_accepted':q['half_steps_accepted'],
      'normal_block_mask_controls':[z['passed'] for z in q['block_algebra_controls']['four_active_sets']],
      'full_state_rollback_rejections':rejections,
      'centroid_global_closing_m_s':q['centroid_diagnostic_global_velocity']['minimum']['closing_m_s'],
      'final_global_closing_m_s':scan['closing_m_s'],'final_global_gap_m':scan['gap_m'],
      'contact_dissipation_J':q['one_step']['ledger']['contact_dissipation_J'],
      'physical_energy_balance_residual_J':q['one_step']['ledger']['physical_energy_balance_residual_J'],
      'energy_closure_claim':False,'runtime_s':time.time()-begin,'launch_sha256':sha(work/'launch.json'),
      'centroid_terminal_exit_marker_sha256':sha(centroid/'actual.exit'),
      'normal_block_terminal_exit_marker_sha256':sha(block/'actual.exit'),
      'centroid_terminal_log_sha256':sha(centroid/'run.log'),'normal_block_terminal_log_sha256':sha(block/'run.log')}
    (work/'reproduction-result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'actual_exit':0,'fresh_root':str(work),'centroid_outputs_identical':len(centroid_outputs),
      'normal_block_outputs_identical':len(outputs),'centroid_metric_digest':centroid_digest,
      'normal_block_metric_digest':digest,'final_global_closing_m_s':scan['closing_m_s'],
      'physical_energy_residual_J':receipt['physical_energy_balance_residual_J'],'runtime_s':receipt['runtime_s']},indent=2))
    return 0
if __name__=='__main__':sys.exit(main())
