"""Start the matched L3/L4 read-only audit after real L3 termination."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

root = Path('/Users/n/numi-solver-mesh4-verlet-20260929')
directory = root/'build/mesh4-L3-terminal-spatial-audit-20260929'
assert (root/'build/mesh4-level3-dt25us-match.exit').read_text().strip() == '0'
for pid in (87159,86349):
    assert subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True).returncode != 0,pid
assert not (directory/'actual.exit').exists()
for name in ('driver.py','wrapper.py','audit_deformable_mesh_energy.py',
             'audit_deformable_mesh_trajectory.py','audit_deformable_mesh_refinement.py'):
    assert (directory/name).is_file(),name
run = subprocess.Popen([sys.executable,str(directory/'wrapper.py')],cwd=root,
    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
    start_new_session=True)
launch = {'wrapper_pid':run.pid,'status':'LAUNCHED_CPU_ONLY','python':sys.executable,
    'native_L3_actual_exit':0,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.py')},
    'matched_L3_L4_spatial_target_m':.001,'no_new_GPU_work':True,'launched_epoch_s':time.time()}
(directory/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
print(json.dumps({'wrapper_pid':run.pid,'launch':str(directory/'launch.json')}))
