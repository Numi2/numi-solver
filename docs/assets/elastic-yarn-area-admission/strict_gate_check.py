#!/usr/bin/env python3
"""Expected original failure / corrected pass at exact area-gate equality."""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
which=sys.argv[1]
assert which in ('frozen','candidate')
path=(HERE/'historical/area_admission.py'
      if which=='frozen' else HERE/'area_admission.py')
spec=importlib.util.spec_from_file_location('area_'+which,path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
G=module.GATE
x=((0.,0.,0.),(1.,0.,0.),(0.,2*G,0.))
r=module.passive_velocity(x,((0.,0.,0.),)*3,(1.,1.,1.))
print(json.dumps({'source':which,'face_area_m2':module.represented_area(x),
                  'strict_gate_m2':G,'status':r['status'],'accepted':r['accepted']},sort_keys=True))
assert not r['accepted'], 'equal-to-gate face must be rejected by strict area admission'
