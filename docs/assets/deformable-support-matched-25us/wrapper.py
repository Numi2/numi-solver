from pathlib import Path
import subprocess,sys,os
p=Path(__file__).resolve().parent
os.setpriority(os.PRIO_PROCESS,0,10)
with (p/'run.log').open('wb') as output:
 r=subprocess.run([sys.executable,str(p/'driver.py')],stdout=output,stderr=subprocess.STDOUT)
(p/'actual.exit').write_text(str(r.returncode)+'\n')
