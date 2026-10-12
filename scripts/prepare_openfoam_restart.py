#!/usr/bin/env python3
"""Copy a completed serial checkpoint into a separate, auditable continuation."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--end',type=int,required=True)
p.add_argument('--write-interval',type=int,default=100)
p.add_argument('--processes',type=int,default=1)
a=p.parse_args();source=a.source.resolve();out=a.output.resolve()
if out.exists():p.error('output must not exist')
times=sorted((int(v.name) for v in source.iterdir() if v.is_dir() and v.name.isdigit() and (v/'U').exists()))
if not times or times[-1]>=a.end:p.error('need an existing field checkpoint before end')
start=times[-1]
if a.write_interval<1 or a.processes<1:p.error('positive interval and process count required')
out.mkdir(parents=True)
for name in dict.fromkeys(['constant','system','0',str(start),'postProcessing']):
    if (source/name).exists():shutil.copytree(source/name,out/name)
for path in source.glob('log.*'):shutil.copy2(path,out/path.name)
shutil.copy2(source/'provenance.json',out/'provenance.json')
if (out/'log.foamRun').exists():(out/'log.foamRun').rename(out/f'log.foamRun.before-{start}')
c=out/'system/controlDict';text=c.read_text()
text=text.replace('startFrom startTime;', 'startFrom latestTime;')
text=re.sub(r'endTime\s+\d+;',f'endTime {a.end};',text)
text=re.sub(r'writeInterval\s+\d+;',lambda m:f'writeInterval {a.write_interval};' if int(re.search(r'\d+',m.group()).group())!=1 else m.group(),text)
c.write_text(text)
(out/'system/decomposeParDict').write_text('FoamFile { version 2; format ascii; class dictionary; object decomposeParDict; }\n'+f'numberOfSubdomains {a.processes};\nmethod scotch;\n')
binding={name:hashlib.sha256((source/str(start)/name).read_bytes()).hexdigest() for name in ['U','p','k','omega','nut','phi']}
(out/'restart-provenance.json').write_text(json.dumps(dict(source=str(source),start=start,end=a.end,processes=a.processes,
    checkpoint_sha256=binding,changed_physical_parameters=False,physical_benchmark_passed=False),indent=2)+'\n')
print(out)
