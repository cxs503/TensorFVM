"""Small reusable evidence helpers for native numerical verification runners."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import numpy as np
import scipy


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class NativeRunEvidence:
    def __init__(self, output, runner):
        self.output=Path(output);self.repo=Path(runner).resolve().parents[1]
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError('evidence output must be empty')
        self.output.mkdir(parents=True,exist_ok=True)
        sources={Path(runner).resolve()} | {Path(m.__file__).resolve() for name,m in list(sys.modules.items())
            if name.startswith('tensorfvm') and getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'}
        self.hashes={}
        for source in sorted(sources):
            relative=source.relative_to(self.repo);target=self.output/'source-snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            self.hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
        write_json(self.output/'sources.json',self.hashes)
        write_json(self.output/'environment.json',dict(python=sys.version,executable=sys.executable,
            numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),device='CPU; GPU untested',
            threads={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}))

    def verify_sources(self):
        for relative,expected in self.hashes.items():
            if hashlib.sha256((self.repo/relative).read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Source changed during run: {relative}')
