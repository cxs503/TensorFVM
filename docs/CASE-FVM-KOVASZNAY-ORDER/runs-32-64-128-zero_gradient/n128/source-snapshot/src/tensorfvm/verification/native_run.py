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


#: Files (and directories) this class writes on its own at construction time.
#: They are pure provenance metadata, regenerated on every run, and therefore
#: never evidence in themselves.
METADATA_NAMES = frozenset({'environment.json', 'sources.json', 'source-snapshot'})


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class NativeRunEvidence:
    """Provenance bundle for a native numerical run.

    A run is resumable into its own evidence directory: a directory holding only
    the metadata written by this class (see :data:`METADATA_NAMES`) is reused, so
    an interrupted run -- or a run whose metadata stub was committed to the repo
    before the measurement finished -- can be completed in place.  A directory
    that already holds measured artefacts (``cases.json``, ``*.npz``,
    ``summary.json``, ...) is still refused unless ``overwrite=True`` is passed
    explicitly, so a stale measurement can never be silently clobbered.
    """

    def __init__(self, output, runner, overwrite=False):
        self.output=Path(output);self.repo=Path(runner).resolve().parents[1]
        if self.output.exists():
            present=[path.name for path in self.output.iterdir()]
            measured=sorted(name for name in present if name not in METADATA_NAMES)
            if measured and not overwrite:
                raise ValueError(
                    'evidence output must be empty: %s already holds measured artefacts %s '
                    '(pass overwrite=True to replace them)' % (self.output, measured))
            # Drop only our own metadata; measured artefacts are left in place and
            # are replaced later by the run's explicit writes.
            shutil.rmtree(self.output/'source-snapshot',ignore_errors=True)
            for name in ('environment.json','sources.json'):
                (self.output/name).unlink(missing_ok=True)
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