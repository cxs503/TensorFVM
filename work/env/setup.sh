#!/bin/bash
# Idempotent sandbox setup (run after any sandbox reset):
#   bash work/env/setup.sh
# venv at /home/user/.venv-tfvm (system pip is PEP-668 locked; only pypi.org reachable),
# headless gmsh via stub GL/X libraries in /home/user/.stublibs (export LD_LIBRARY_PATH=/home/user/.stublibs).
set -e
cd "$(dirname "$0")/../.."
if [ ! -x /home/user/.venv-tfvm/bin/python ]; then python3 -m venv /home/user/.venv-tfvm; fi
/home/user/.venv-tfvm/bin/python -c "import torch,numpy,scipy,pyamg,gmsh,pytest" 2>/dev/null || \
  /home/user/.venv-tfvm/bin/pip install -q torch numpy scipy pyamg gmsh pytest
/home/user/.venv-tfvm/bin/python -c "import tensorfvm" 2>/dev/null || /home/user/.venv-tfvm/bin/pip install -q -e .
LD_LIBRARY_PATH=/home/user/.stublibs /home/user/.venv-tfvm/bin/python -c "import gmsh" 2>/dev/null || bash work/env/make_stubs.sh
echo "ok: $(/home/user/.venv-tfvm/bin/python -c 'import torch;print(torch.__version__)')  export LD_LIBRARY_PATH=/home/user/.stublibs for gmsh"
