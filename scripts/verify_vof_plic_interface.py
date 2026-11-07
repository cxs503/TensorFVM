#!/usr/bin/env python3
"""Run the PLIC interface-transport verification surface.

Thin wrapper around :mod:`tensorfvm.verification.vof_plic_interface` so the
artefacts quoted in ``docs/m4956-vof-plic-interface/README.md`` can be
regenerated with one command::

    python3 scripts/verify_vof_plic_interface.py            # full study
    python3 scripts/verify_vof_plic_interface.py --quick    # small grids
"""

import sys

from tensorfvm.verification.vof_plic_interface import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))