#!/usr/bin/env python3
"""Render frozen independent SST diagnostic fields."""
import argparse
from pathlib import Path
from tensorfvm.verification.sst_channel_report import plot_reference

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    plot_reference(p.parse_args().directory)
