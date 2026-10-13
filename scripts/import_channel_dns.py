#!/usr/bin/env python3
"""Transcribe downloaded original DNS files, verifying headers and matching grids."""
import argparse
from tensorfvm.verification.channel_dns import import_profiles
p=argparse.ArgumentParser(description=__doc__)
for name in ('mean','stdev','output','case'):p.add_argument('--'+name,required=True)
a=p.parse_args();m=import_profiles(a.mean,a.stdev,a.output,case=a.case)
print(dict(Re_tau=m['Re_tau_reported'],Cf=m['Cf'],rows=m['files']['mean']['rows']))
