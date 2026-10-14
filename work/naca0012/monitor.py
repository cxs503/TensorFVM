#!/usr/bin/env python3
"""monitor.py LOG [LOG ...]: progress table from run_naca.py JSON lines (system python3 is enough)."""
import sys, json
for f in sys.argv[1:]:
    print('==', f)
    print('%5s %9s %8s %5s %7s %7s %7s | %8s %8s %8s %8s %8s %4s %6s' % ('it', 'dU', 'mom', 'blend', 'Cl', 'Cd', 'cpmin', 'cf.05', 'cf.1', 'cf.3', 'cf.6', 'cf.9', 'rev', 't[s]'))
    rows = [json.loads(l) for l in open(f) if l.startswith('{')]
    for d in rows[-8:]:
        print('%5d %9.2e %8.2e %5.2f %7.4f %7.4f %7.3f | %8.5f %8.5f %8.5f %8.5f %8.5f %4d %6.0f' % (
            d['it'], d['dU'], d['mom'], d['blend'], d['Cl'], d['Cd'], d['cp_min'], d['cf_up_0p05'], d['cf_up_0p1'],
            d['cf_up_0p3'], d['cf_up_0p6'], d['cf_up_0p9'], d['reversed_upper'], d['elapsed']))
