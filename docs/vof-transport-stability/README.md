# VOF Transport Stability

Von Neumann stability analysis of the `VOFTransport` blend and upwind schemes.

See `report.md` for the full analysis and results.

Reproduce:
```bash
PYTHONPATH=src python3 -m pytest tests/test_vof_transport_stability.py -v
PYTHONPATH=src python3 scripts/verify_vof_stability.py
```
