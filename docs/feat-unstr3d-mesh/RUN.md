# Reproducing the FEAT-UNSTR3D-MESH gates

```bash
export PYTHONPATH=src
python -m pytest tests/test_unstr3d_msh_import.py tests/test_unstr3d_benchmark_cases.py -q
python -m pytest tests/test_unstr3d_poisson_order.py -q      # ~60 s local
python -m pytest tests/test_unstr3d_convection_45deg.py -q   # ~2 min local
python -c "from tensorfvm.unstr3d_benchmark import sql_statements; print('\n'.join(sql_statements()))"
```

Heavy ladders (`n = 16`, 24576 tets) must be submitted through the compute
queue, not run on the control plane.
