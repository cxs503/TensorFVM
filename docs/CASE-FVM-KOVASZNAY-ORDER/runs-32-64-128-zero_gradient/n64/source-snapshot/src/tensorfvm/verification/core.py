"""Solver-independent acceptance, raw output, provenance and refinement."""
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import numpy as np
import torch

ERROR_LIMIT = 0.03


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    limit: float
    description: str
    unit: str = "1"

    def record(self):
        return dict(name=self.name, value=self.value, limit=self.limit,
                    description=self.description, unit=self.unit,
                    passed=bool(math.isfinite(self.value) and self.value < self.limit))


def relative_error(actual, reference, norm="l2"):
    a, r = np.asarray(actual), np.asarray(reference)
    if a.shape != r.shape or not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("error comparison requires finite arrays with identical shape")
    denominator = np.linalg.norm(r.ravel()) if norm == "l2" else np.max(np.abs(r))
    numerator = np.linalg.norm((a-r).ravel()) if norm == "l2" else np.max(np.abs(a-r))
    if denominator <= 0:
        raise ValueError("zero reference requires an absolute, explicitly scaled metric")
    return float(numerator/denominator)


def evaluate(metrics):
    rows = [m.record() for m in metrics]
    if not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique metrics required")
    return rows, all(r["passed"] for r in rows)


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[3]
    paths = list((root/"src/tensorfvm/verification").glob("*.py"))
    paths += [root/"pyproject.toml", root/"src/tensorfvm/__init__.py"]
    paths += [root/"src/tensorfvm"/name for name in
              ("solver.py", "backend_registry.py", "body_fitted.py", "mesh_api.py",
               "periodic_mac.py", "runtime.py")]
    # Scope deliberately includes every imported numerical parent, not plotting artifacts.
    import matplotlib
    return dict(matplotlib=matplotlib.__version__, git_base=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                python=platform.python_version(), torch=torch.__version__,
                numpy=np.__version__, platform=platform.platform(),
                torch_threads=torch.get_num_threads(), device="cpu",
                source_sha256={str(p.relative_to(root)):sha(p) for p in sorted(paths)})


def save_run(directory, result):
    directory.mkdir(parents=True, exist_ok=True)
    fields = result.pop("fields")
    np.savez_compressed(directory/"fields.npz", **fields)
    history = result.pop("history")
    write_json(directory/"history.json", history)
    if history:
        # Scalar diagnostics only; structured operator records stay in history.json.
        keys = [k for k,v in history[0].items() if isinstance(v,(int,float,bool))]
        with (directory/"history.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=keys,extrasaction="ignore")
            writer.writeheader();writer.writerows(history)
    write_json(directory/"result.json",result)
    return result


def metric_value(row, name):
    return next(m["value"] for m in row["metrics"] if m["name"]==name)


def observed_orders(rows, metric):
    values = [metric_value(r,metric) for r in rows]
    return [math.log(a/b)/math.log(rows[i]["spacing_m"]/rows[i+1]["spacing_m"])
            for i,(a,b) in enumerate(zip(values,values[1:]))]


def artifact_manifest(directory):
    return {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json",)}
