"""Compatibility CLI for the shared physical-mesh report publisher."""
import argparse
from pathlib import Path
from ..external_mesh_report import publish


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("outputs",type=Path,nargs="+")
    args=parser.parse_args()
    for output in args.outputs:publish(output)
