#!/usr/bin/env python3
"""Record the provenance of every frozen NASA-TMR reference artefact.

The validation case definitions, the experimental/theoretical reference data and
the official nested PLOT3D grids are downloaded verbatim from the Turbulence
Modeling Resource.  This script writes
``validation_tmr/reference/tmr/provenance.json`` with the upstream URL, the
SHA-256 digest, the byte size and the retrieval timestamp of every file, so a
reviewer can re-fetch the identical bytes and re-run the cases.

Upstream note: ``turbmodels.larc.nasa.gov`` now redirects to the NASA TMR landing
page.  The case pages and reference data are served unchanged from the TMR
working-group mirror ``https://tmbwg.github.io/turbmodels`` (the same mirror the
repository already cites for the SST model description); the grid archives are
served from the NASA upload endpoints linked by those pages.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "validation_tmr/reference/tmr"

MIRROR = "https://tmbwg.github.io/turbmodels/"
NASA = "https://www.nasa.gov/wp-content/uploads/2026/02/"
SOURCES = {
    "page_flatplate_val.html": (MIRROR + "flatplate_val.html",
                                "2DZP flat plate validation case definition"),
    "page_flatplate_grids.html": (MIRROR + "flatplate_grids.html",
                                  "2DZP official nested grid family description"),
    "page_backstep_val.html": (MIRROR + "backstep_val.html",
                               "2DBFS backward-facing-step case definition"),
    "page_backstep_grids.html": (MIRROR + "backstep_grids.html",
                                 "2DBFS official nested grid family description"),
    "page_spalart.html": (MIRROR + "spalart.html",
                          "Spalart-Allmaras model definition used by the solver"),
    "FlatPlate_validation_cf_K-S.dat": (
        MIRROR + "FlatPlate_validation/cf_K-S.dat",
        "Cf(Re_theta) from the Karman-Schoenherr relation"),
    "FlatPlate_validation_u+y+.dat": (
        MIRROR + "FlatPlate_validation/u+y+.dat",
        "u+ vs y+ from Coles law of the wall with van Driest damping"),
    "FlatPlate_validation_cf_as_function_of_x.dat": (
        MIRROR + "FlatPlate_validation/cf_as_function_of_x.dat",
        "typical model Cf(x) variation, kept for context only"),
    "FlatPlate_validation_retheta_variation_typical.dat": (
        MIRROR + "FlatPlate_validation/retheta_variation_typical.dat",
        "typical Re_theta(x), used to confirm the sampling window"),
    "Backstep_validation_cf.exp.dat": (
        MIRROR + "Backstep_validation/cf.exp.dat",
        "Driver & Seegmiller bottom-wall skin friction (TMR distributed)"),
    "Backstep_validation_cp.expnew.dat": (
        MIRROR + "Backstep_validation/cp.expnew.dat",
        "Driver & Seegmiller wall pressure coefficient, shifted to Cp=0 at x/H=40"),
    "Backstep_validation_profiles.exp.dat": (
        MIRROR + "Backstep_validation/profiles.exp.dat",
        "Driver & Seegmiller velocity/turbulence profiles at x/H = -4, 1, 4, 6, 10"),
    "grids/flatplate_clust2_4levelsdown_35x25.p2dfmt": (
        NASA + "flatplategrids-grids.zip",
        "official TMR flat plate grid 35x25 (4 levels down)"),
    "grids/flatplate_clust2_3levelsdown_69x49.p2dfmt": (
        NASA + "flatplategrids-grids.zip",
        "official TMR flat plate grid 69x49 (3 levels down)"),
    "grids/flatplate_clust2_2levelsdown_137x97.p2dfmt": (
        NASA + "flatplategrids-grids.zip",
        "official TMR flat plate grid 137x97 (2 levels down)"),
    "grids/backstep5_4levdn.p2dfmt": (
        NASA + "backstep-grids.zip",
        "official TMR 2DBFS grid 33x33/13x33/49x57/17x57 (4 levels down)"),
    "grids/backstep5_3levdn.p2dfmt": (
        NASA + "backstep-grids.zip",
        "official TMR 2DBFS grid 65x65/25x65/97x113/33x113 (3 levels down)"),
    "grids/backstep5_2levdn.p2dfmt": (
        NASA + "backstep-grids.zip",
        "official TMR 2DBFS grid 129x129/49x129/193x225/65x225 (2 levels down)"),
}
CITATIONS = [
    dict(kind="case-definition",
         title="NASA Turbulence Modeling Resource: 2D Zero Pressure Gradient Flat "
               "Plate Validation Case",
         url="https://tmbwg.github.io/turbmodels/flatplate_val.html"),
    dict(kind="case-definition",
         title="NASA Turbulence Modeling Resource: 2D Backward Facing Step "
               "Validation Case",
         url="https://tmbwg.github.io/turbmodels/backstep_val.html"),
    dict(kind="experiment", title="Driver, D. M. and Seegmiller, H. L., "
         "\"Features of Reattaching Turbulent Shear Layer in Divergent Channel "
         "Flow\", AIAA Journal 23(2):163-171, 1985",
         doi="10.2514/3.8890"),
    dict(kind="theory", title="Schoenherr, K. E., Trans. SNAME 40:279-313, 1932 "
         "(Karman-Schoenherr skin-friction relation, as tabulated by TMR)",
         url="https://tmbwg.github.io/turbmodels/flatplate_val.html"),
    dict(kind="theory", title="Coles, D., \"The law of the wake in the turbulent "
         "boundary layer\", J. Fluid Mech. 1(2):191-226, 1956",
         doi="10.1017/S0022112056000135"),
    dict(kind="model", title="Spalart, P. R. and Allmaras, S. R., \"A one-equation "
         "turbulence model for aerodynamic flows\", 1992 (as specified by TMR)",
         url="https://tmbwg.github.io/turbmodels/spalart.html"),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    payload = dict(
        name="NASA Turbulence Modeling Resource reference artefacts",
        mirror_root=MIRROR,
        retrieval=dict(timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       tool="curl -sSL", network="https://tmbwg.github.io, "
                       "https://www.nasa.gov"),
        files={}, citations=CITATIONS)
    missing = []
    for name, (url, description) in SOURCES.items():
        path = REFERENCE / name
        if not path.exists():
            missing.append(name)
            continue
        payload["files"][name] = dict(url=url, description=description,
                                      bytes=path.stat().st_size,
                                      sha256=digest(path))
    (REFERENCE / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"recorded {len(payload['files'])} files; missing {missing}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())