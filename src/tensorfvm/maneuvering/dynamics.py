"""3-DOF (surge-sway-yaw) manoeuvring dynamics, turning-circle and zig-zag metrics.

Equations of motion (body axes ``x`` fwd, ``y`` starboard, ``z`` down)
---------------------------------------------------------------------
::

    (m + m_x) u_dot - (m + m_y) v r = X
    (m + m_y) v_dot + (m + m_x) u r = Y
    (I_z + J_z) r_dot                = N
    psi_dot = r,  x_dot = u cos psi - v sin psi,  y_dot = u sin psi + v cos psi

``m`` is the displacement mass ``rho * Cb L B T``; the added masses come from
slender-body theory over the hull's own lateral planform
(``m_y = int rho pi T_e(x)^2 dx``, ``J_z = int rho pi T_e(x)^2 x^2 dx``,
``m_x = 0.1 m_y``).  Slender body is an **upper bound** for the sway added mass
(a documented over-estimate of roughly a factor two for tankers, see
``docs/pmm-cfd``), and added mass mostly affects the transient, not the steady
turning diameter; the report states this.

Two force-model modes (the "MMG" vs "direct" pair)
-------------------------------------------------
``mode="mmg"``     hull force from derivatives only: ``Y = Y_v v + Y_r r
                   (+ cubic terms if ``include_cubic``)``, rudder force from the
                   polar *slope* (``f_alpha sin alpha_R``).
``mode="direct"``  hull force integrated over the planform at every time step
                   (nonlinear cross-flow + lift) and rudder force from the full
                   polar ``C_L(alpha_R)``.

Both modes share the same physics, the same propeller model and the same
integration scheme, so the difference between them measures the *model-form*
error of the derivative formulation -- which is exactly what the CFD/MMG
arbitration asks for.

Nothing in this module claims experimental agreement: the metric extraction is
verified against analytic trajectories (a synthetic circle and a synthetic
zig-zag), and the predicted numbers are compared with a literature *band* in the
report with the band's provenance flag attached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hull import (
    Kvlcc2Conditions, DEFAULT_CONDITIONS, RudderGeometry, rudder_geometry,
    section_data,
)
from .hullforces import HullForceModelSet, linear_derivatives
from .propeller import PropellerGeometry, open_water_point
from .rudder import (
    DEFAULT_INTERACTION, RudderInteraction, RudderPolar, analytic_rudder_polar,
    rudder_normal_force,
)
from .selfpropulsion import SelfPropulsionPoint

__all__ = [
    "ManoeuvreModel", "operating_point_for_rudder", "Trajectory", "simulate_turning_circle", "simulate_zigzag",
    "turning_circle_metrics", "zigzag_metrics", "added_masses",
]


def added_masses(model_hull: HullForceModelSet, n: int = 121) -> Dict[str, float]:
    """Slender-body added mass/inertia of the hull (documented upper bound)."""
    c = model_hull.conditions
    data = section_data(model_hull.hull, n)
    x = data["x"] - 0.5 * c.length
    m22 = c.rho * math.pi * data["draft"] ** 2
    m_y = float(np.trapezoid(m22, data["x"]))
    j_z = float(np.trapezoid(m22 * x * x, data["x"]))
    return {"m_x": 0.1 * m_y, "m_y": m_y, "J_z": j_z,
            "m_y_over_mass": m_y / c.mass, "J_z_over_mass_L2": j_z / (c.mass * c.length ** 2)}


#: steady-turn drift/yaw-rate of the direct model at 35 deg rudder (prime units),
#: used to scale the operating point of the effective-derivative (MMG) model
STEADY_TURN_REFERENCE = (-0.130, 0.220)


def operating_point_for_rudder(delta_deg: float) -> Tuple[float, float]:
    """Operating point ``(v', r')`` of the effective-derivative model.

    The steady turn of a ship scale almost linearly with rudder angle at moderate
    angles, so the reference steady state found at 35 deg (``v' = -0.13``,
    ``r' = 0.22`` from the direct solution) is scaled by ``delta/35``.  This is the
    MMG convention in spirit: derivatives are *effective* quantities measured at
    the amplitude of the manoeuvre being predicted, not at zero drift.  The
    dependence on the manoeuvre is disclosed as a limitation.
    """
    scale = min(max(abs(delta_deg) / 35.0, 0.0), 1.5)
    return (STEADY_TURN_REFERENCE[0] * scale, STEADY_TURN_REFERENCE[1] * scale)


@dataclass
class ManoeuvreModel:
    """Everything needed to integrate a manoeuvre (model scale, SI units)."""

    conditions: Kvlcc2Conditions = field(default_factory=lambda: DEFAULT_CONDITIONS)
    hull_model: HullForceModelSet = field(default_factory=HullForceModelSet)
    propeller: PropellerGeometry = field(default_factory=PropellerGeometry)
    self_propulsion: Optional[SelfPropulsionPoint] = None
    polar: Optional[RudderPolar] = None
    rudder: RudderGeometry = field(default_factory=rudder_geometry)
    interaction: RudderInteraction = DEFAULT_INTERACTION
    mode: str = "direct"
    #: operating point of the Taylor (MMG-style) derivative model, prime units;
    #: ``None`` means "derive from :func:`operating_point_for_rudder`"
    mmg_operating_point: Optional[Tuple[float, float]] = None
    rudder_operating_deg: float = 35.0
    include_cubic: bool = False
    ct_table_size: int = 21
    #: radius of gyration of the ship about z as a fraction of L
    k_zz_ratio: float = 0.25
    _kt_table: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None,
                                                               repr=False)
    _derivatives: Optional[Dict[str, float]] = field(default=None, repr=False)
    _cubic: Optional[Dict[str, float]] = field(default=None, repr=False)
    _masses: Optional[Dict[str, float]] = field(default=None, repr=False)

    def __post_init__(self):
        if self.mode not in ("mmg", "direct"):
            raise ValueError("mode must be 'mmg' or 'direct'")
        if self.mmg_operating_point is None and self.mode == "mmg":
            self.mmg_operating_point = operating_point_for_rudder(
                self.rudder_operating_deg)
        if self.polar is None:
            self.polar = analytic_rudder_polar(self.rudder)
        if self.self_propulsion is None:
            raise ValueError("a self-propulsion point is required")

    # -- cached tables ----------------------------------------------------
    @property
    def masses(self) -> Dict[str, float]:
        if self._masses is None:
            self._masses = added_masses(self.hull_model)
        return self._masses

    @property
    def derivatives(self) -> Dict[str, float]:
        if self._derivatives is None:
            self._derivatives = linear_derivatives(
                self.hull_model, *self.mmg_operating_point)
        return self._derivatives

    @property
    def cubic(self) -> Dict[str, float]:
        """Cubic cross-flow derivative fit ``Y_vvv``, ``N_vvv``, ``Y_vrr``, ``N_vrr``."""
        if self._cubic is None:
            self._cubic = self._fit_cubic()
        return self._cubic

    @property
    def kt_table(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._kt_table is None:
            js = np.linspace(0.01, 1.0, self.ct_table_size)
            kts = np.array([open_water_point(self.propeller, float(j),
                                             rho=self.conditions.rho).KT for j in js])
            self._kt_table = (js, kts)
        return self._kt_table

    def _fit_cubic(self) -> Dict[str, float]:
        """Polynomial (MMG-style) cross-flow derivatives of the direct hull model.

        A six-term cubic expansion ``Y = Y_v v + Y_r r + Y_vvv v^3 + Y_vvr v^2 r +
        Y_vrr v r^2 + Y_rrr r^3`` (same for ``N``) is fitted by least squares to
        the direct force model over the envelope a 35 deg turn actually visits
        (``|v'| <= 0.25``, ``|r'| <= 0.40``).  The fit residual is stored, so the
        arbitration can state how well the derivative model represents the force
        model before any manoeuvre is run.
        """
        c = self.conditions
        q = 0.5 * c.rho * c.u_design ** 2 * c.length * c.draft
        qn = q * c.length
        d = self.derivatives
        rows, y_target, n_target = [], [], []
        for v_prime in (-0.25, -0.15, -0.075, 0.075, 0.15, 0.25):
            for r_prime in (-0.40, -0.20, 0.0, 0.20, 0.40):
                res = self.hull_model.force(c.u_design, v_prime * c.u_design,
                                            r_prime * c.u_design / c.length,
                                            n_stations=81)
                y_lin = q * (d["Y_v"] * v_prime + d["Y_r"] * r_prime)
                n_lin = qn * (d["N_v"] * v_prime + d["N_r"] * r_prime)
                rows.append([v_prime ** 3, v_prime ** 2 * r_prime,
                             v_prime * r_prime ** 2, r_prime ** 3])
                y_target.append((res["Y"] - y_lin) / q)
                n_target.append((res["N"] - n_lin) / qn)
        A = np.asarray(rows)
        y_cf = np.linalg.lstsq(A, np.asarray(y_target), rcond=None)[0]
        n_cf = np.linalg.lstsq(A, np.asarray(n_target), rcond=None)[0]
        y_res = float(np.max(np.abs(A @ y_cf - np.asarray(y_target))))
        n_res = float(np.max(np.abs(A @ n_cf - np.asarray(n_target))))
        return {"Y_vvv": float(y_cf[0]), "Y_vvr": float(y_cf[1]),
                "Y_vrr": float(y_cf[2]), "Y_rrr": float(y_cf[3]),
                "N_vvv": float(n_cf[0]), "N_vvr": float(n_cf[1]),
                "N_vrr": float(n_cf[2]), "N_rrr": float(n_cf[3]),
                "fit_residual_Y_prime": y_res, "fit_residual_N_prime": n_res}

    # -- forces -----------------------------------------------------------
    def forces(self, u: float, v: float, r: float, delta_rad: float
               ) -> Tuple[float, float, float]:
        c = self.conditions
        if self.mode == "direct":
            hull = self.hull_model.force(u, v, r)
            x_h, y_h, n_h = hull["X"], hull["Y"], hull["N"]
            linear_rudder = False
        else:
            d = self.derivatives
            speed = max(math.hypot(u, v), 1e-6)
            vp = v / speed
            rp = r * c.length / speed
            q_inst = 0.5 * c.rho * speed * speed * c.length * c.draft
            y_h = q_inst * (d["Y_0"] + d["Y_v"] * vp + d["Y_r"] * rp)
            n_h = q_inst * c.length * (d["N_0"] + d["N_v"] * vp + d["N_r"] * rp)
            if self.include_cubic:
                cu = self.cubic
                y_h += q_inst * (cu["Y_vvv"] * vp ** 3 + cu["Y_vvr"] * vp ** 2 * rp
                                 + cu["Y_vrr"] * vp * rp ** 2 + cu["Y_rrr"] * rp ** 3)
                n_h += q_inst * c.length * (
                    cu["N_vvv"] * vp ** 3 + cu["N_vvr"] * vp ** 2 * rp
                    + cu["N_vrr"] * vp * rp ** 2 + cu["N_rrr"] * rp ** 3)
            x_h = -self.hull_model.resistance(u)
            linear_rudder = True
        kt_js, kt_values = self.kt_table
        n_rps = self.self_propulsion.rps
        j = (1.0 - self.self_propulsion.wake_fraction) * max(u, 1e-9) / (
            n_rps * self.propeller.diameter)
        kt = float(np.interp(j, kt_js, kt_values))
        thrust = c.rho * n_rps ** 2 * self.propeller.diameter ** 4 * kt
        rud = rudder_normal_force(delta_rad, u, v, r, polar=self.polar,
                                  geometry=self.rudder, interaction=self.interaction,
                                  KT=max(kt, 0.0), J=max(j, 1e-6),
                                  conditions=c, linear=linear_rudder)
        x = x_h + (1.0 - self.self_propulsion.thrust_deduction) * thrust + rud["F_x"]
        y = y_h + rud["F_y"]
        n = n_h + rud["M_z"]
        return x, y, n

    # -- state derivative -------------------------------------------------
    def state_derivative(self, state: Sequence[float], delta_rad: float
                         ) -> np.ndarray:
        u, v, r, x, y, psi = state
        c = self.conditions
        m = c.mass
        ms = self.masses
        x_force, y_force, n_moment = self.forces(u, v, r, delta_rad)
        u_dot = (x_force + (m + ms["m_y"]) * v * r) / (m + ms["m_x"])
        v_dot = (y_force - (m + ms["m_x"]) * u * r) / (m + ms["m_y"])
        i_z = m * (self.k_zz_ratio * c.length) ** 2
        r_dot = n_moment / (i_z + ms["J_z"])
        return np.array([u_dot, v_dot, r_dot,
                         u * math.cos(psi) - v * math.sin(psi),
                         u * math.sin(psi) + v * math.cos(psi), r])


@dataclass
class Trajectory:
    """Time history of a manoeuvre."""

    time: np.ndarray
    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    delta: np.ndarray
    dt: float
    mode: str
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return float(self.meta.get("length", 7.0))

    def as_arrays(self) -> Dict[str, np.ndarray]:
        return {"time": self.time, "u": self.u, "v": self.v, "r": self.r,
                "x": self.x, "y": self.y, "psi": self.psi, "delta": self.delta}


def _integrate(model: ManoeuvreModel, controller: Callable[[float, np.ndarray], float],
               t_end: float, dt: float, state0: Sequence[float],
               stop: Optional[Callable[[float, np.ndarray], bool]] = None
               ) -> Trajectory:
    """Classical RK4 with a rudder-angle controller (rudder rate limited)."""
    state = np.array(state0, dtype=float)
    t = 0.0
    rows = []
    diverged = False
    delta = controller(0.0, state)
    max_rate = math.radians(2.33)  # 2.33 deg/s model rudder rate (5 deg/s full scale)
    n_steps = int(round(t_end / dt))
    for step in range(n_steps + 1):
        rows.append([t, *state, delta])
        if stop is not None and stop(float(t), state):
            break
        target = controller(t, state)
        step_rate = max_rate * dt
        delta = delta + float(np.clip(target - delta, -step_rate, step_rate))
        k1 = model.state_derivative(state, delta)
        k2 = model.state_derivative(state + 0.5 * dt * k1, delta)
        k3 = model.state_derivative(state + 0.5 * dt * k2, delta)
        k4 = model.state_derivative(state + dt * k3, delta)
        state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = (step + 1) * dt
        if (not np.all(np.isfinite(state))) or abs(state[2]) > 5.0 \
                or abs(state[1]) > 10.0 or abs(state[0]) > 10.0:
            diverged = True
            break
    data = np.array(rows)
    traj = Trajectory(time=data[:, 0], u=data[:, 1], v=data[:, 2], r=data[:, 3],
                      x=data[:, 4], y=data[:, 5], psi=data[:, 6], delta=data[:, 7],
                      dt=dt, mode=model.mode,
                      meta={"length": model.conditions.length,
                            "rudder_rate_deg_s": math.degrees(max_rate)})
    if diverged:
        traj.meta["diverged"] = True
    return traj


def simulate_turning_circle(model: ManoeuvreModel, delta_deg: float = 35.0,
                           dt: float = 0.25, t_end: Optional[float] = None,
                           heading_limit_deg: float = 400.0) -> Trajectory:
    """Full-ahead turning circle with a constant rudder angle."""
    delta = math.radians(delta_deg)
    if t_end is None:
        t_end = 12.0 * model.conditions.length / model.conditions.u_design
    u0 = model.self_propulsion.speed if model.self_propulsion else model.conditions.u_design
    state0 = (u0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def controller(_t: float, _state: np.ndarray) -> float:
        return delta

    def stop(_t: float, state: np.ndarray) -> bool:
        return state[5] >= math.radians(heading_limit_deg)

    traj = _integrate(model, controller, t_end, dt, state0, stop)
    traj.meta.update({"delta_deg": delta_deg, "u0": u0})
    return traj


def simulate_zigzag(model: ManoeuvreModel, psi_a_deg: float = 10.0,
                    delta_deg: float = 10.0, dt: float = 0.25,
                    cycles: int = 2, t_end: Optional[float] = None) -> Trajectory:
    """``psi_a / delta`` zig-zag (default 10/10), heading- and rate-limited."""
    psi_a = math.radians(psi_a_deg)
    delta = math.radians(delta_deg)
    if t_end is None:
        t_end = 20.0 * model.conditions.length / model.conditions.u_design
    u0 = model.self_propulsion.speed if model.self_propulsion else model.conditions.u_design
    state0 = (u0, 0.0, 0.0, 0.0, 0.0, 0.0)
    history: List[float] = []

    def controller(_t: float, state: np.ndarray) -> float:
        psi = state[5]
        # count completed half-cycles by tracking sign changes of the command
        if not history:
            history.append(delta)
            return delta
        current = history[-1]
        if current > 0.0 and psi >= psi_a:
            current = -delta
            history.append(current)
        elif current < 0.0 and psi <= -psi_a:
            current = delta
            history.append(current)
        return current

    def stop(_t: float, _state: np.ndarray) -> bool:
        return len(history) > cycles * 2

    traj = _integrate(model, controller, t_end, dt, state0, stop)
    traj.meta.update({"psi_a_deg": psi_a_deg, "delta_deg": delta_deg,
                      "reversals": len(history) - 1, "u0": u0})
    return traj


# --------------------------------------------------------------------------- #
# metric extraction
# --------------------------------------------------------------------------- #
def _first_crossing(t: np.ndarray, psi: np.ndarray, target: float) -> Optional[float]:
    """Interpolated time at which ``psi`` first reaches ``target`` (radians)."""
    for i in range(1, len(psi)):
        if psi[i - 1] < target <= psi[i]:
            w = (target - psi[i - 1]) / (psi[i] - psi[i - 1])
            return float(t[i - 1] + w * (t[i] - t[i - 1]))
    return None


def _interp_state(traj: Trajectory, time: float) -> Dict[str, float]:
    return {"x": float(np.interp(time, traj.time, traj.x)),
            "y": float(np.interp(time, traj.time, traj.y)),
            "psi": float(np.interp(time, traj.time, traj.psi)),
            "u": float(np.interp(time, traj.time, traj.u)),
            "v": float(np.interp(time, traj.time, traj.v)),
            "r": float(np.interp(time, traj.time, traj.r))}


def turning_circle_metrics(traj: Trajectory) -> Dict[str, float]:
    """Advance, transfer, tactical diameter and steady drift angle.

    Advance ``A_D`` and transfer ``A_T`` are the longitudinal/lateral
    displacements (in the initial-heading frame) at a heading change of 90 deg.
    Tactical diameter ``T_D`` is the **maximum transfer**, i.e. the largest lateral
    displacement of the path measured perpendicular to the original course (for a
    steady circle this is the circle diameter ``2 R``).  ``steady_turning_radius``
    is reported independently as ``U/r`` in the steady phase, so the two routes can
    be cross-checked.
    """
    if traj.meta.get("diverged"):
        raise ValueError("the trajectory left the model envelope (diverged) before "
                         "the turning circle closed; the derivative model is not "
                         "valid at this rudder angle")
    L = traj.length
    t90 = _first_crossing(traj.time, traj.psi, math.radians(90.0))
    t180 = _first_crossing(traj.time, traj.psi, math.radians(180.0))
    t270 = _first_crossing(traj.time, traj.psi, math.radians(270.0))
    if t90 is None or t180 is None or t270 is None:
        raise ValueError("trajectory did not reach 270 deg of heading change; "
                         "extend t_end")
    s90 = _interp_state(traj, t90)
    s270 = _interp_state(traj, t270)
    s180 = _interp_state(traj, t180)
    steady = slice(max(0, len(traj.time) // 2), len(traj.time))
    u_s = float(np.mean(traj.u[steady]))
    v_s = float(np.mean(traj.v[steady]))
    r_s = float(np.mean(traj.r[steady]))
    drift = math.degrees(math.atan2(-v_s, u_s))
    y_max = float(np.max(traj.y))
    return {
        "advance_over_L": s90["x"] / L,
        "transfer_over_L": s90["y"] / L,
        "tactical_diameter_over_L": (y_max - traj.y[0]) / L,
        "tactical_diameter_m": y_max - traj.y[0],
        "tactical_diameter_2R_over_L": 2.0 * (float(np.mean(traj.u[steady]))
                                              / float(np.mean(traj.r[steady]))) / L
        if abs(float(np.mean(traj.r[steady]))) > 1e-12 else float("inf"),
        "advance_m": s90["x"],
        "transfer_m": s90["y"],
        "x_180_over_L": s180["x"] / L,
        "y_180_over_L": s180["y"] / L,
        "time_90_s": t90, "time_180_s": t180, "time_270_s": t270,
        "steady_r_1_s": r_s,
        "steady_drift_deg": drift,
        "steady_speed_m_s": u_s,
        "steady_turning_radius_over_L": (u_s / r_s) / L if abs(r_s) > 1e-12 else float("inf"),
        "delta_deg": traj.meta.get("delta_deg", float("nan")),
        "mode": traj.mode,
    }


def zigzag_metrics(traj: Trajectory, psi_a_deg: float = 10.0) -> Dict[str, float]:
    """First/second overshoot angles and the transient lateral shift.

    ``psi_1``  first overshoot beyond the commanded heading
    ``psi_2``  second overshoot (after the first rudder reversal)
    ``A_TE``   lateral (transverse) displacement of the ship at the first and
               second overshoot, measured from the initial track
    """
    psi_a = math.radians(psi_a_deg)
    psi = traj.psi
    # segment the record at the extremes that follow each reversal
    extrema = []
    for i in range(1, len(psi) - 1):
        if (psi[i] >= psi[i - 1] and psi[i] > psi[i + 1]) or \
                (psi[i] <= psi[i - 1] and psi[i] < psi[i + 1]):
            extrema.append(i)
    overshoots = [abs(psi[i]) - psi_a for i in extrema]
    overshoots = [math.degrees(o) for o in overshoots if o > -1e-3]
    out = {"psi_a_deg": psi_a_deg, "mode": traj.mode,
           "overshoot_deg": overshoots}
    out["psi_1_deg"] = overshoots[0] if len(overshoots) > 0 else float("nan")
    out["psi_2_deg"] = overshoots[1] if len(overshoots) > 1 else float("nan")
    if extrema:
        i1 = extrema[0]
        out["A_TE_1_over_L"] = traj.y[i1] / traj.length
        out["t_1_s"] = float(traj.time[i1])
    if len(extrema) > 1:
        i2 = extrema[1]
        out["A_TE_2_over_L"] = traj.y[i2] / traj.length
        out["t_2_s"] = float(traj.time[i2])
    out["max_psi_deg"] = math.degrees(float(np.max(np.abs(psi))))
    out["delta_deg"] = traj.meta.get("delta_deg", float("nan"))
    return out