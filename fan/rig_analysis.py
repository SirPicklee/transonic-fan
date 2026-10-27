"""
Processing of the NASA Rotor 67 rig survey data.

From the radial surveys of total pressure, total temperature, static
pressure and flow angle at the rotor inlet (station 1) and exit (station 2):

  1. local Mach number, velocity and density from p0, ps and T0,
  2. mass flow through each annular element of table IV,
  3. per-streamline total pressure ratio, total temperature ratio and
     efficiency across the rotor,
  4. mass-averaged temperature ratio and energy-averaged pressure ratio,
     which together give the rotor adiabatic efficiency.

The published values (tables III and V) are the check: if the chain
reproduces them, it can be trusted for the comparison with the design model.
"""

import numpy as np

from fan.rig_data import POINTS, load_overall, load_streamlines, load_survey
from fan.velocity_triangles import GAMMA, R_AIR

_K = (GAMMA - 1.0) / GAMMA
R_HUB2, R_SHROUD2 = 0.120167, 0.244043     # station 2 annulus [m]


def flow_state(p0, t0, ps):
    """Mach number, velocity [m/s] and density [kg/m^3] from total and static
    pressure and total temperature (isentropic relations)."""
    mach = np.sqrt(2.0 / (GAMMA - 1.0) * ((p0 / ps) ** _K - 1.0))
    t = t0 / (1.0 + 0.5 * (GAMMA - 1.0) * mach ** 2)
    v = mach * np.sqrt(GAMMA * R_AIR * t)
    rho = ps / (R_AIR * t)
    return mach, v, rho


def _at_radii(survey: dict, radii: np.ndarray) -> dict:
    """Survey values at the given radii (each must be a measured radius)."""
    idx = [int(np.argmin(np.abs(survey["radius"] - r))) for r in radii]
    return {k: v[idx] for k, v in survey.items()}


def element_mass_flow(survey: dict, area: np.ndarray, slope_deg: np.ndarray) -> np.ndarray:
    """Mass flow [kg/s] through each annular element, rho * V_axial * dA, with
    V_axial = V cos(flow angle from meridional) cos(streamline slope)."""
    _, v, rho = flow_state(survey["p0"], survey["t0"], survey["ps"])
    v_axial = (v * np.cos(np.radians(survey["flow_angle"]))
               * np.cos(np.radians(slope_deg)))
    return rho * v_axial * area


def process(point: str) -> dict:
    """Rotor performance from the raw surveys of one operating point."""
    sl = load_streamlines()
    s1 = _at_radii(load_survey(1, point), sl["r1"])
    s2 = _at_radii(load_survey(2, point), sl["r2"])
    m1 = element_mass_flow(s1, sl["area1"], sl["slope1"])
    m2 = element_mass_flow(s2, sl["area2"], sl["slope2"])

    pr = s2["p0"] / s1["p0"]
    tr = s2["t0"] / s1["t0"]
    eta = (pr ** _K - 1.0) / (tr - 1.0)

    w = m2 / m2.sum()                               # mass weights at the exit
    tr_avg = float(np.sum(w * tr))
    pr_avg = float(np.sum(w * pr ** _K) ** (1.0 / _K))   # energy average
    eta_avg = (pr_avg ** _K - 1.0) / (tr_avg - 1.0)

    return {
        "span": (s2["radius"] - R_HUB2) / (R_SHROUD2 - R_HUB2),
        "pressure_ratio": pr, "temperature_ratio": tr, "efficiency": eta,
        "flow_station1": float(m1.sum()), "flow_station2": float(m2.sum()),
        "pressure_ratio_avg": pr_avg, "temperature_ratio_avg": tr_avg,
        "efficiency_avg": eta_avg,
    }


if __name__ == "__main__":
    for point in POINTS:
        r, o = process(point), load_overall(point)
        print(f"\n{point.replace('_', ' ')} (orifice flow {o['orifice_flow_kg_s']:.3f} kg/s)")
        print(f"  {'':28} {'this code':>10} {'NASA':>8}")
        print(f"  {'mass flow, station 1 [kg/s]':28} {r['flow_station1']:10.2f} "
              f"{o['station1_flow_kg_s']:8.2f}")
        print(f"  {'mass flow, station 2 [kg/s]':28} {r['flow_station2']:10.2f} "
              f"{o['station2_flow_kg_s']:8.2f}")
        print(f"  {'total pressure ratio':28} {r['pressure_ratio_avg']:10.3f} "
              f"{o['pressure_ratio']:8.3f}")
        print(f"  {'total temperature ratio':28} {r['temperature_ratio_avg']:10.3f} "
              f"{o['temperature_ratio']:8.3f}")
        print(f"  {'adiabatic efficiency':28} {r['efficiency_avg']:10.3f} "
              f"{o['efficiency']:8.3f}")
        print(f"\n  {'span':>6} {'PR':>7} {'TR':>7} {'eta':>7}")
        for s, p, t, e in zip(r["span"], r["pressure_ratio"], r["temperature_ratio"],
                              r["efficiency"]):
            print(f"  {s:6.2f} {p:7.3f} {t:7.4f} {e:7.3f}")