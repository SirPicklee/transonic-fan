"""
Shock loss of a transonic rotor section: Miller, Lewis and Hartmann (1961).

A single normal shock stands at the entrance of the blade passage, running
from the leading edge of one blade, normal to the inlet flow, to the suction
surface of the neighbouring blade. Upstream of it the flow is not uniform:
along the convex suction surface the supersonic inlet flow expands
(Prandtl-Meyer) by the angle the surface turns away from the inlet flow
before the shock, and reaches Ms > M1. The shock is taken at the average
Mach number (M1 + Ms) / 2.

The simpler estimate, a normal shock at M1, ignores this expansion and so
underestimates the loss. Both are here. Loss coefficients are on the inlet
relative dynamic head, as in losses.py.

The expansion depends on the blade shape: the less camber ahead of the
shock, the weaker the expansion. Sections are built with the two-arc camber
line of blade_sections.py; front_fraction = 0.5 is a circular arc.

Geometry (blade_sections.py): X axial, Y tangential, the blade moving in +Y.
The suction side is the "lower" surface, facing -Y; the neighbouring blade
across that passage sits one pitch away in -Y.
"""

import math

import numpy as np
from scipy.optimize import brentq

from fan.blade_angles import metal_angles
from fan.blade_sections import chord_from_solidity, max_camber_position, mca_section
from fan.velocity_triangles import GAMMA

_EXP = GAMMA / (GAMMA - 1.0)


def normal_shock_p0_ratio(mach):
    """Total pressure ratio p02/p01 across a normal shock (1 for M <= 1)."""
    m = np.maximum(np.asarray(mach, dtype=float), 1.0)
    g = GAMMA
    a = ((g + 1.0) * m ** 2 / ((g - 1.0) * m ** 2 + 2.0)) ** (g / (g - 1.0))
    b = ((g + 1.0) / (2.0 * g * m ** 2 - (g - 1.0))) ** (1.0 / (g - 1.0))
    return a * b


def _p_over_p0(mach):
    return (1.0 + 0.5 * (GAMMA - 1.0) * np.asarray(mach, dtype=float) ** 2) ** (-_EXP)


def shock_loss(mach_rel1):
    """Loss coefficient of a normal shock at the inlet relative Mach number."""
    m = np.asarray(mach_rel1, dtype=float)
    return (1.0 - normal_shock_p0_ratio(m)) / (1.0 - _p_over_p0(m))


def prandtl_meyer(mach: float) -> float:
    """Prandtl-Meyer angle nu(M) [rad]."""
    a = math.sqrt((GAMMA + 1.0) / (GAMMA - 1.0))
    root = math.sqrt(max(mach * mach - 1.0, 0.0))
    return a * math.atan(root / a) - math.atan(root)


def mach_from_prandtl_meyer(nu: float) -> float:
    """Supersonic Mach number for a Prandtl-Meyer angle [rad]."""
    if nu <= 0.0:
        return 1.0
    return brentq(lambda m: prandtl_meyer(m) - nu, 1.0, 10.0)


def suction_surface_turning(section: dict, beta1_deg: float, pitch: float) -> float:
    """Expansion angle [rad] of the supersonic inflow along the suction surface,
    up to the point where the passage shock meets it: the suction surface
    direction there minus the inlet flow direction (zero if the surface turns
    into the flow). The shock starts at the leading edge of the neighbouring
    blade (one pitch in -Y) and runs normal to the inlet flow direction.

    Measured from the flow direction, not from the surface direction at the
    leading edge: a thickness wedge inclined into the flow compresses it there."""
    xs, ys = (np.asarray(v) for v in section["lower"])
    qx, qy = xs[0], ys[0] - pitch                  # neighbour's leading edge
    b = math.radians(beta1_deg)
    nx, ny = -math.sin(b), math.cos(b)             # normal to the inlet flow, towards +Y

    ex, ey = np.diff(xs), np.diff(ys)              # suction surface segments
    wx, wy = xs[:-1] - qx, ys[:-1] - qy
    den = nx * ey - ny * ex
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * ey - wy * ex) / den              # distance along the shock line
        u = (wx * ny - wy * nx) / den              # position along the segment
    hit = np.nonzero(np.isfinite(t) & (t > 0.0) & (u >= 0.0) & (u <= 1.0))[0]
    if hit.size == 0:
        return 0.0
    i = hit[np.argmin(t[hit])]
    surface_angle = math.atan2(ey[i], ex[i])
    return max(0.0, surface_angle - b)


def mlh_details(sol, solidity, thickness_ratio, n_blades: int = 22,
                front_fraction: float = 0.5):
    """Suction-surface expansion angle [rad], suction-surface Mach number and
    MLH loss coefficient along the span (subsonic streamlines: zero loss)."""
    m1 = sol.mach_rel1()
    turning = np.zeros_like(m1)
    ms = m1.copy()
    loss = np.zeros_like(m1)
    sup = np.nonzero(m1 > 1.0)[0]
    if sup.size == 0:
        return turning, ms, loss

    k1, k2, *_ = metal_angles(sol.beta1, sol.beta2, solidity,
                              a_over_c=max_camber_position(front_fraction))
    radius = 0.5 * (sol.r1 + sol.r2)
    chord = chord_from_solidity(solidity, radius, n_blades)
    pitch = 2.0 * np.pi * radius / n_blades
    for i in sup:
        sec = mca_section(k1[i], k2[i], chord[i], thickness_ratio[i], front_fraction)
        turning[i] = suction_surface_turning(sec, sol.beta1[i], pitch[i])
        ms[i] = mach_from_prandtl_meyer(prandtl_meyer(m1[i]) + turning[i])
        m_avg = 0.5 * (m1[i] + ms[i])
        loss[i] = (1.0 - normal_shock_p0_ratio(m_avg)) / (1.0 - _p_over_p0(m1[i]))
    return turning, ms, loss


def mlh_shock_loss(sol, solidity, thickness_ratio, n_blades: int = 22,
                   front_fraction: float = 0.5):
    """MLH shock loss coefficient along the span."""
    return mlh_details(sol, solidity, thickness_ratio, n_blades, front_fraction)[2]


if __name__ == "__main__":
    from fan.annulus import ROTOR67, Annulus
    from fan.losses import analyze, mass_averaged_efficiency, rotor67_solidity, rotor67_thickness
    from fan.radial_equilibrium import design_for_pressure_ratio
    from fan.rig_analysis import process

    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    meas = process("peak_efficiency")

    sol, eta, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1, shock_model="normal")
    probes = [np.interp(s, sol.span, eta) for s in meas["span"][:3]]
    print("Efficiency without tip clearance loss; probes at 90, 81 and 71 % span\n")
    print(f"{'shock model':24} {'efficiency':>10} {'turn 60%':>9} {'turn tip':>9} "
          f"{'eta 90%':>8} {'eta 81%':>8} {'eta 71%':>8}")
    print(f"{'normal shock at M1':24} {mass_averaged_efficiency(sol):10.4f} {'':>9} {'':>9} "
          + " ".join(f"{p:8.3f}" for p in probes))

    for ff in (0.5, 0.35, 0.25, 0.15):
        sol, eta, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1,
                               shock_model="mlh", front_fraction=ff)
        turning, _, _ = mlh_details(sol, rotor67_solidity(sol.span),
                                    rotor67_thickness(sol.span), front_fraction=ff)
        probes = [np.interp(s, sol.span, eta) for s in meas["span"][:3]]
        print(f"{'MLH, front camber ' + format(ff, '.2f'):24} "
              f"{mass_averaged_efficiency(sol):10.4f} "
              f"{math.degrees(turning[60]):8.2f}° {math.degrees(turning[-1]):8.2f}° "
              + " ".join(f"{p:8.3f}" for p in probes))

    print(f"{'measured':24} {meas['efficiency_avg']:10.4f} {'':>9} {'':>9} "
          + " ".join(f"{e:8.3f}" for e in meas["efficiency"][:3]))