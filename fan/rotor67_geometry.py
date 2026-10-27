"""
NASA Rotor 67 blade geometry (Strazisar et al., NASA TP-2879, table I).

Fourteen blade sections on surfaces of revolution, from hub to tip. From the
transcribed leading and trailing edge circle centres, camber line angles and
maximum thickness, each section gets:

    chord        straight line from leading to trailing edge circle centre,
                 sqrt(meridional length^2 + (r_mean * angular change)^2)
    solidity     Z chord / (2 pi r_mean)
    metal angles kappa = -BETA (this project measures rotor angles negative,
                 against the rotation; the report gives them positive)
    front camber fraction of the camber between the first and the middle
                 tabulated point (0.5 for a circular arc)

The published solidities, 3.11 at the hub and 1.29 at the tip, are an
independent check of the transcription.
"""

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "rotor67_blade.csv"
N_BLADES = 22


def load_blade() -> dict:
    """Section data in SI units (m, rad, deg for angles), hub section first."""
    with open(DATA, newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    rows = list(csv.DictReader(lines))

    def col(key):
        return np.array([float(r[key]) for r in rows])

    r_le, r_te = col("r_le") / 100.0, col("r_te") / 100.0
    merid = col("merid_te") / 100.0
    d_ang = col("ang_te") - col("ang_le")
    r_mean = 0.5 * (r_le + r_te)
    tangential = r_mean * d_ang
    chord = np.hypot(merid, tangential)
    kappa1, kappa_mid, kappa2 = -col("beta_k1"), -col("beta_k18"), -col("beta_k35")
    camber = kappa2 - kappa1
    t_max = col("t_max") / 100.0

    return {
        "r_le": r_le, "r_te": r_te, "r_mean": r_mean,
        "z_le": col("z_le") / 100.0, "z_te": col("z_te") / 100.0,
        "span": (r_mean - r_mean[0]) / (r_mean[-1] - r_mean[0]),
        "chord": chord,
        "solidity": N_BLADES * chord / (2.0 * np.pi * r_mean),
        "stagger": -np.degrees(np.arctan2(tangential, merid)),
        "kappa1": kappa1, "kappa2": kappa2, "camber": camber,
        "front_fraction": (kappa_mid - kappa1) / camber,
        "t_max": t_max, "thickness_ratio": t_max / chord,
    }


if __name__ == "__main__":
    from fan.annulus import ROTOR67, Annulus
    from fan.blade_angles import metal_angles
    from fan.losses import analyze, rotor67_solidity
    from fan.radial_equilibrium import design_for_pressure_ratio

    b = load_blade()
    print("NASA Rotor 67 blade, table I of TP-2879\n")
    print(f"{'sec':>3} {'r [mm]':>7} {'chord':>6} {'sigma':>6} {'kappa1':>7} {'kappa2':>7} "
          f"{'camber':>7} {'front':>6} {'t/c':>6} {'stagger':>8}")
    for i in range(len(b["chord"])):
        print(f"{i + 1:3d} {b['r_mean'][i] * 1000:7.1f} {b['chord'][i] * 1000:6.1f} "
              f"{b['solidity'][i]:6.3f} {b['kappa1'][i]:7.2f} {b['kappa2'][i]:7.2f} "
              f"{b['camber'][i]:7.2f} {b['front_fraction'][i]:6.2f} "
              f"{b['thickness_ratio'][i]:6.3f} {b['stagger'][i]:8.1f}")
    print("\nPublished solidity: 3.11 at the hub, 1.29 at the tip")

    # Our free-vortex design at the same span fractions
    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)
    k1, k2, camber, *_ = metal_angles(sol.beta1, sol.beta2, rotor67_solidity(sol.span))

    print("\nRotor 67 against our free-vortex design (metal angles in degrees)")
    print(f"{'span':>5} {'R67 k1':>7} {'ours':>7} {'R67 k2':>7} {'ours':>7} "
          f"{'R67 cam':>8} {'ours':>7}")
    for i in (0, 4, 7, 10, 13):
        s = b["span"][i]
        print(f"{s:5.2f} {b['kappa1'][i]:7.1f} {np.interp(s, sol.span, k1):7.1f} "
              f"{b['kappa2'][i]:7.1f} {np.interp(s, sol.span, k2):7.1f} "
              f"{b['camber'][i]:8.1f} {np.interp(s, sol.span, camber):7.1f}")