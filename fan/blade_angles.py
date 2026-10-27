"""
Blade metal angles from flow angles: incidence and Carter's deviation rule.

The air does not follow the blade exactly. At the leading edge the flow
meets the blade at an incidence angle i; at the trailing edge it leaves
less turned than the blade, by the deviation angle delta. For circular-arc
camber lines Carter's rule gives

    delta = m * theta / sqrt(sigma),   m = 0.23 + 0.002 * |kappa2| (deg),

with camber theta = kappa2 - kappa1 and solidity sigma.

Sign convention (as in velocity_triangles.py): angles from axial, rotor
relative angles negative; the rotor turns the flow towards positive angles.
    kappa1 = beta1 + i       (positive i: flow hits the pressure side)
    kappa2 = beta2 + delta   (positive delta: the flow is under-turned)
"""

import numpy as np


def carter_m(kappa2_deg: np.ndarray) -> np.ndarray:
    """Carter coefficient for a circular-arc camber line (max camber at mid-chord)."""
    return 0.23 + 0.002 * np.abs(kappa2_deg)


def metal_angles(beta1: np.ndarray, beta2: np.ndarray, solidity: np.ndarray,
                 incidence: float | np.ndarray = 0.0, iterations: int = 20):
    """Blade inlet and outlet metal angles [deg] for given flow angles [deg].

    Carter's rule is implicit (m depends on kappa2, theta on delta). For a
    fixed m it has the closed form delta = m (beta2 - kappa1) / (sqrt(sigma) - m),
    which is iterated on m. Returns (kappa1, kappa2, camber, stagger, deviation).
    """
    kappa1 = np.asarray(beta1, dtype=float) + incidence
    beta2 = np.asarray(beta2, dtype=float)
    root_sigma = np.sqrt(np.asarray(solidity, dtype=float))
    kappa2 = beta2.copy()
    for _ in range(iterations):
        m = carter_m(kappa2)
        delta = m * (beta2 - kappa1) / (root_sigma - m)
        kappa2 = beta2 + delta
    camber = kappa2 - kappa1
    stagger = 0.5 * (kappa1 + kappa2)          # circular arc: chord line bisects
    return kappa1, kappa2, camber, stagger, delta


if __name__ == "__main__":
    from fan.annulus import ROTOR67, Annulus
    from fan.losses import analyze, rotor67_solidity
    from fan.radial_equilibrium import design_for_pressure_ratio

    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)

    sigma = rotor67_solidity(sol.span)
    k1, k2, camber, stagger, dev = metal_angles(sol.beta1, sol.beta2, sigma)

    print("Free-vortex rotor, zero incidence, Carter's rule\n")
    print(f"{'span':>5} {'beta1':>7} {'beta2':>7} {'turning':>8} {'dev':>6} "
          f"{'kappa1':>7} {'kappa2':>7} {'camber':>7} {'stagger':>8}")
    for i in range(0, len(sol.span), 20):
        print(f"{sol.span[i]:5.2f} {sol.beta1[i]:7.1f} {sol.beta2[i]:7.1f} "
              f"{sol.beta2[i] - sol.beta1[i]:8.1f} {dev[i]:6.1f} {k1[i]:7.1f} "
              f"{k2[i]:7.1f} {camber[i]:7.1f} {stagger[i]:8.1f}")