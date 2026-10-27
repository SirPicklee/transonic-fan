"""
Tip clearance loss: Denton's (1993) leakage mixing model.

The pressure difference across the blade tip drives a leakage jet through
the gap g. With suction- and pressure-surface velocities Vs and Vp,

    leakage velocity      V_L = sqrt(Vs^2 - Vp^2)
    leakage mass flow     dm  = Cc rho g V_L dz          (per unit chord dz)
    mixing loss           T ds = Vs^2 (1 - Vp/Vs)        (per unit leakage mass)

Integrated over the chord and divided by the passage flow
rho Cx2 s H, the loss coefficient on the exit relative dynamic head is

    zeta = 2 Cc (g/H) sigma / cos(beta2) (Vs/W2)^3 (1 - Vp/Vs) sqrt(1 - (Vp/Vs)^2)

for a uniform blade loading along the chord, with Vs, Vp = W_m +/- dV/2,
W_m = (W1 + W2)/2 and dV = (change of tangential velocity) / sigma from the
blade circulation. The efficiency penalty is zeta (W2^2 / 2) / dh0.

The loss is concentrated near the casing. How it spreads along the span is
not part of Denton's model; here it is spread over the outer part of the
span with a linear ramp, keeping its mass average equal to the passage loss.
It is applied after the radial equilibrium solution, not inside it.
"""

import numpy as np


def denton_leakage_loss(w1, w2, cx2, d_w_theta, solidity, gap, span, cc=0.8):
    """Passage loss coefficient on the exit relative dynamic head (W2^2 / 2)."""
    wm = 0.5 * (w1 + w2)
    dv = np.abs(d_w_theta) / solidity
    vs, vp = wm + 0.5 * dv, wm - 0.5 * dv
    ratio = vp / vs
    cos_b2 = cx2 / w2
    return (2.0 * cc * (gap / span) * solidity / cos_b2
            * (vs / w2) ** 3 * (1.0 - ratio) * np.sqrt(1.0 - ratio ** 2))


def efficiency_penalty(zeta, w2, work):
    """Mass-averaged efficiency drop for a loss coefficient on W2^2 / 2."""
    return zeta * 0.5 * w2 ** 2 / work


def distribute(span, mass_weight, total, width=0.3):
    """Spread a mass-averaged efficiency penalty over the outer `width` of the
    span with a linear ramp (zero at 1 - width, largest at the tip)."""
    w = np.clip((span - (1.0 - width)) / width, 0.0, None)
    mean_w = np.trapezoid(w * mass_weight, span) / np.trapezoid(mass_weight, span)
    return total * w / mean_w


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from fan.annulus import ROTOR67, Annulus
    from fan.losses import analyze, mass_averaged_efficiency, rotor67_solidity
    from fan.radial_equilibrium import design_for_pressure_ratio
    from fan.rig_analysis import process
    from fan.validation import monte_carlo

    GAP = 0.001          # Rotor 67 tip gap [m] (ERCOFTAC UFR 2-07, from NASA TP-2879)

    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, eta, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)

    span_len = 0.5 * ((inlet.r_tip - inlet.r_hub) + (exit_.r_tip - exit_.r_hub))
    t = -1                                                    # tip streamline
    d_w_theta = 2.0 * abs(sol.r2[t] * sol.c_theta2[t]) / (sol.r1[t] + sol.r2[t])
    sigma_tip = float(rotor67_solidity(np.array(1.0)))

    def penalty(gap):
        zeta = denton_leakage_loss(sol.w1[t], sol.w2[t], sol.cx2[t], d_w_theta,
                                   sigma_tip, gap, span_len)
        return efficiency_penalty(zeta, sol.w2[t], sol.work)

    print(f"Tip gap / span = {GAP / span_len * 100:.2f} %")
    print(f"{'gap [mm]':>9} {'efficiency drop [points]':>26}")
    for g in (0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3):
        print(f"{g * 1000:9.1f} {penalty(g) * 100:26.2f}")

    d_eta = penalty(GAP)
    eta_tc = eta - distribute(sol.span, sol.rho_cx2_r, d_eta)

    meas = process("peak_efficiency")
    mc = monte_carlo("peak_efficiency")
    band_avg = 2.0 * mc["efficiency_avg"].std()
    band = 2.0 * mc["efficiency"].std(axis=0)

    eta0 = mass_averaged_efficiency(sol)
    print(f"\nModel without tip clearance  {eta0:.3f}")
    print(f"Model with tip clearance     {eta0 - d_eta:.3f}")
    print(f"Measured                     {meas['efficiency_avg']:.3f} +/- {band_avg:.3f}")

    before = np.interp(meas["span"], sol.span, eta)
    after = np.interp(meas["span"], sol.span, eta_tc)
    print(f"\n{'span':>6} {'measured':>9} {'+/-2s':>7} {'model':>7} {'+ gap':>7} "
          f"{'gap - meas':>11}")
    for s, e, b, m0, m1 in zip(meas["span"], meas["efficiency"], band, before, after):
        print(f"{s:6.2f} {e:9.3f} {b:7.3f} {m0:7.3f} {m1:7.3f} {m1 - e:11.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(eta, sol.span, "--", label="model, no tip clearance")
    ax.plot(eta_tc, sol.span, label="model + 1 mm tip gap (Denton)")
    ax.errorbar(meas["efficiency"], meas["span"], xerr=band, fmt="ko", capsize=3,
                label="NASA TP-2879, 2-sigma")
    ax.set_xlabel("isentropic efficiency")
    ax.set_ylabel("span fraction (0 = hub, 1 = tip)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Effect of the tip clearance loss")
    fig.tight_layout()
    fig.savefig("figures/tip_clearance.png", dpi=150)
    print("\nSaved figures/tip_clearance.png")