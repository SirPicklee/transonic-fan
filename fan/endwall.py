"""
Hub endwall loss: secondary flow and endwall friction near the hub.

Two contributions, both evaluated on the hub section, turned into a
mass-averaged efficiency penalty and spread over the inner part of the span
(the ramp of the tip clearance loss, mirrored):

1. Secondary loss, Howell's correlation: secondary drag coefficient
   C_Ds = 0.018 C_L^2 for the blade row, half of it attributed to the hub
   endwall. Lift coefficient from the flow turning,
       C_L = (2 / sigma) cos(beta_m) |tan(beta1) - tan(beta2)|,
       tan(beta_m) = (tan(beta1) + tan(beta2)) / 2,
   and loss coefficient on the inlet relative dynamic head
       zeta = C_D sigma cos^2(beta1) / cos^3(beta_m).
   Howell's data come from low-speed cascades. The casing side is not treated
   this way: at the highly staggered tip (beta_m about -66 deg) the 1/cos^3
   factor makes the correlation meaningless, and the tip already has the
   clearance loss.

2. Endwall friction, Denton's dissipation model: entropy production
   T dS = C_d rho V^3 per unit wetted area, C_d = 0.002, with V^3 the mean of
   the suction- and pressure-side velocity cubes, over one passage of the hub
   (pitch x axial chord). The hub turns with the blade: relative velocities.
"""

import math

import numpy as np

from fan.velocity_triangles import CP, GAMMA, R_AIR


def howell_secondary_loss(beta1_deg: float, beta2_deg: float, solidity: float,
                          share: float = 0.5):
    """Loss coefficient on the inlet relative dynamic head of Howell's secondary
    drag, `share` of it attributed to one endwall. Returns (zeta, C_L)."""
    t1, t2 = math.tan(math.radians(beta1_deg)), math.tan(math.radians(beta2_deg))
    cos_m = 1.0 / math.sqrt(1.0 + (0.5 * (t1 + t2)) ** 2)
    cl = 2.0 / solidity * cos_m * abs(t1 - t2)
    cd = share * 0.018 * cl ** 2
    cos_1 = math.cos(math.radians(beta1_deg))
    return cd * solidity * cos_1 ** 2 / cos_m ** 3, cl


def endwall_dissipation_penalty(w1, w2, d_w_theta, solidity, pitch, axial_chord,
                                rho, mass_flow_passage, work, cd=0.002):
    """Efficiency drop from friction on one endwall of a blade passage."""
    wm = 0.5 * (w1 + w2)
    dv = abs(d_w_theta) / solidity
    vs, vp = wm + 0.5 * dv, wm - 0.5 * dv
    v_cubed = 0.5 * (vs ** 3 + vp ** 3)
    return cd * rho * v_cubed * pitch * axial_chord / (mass_flow_passage * work)


def distribute_from_hub(span, mass_weight, total, width=0.3):
    """Spread a mass-averaged efficiency penalty over the inner `width` of the
    span with a linear ramp (largest at the hub, zero at `width`)."""
    w = np.clip((width - span) / width, 0.0, None)
    mean_w = np.trapezoid(w * mass_weight, span) / np.trapezoid(mass_weight, span)
    return total * w / mean_w


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from fan.annulus import ROTOR67, Annulus
    from fan.blade_sections import chord_from_solidity
    from fan.losses import analyze, mass_averaged_efficiency, rotor67_solidity
    from fan.radial_equilibrium import design_for_pressure_ratio
    from fan.rig_analysis import process
    from fan.tip_clearance import denton_leakage_loss, distribute, efficiency_penalty
    from fan.validation import monte_carlo

    Z, GAP, P01 = 22, 0.001, 101325.0
    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, eta, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)
    sigma = rotor67_solidity(sol.span)
    span_len = 0.5 * ((inlet.r_tip - inlet.r_hub) + (exit_.r_tip - exit_.r_hub))

    def d_w_theta(i):
        return 2.0 * abs(sol.r2[i] * sol.c_theta2[i]) / (sol.r1[i] + sol.r2[i])

    # Hub: Howell secondary loss and Denton endwall friction
    h = 0
    zeta_sec, cl_hub = howell_secondary_loss(sol.beta1[h], sol.beta2[h], sigma[h])
    d_sec = zeta_sec * 0.5 * sol.w1[h] ** 2 / sol.work
    r_h = 0.5 * (sol.r1[h] + sol.r2[h])
    t1, t2 = math.tan(math.radians(sol.beta1[h])), math.tan(math.radians(sol.beta2[h]))
    axial_chord = chord_from_solidity(sigma[h], r_h, Z) / math.sqrt(1.0 + (0.5 * (t1 + t2)) ** 2)
    t_static = sol.t01 - sol.cx1[h] ** 2 / (2.0 * CP)
    rho1 = P01 * (t_static / sol.t01) ** (GAMMA / (GAMMA - 1.0)) / (R_AIR * t_static)
    d_fric = endwall_dissipation_penalty(sol.w1[h], sol.w2[h], d_w_theta(h), sigma[h],
                                         2.0 * math.pi * r_h / Z, axial_chord, rho1,
                                         mdot / Z, sol.work)

    # Tip: Denton leakage loss, as in tip_clearance.py
    t = -1
    zeta_tip = denton_leakage_loss(sol.w1[t], sol.w2[t], sol.cx2[t], d_w_theta(t),
                                   sigma[t], GAP, span_len)
    d_tip = efficiency_penalty(zeta_tip, sol.w2[t], sol.work)

    print(f"Hub lift coefficient C_L = {cl_hub:.3f}")
    print(f"Efficiency penalties [points]: hub secondary {d_sec * 100:.2f}, "
          f"hub friction {d_fric * 100:.2f}, tip clearance {d_tip * 100:.2f}")

    eta_tip = eta - distribute(sol.span, sol.rho_cx2_r, d_tip)
    eta_all = eta_tip - distribute_from_hub(sol.span, sol.rho_cx2_r, d_sec + d_fric)

    meas = process("peak_efficiency")
    mc = monte_carlo("peak_efficiency")
    band = 2.0 * mc["efficiency"].std(axis=0)
    eta0 = mass_averaged_efficiency(sol)
    print(f"\nModel, profile and shock loss only      {eta0:.3f}")
    print(f"  + tip clearance                       {eta0 - d_tip:.3f}")
    print(f"  + tip clearance + hub endwall         {eta0 - d_tip - d_sec - d_fric:.3f}")
    print(f"Measured                                {meas['efficiency_avg']:.3f} "
          f"+/- {2.0 * mc['efficiency_avg'].std():.3f}")

    print(f"\n{'span':>6} {'measured':>9} {'+/-2s':>7} {'tip':>7} {'tip+hub':>8} {'diff':>7}")
    for s, e, b in zip(meas["span"], meas["efficiency"], band):
        a1 = np.interp(s, sol.span, eta_tip)
        a2 = np.interp(s, sol.span, eta_all)
        print(f"{s:6.2f} {e:9.3f} {b:7.3f} {a1:7.3f} {a2:8.3f} {a2 - e:7.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(eta, sol.span, ":", label="profile + shock loss")
    ax.plot(eta_tip, sol.span, "--", label="+ tip clearance")
    ax.plot(eta_all, sol.span, label="+ hub endwall")
    ax.errorbar(meas["efficiency"], meas["span"], xerr=band, fmt="ko", capsize=3,
                label="NASA TP-2879, 2-sigma")
    ax.set_xlabel("isentropic efficiency")
    ax.set_ylabel("span fraction (0 = hub, 1 = tip)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Endwall losses against the rig data")
    fig.tight_layout()
    fig.savefig("figures/endwall_losses.png", dpi=150)
    print("\nSaved figures/endwall_losses.png")