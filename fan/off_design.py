"""
Off-design analysis of NASA Rotor 67 with its real blade geometry.

The blade is fixed (NASA TP-2879 table I, rotor67_geometry.py) and the mass
flow is imposed; the flow has to adapt to the blade:

  inlet    uniform axial inflow from continuity; incidence i = kappa1 - beta1
  exit     relative flow angle beta2 = kappa2 - delta, with Carter's deviation
           for the section's camber, solidity and maximum camber position,
           held at that value (an assumption), so the exit swirl follows the
           axial velocity: C_theta2 = U2 + Cx2 tan(beta2)
  balance  simple radial equilibrium with this swirl, iterated together with
           the losses until swirl and efficiency agree

Losses: Lieblein's profile loss with its incidence term,
    D_eq = (W1/W2) [1.12 + a (i - i_ref)^1.43 + 0.61 (cos^2 beta1 / sigma)(...)],
a = 0.007 (circular-arc C.4 sections). The reference (minimum-loss) incidence
i_ref is the incidence at the peak-efficiency test point: an assumption, a
peak-efficiency point being the minimum-loss point of its speed line. Only
incidences above i_ref (towards stall) are penalised; the choke side is not
modelled. Shock loss: normal shock at M1, or MLH on the real sections. Tip
clearance and hub endwall losses are applied afterwards, as in
tip_clearance.py and endwall.py.
"""

import math

import numpy as np

from fan.annulus import ROTOR67, Annulus, axial_mach, axial_velocity
from fan.blade_angles import carter_m
from fan.blade_sections import max_camber_position, mca_section
from fan.endwall import distribute_from_hub, endwall_dissipation_penalty, howell_secondary_loss
from fan.losses import efficiency_from_loss, equivalent_diffusion_ratio, mass_averaged_efficiency
from fan.radial_equilibrium import solve_exit_swirl
from fan.rotor67_geometry import load_blade
from fan.shock import (mach_from_prandtl_meyer, normal_shock_p0_ratio, prandtl_meyer,
                       shock_loss, suction_surface_turning)
from fan.tip_clearance import denton_leakage_loss, distribute, efficiency_penalty
from fan.velocity_triangles import CP, GAMMA, R_AIR

INLET = ROTOR67["annulus_inlet"]
EXIT = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
RPM = ROTOR67["rpm"]
Z, GAP, P01 = 22, 0.001, 101325.0
A_INCIDENCE = 0.007
_EXP = GAMMA / (GAMMA - 1.0)


class StallError(RuntimeError):
    """No converged solution at this mass flow: the rotor is predicted to stall."""


def blade_on_grid(n_points: int = 101) -> dict:
    """Rotor 67 geometry on the exit span grid, with Carter's deviation and
    the resulting exit relative flow angle beta2 [deg]."""
    b = load_blade()
    span = np.linspace(0.0, 1.0, n_points)
    keys = ("kappa1", "kappa2", "camber", "solidity", "chord", "thickness_ratio",
            "front_fraction")
    g = {k: np.interp(span, b["span"], b[k]) for k in keys}
    g["span"] = span
    g["a_over_c"] = np.array([max_camber_position(f) for f in g["front_fraction"]])
    m = carter_m(g["kappa2"], g["a_over_c"])
    g["deviation"] = m * g["camber"] / np.sqrt(g["solidity"])
    g["beta2"] = g["kappa2"] - g["deviation"]
    return g


def profile_loss_off_design(sol, g, i_excess):
    """Lieblein profile loss with the incidence term. Returns (loss, D_eq)."""
    d_eq = (equivalent_diffusion_ratio(sol, g["solidity"])
            + sol.w1 / sol.w2 * A_INCIDENCE * np.maximum(i_excess, 0.0) ** 1.43)
    d = d_eq - 1.0
    loss_parameter = 0.004 * (1.0 + 3.1 * d ** 2 + 0.4 * d ** 8)
    cos_b2 = sol.cx2 / sol.w2
    return loss_parameter * 2.0 * g["solidity"] / cos_b2 * (sol.w2 / sol.w1) ** 2, d_eq


def mlh_loss_on_blade(sol, g):
    """MLH shock loss on the real Rotor 67 sections, at the actual incidence."""
    m1 = sol.mach_rel1()
    loss = np.zeros_like(m1)
    pitch = 2.0 * np.pi * 0.5 * (sol.r1 + sol.r2) / Z
    for i in np.nonzero(m1 > 1.0)[0]:
        sec = mca_section(g["kappa1"][i], g["kappa2"][i], g["chord"][i],
                          g["thickness_ratio"][i], g["front_fraction"][i])
        turning = suction_surface_turning(sec, sol.beta1[i], pitch[i])
        ms = mach_from_prandtl_meyer(prandtl_meyer(m1[i]) + turning)
        p_over_p0 = (1.0 + 0.5 * (GAMMA - 1.0) * m1[i] ** 2) ** (-_EXP)
        loss[i] = (1.0 - normal_shock_p0_ratio(0.5 * (m1[i] + ms))) / (1.0 - p_over_p0)
    return loss


def analyze_point(mass_flow: float, g: dict, i_ref=None, shock_model: str = "normal",
                  relaxation: float = 0.3, tol: float = 1e-6, max_iter: int = 400) -> dict:
    """Rotor 67 at one mass flow (design speed). i_ref: reference incidence along
    the span [deg]; None means no incidence loss (used at the reference point)."""
    omega = RPM * 2.0 * math.pi / 60.0
    r2 = np.linspace(EXIT.r_hub, EXIT.r_tip, len(g["span"]))
    u2 = omega * r2
    tan_b2 = np.tan(np.radians(g["beta2"]))

    cx = np.full_like(r2, axial_velocity(axial_mach(INLET, mass_flow)))
    c_theta = u2 + cx * tan_b2
    eta = np.full_like(r2, 0.92)
    for _ in range(max_iter):
        try:
            sol = solve_exit_swirl(INLET, EXIT, RPM, mass_flow, c_theta, eta)
        except ValueError as err:
            raise StallError(f"{mass_flow:.2f} kg/s: radial equilibrium fails ({err})")
        incidence = g["kappa1"] - sol.beta1
        i_excess = 0.0 if i_ref is None else incidence - i_ref
        w_p, d_eq = profile_loss_off_design(sol, g, i_excess)
        w_s = mlh_loss_on_blade(sol, g) if shock_model == "mlh" else shock_loss(sol.mach_rel1())
        eta_new = efficiency_from_loss(sol, w_p + w_s)
        if not np.all(np.isfinite(eta_new)) or d_eq.max() > 2.5:
            raise StallError(f"{mass_flow:.2f} kg/s: loss runaway, D_eq {np.nanmax(d_eq):.2f}")
        c_theta_new = u2 + sol.cx2 * tan_b2
        change = max(np.max(np.abs(eta_new - eta)), np.max(np.abs(c_theta_new - c_theta)) / 100.0)
        eta = relaxation * eta_new + (1.0 - relaxation) * eta
        c_theta = relaxation * c_theta_new + (1.0 - relaxation) * c_theta
        if change < tol:
            break
    else:
        raise StallError(f"{mass_flow:.2f} kg/s: no converged solution")

    # Endwall losses, applied after convergence
    sigma = g["solidity"]
    span_len = 0.5 * ((INLET.r_tip - INLET.r_hub) + (EXIT.r_tip - EXIT.r_hub))

    def d_w_theta(i):
        return 2.0 * abs(sol.r2[i] * sol.c_theta2[i]) / (sol.r1[i] + sol.r2[i])

    t = -1
    zeta_tip = denton_leakage_loss(sol.w1[t], sol.w2[t], sol.cx2[t], d_w_theta(t),
                                   sigma[t], GAP, span_len)
    d_tip = efficiency_penalty(zeta_tip, sol.w2[t], sol.work)

    h = 0
    zeta_sec, _ = howell_secondary_loss(sol.beta1[h], sol.beta2[h], sigma[h])
    d_sec = zeta_sec * 0.5 * sol.w1[h] ** 2 / sol.work
    t1, t2 = math.tan(math.radians(sol.beta1[h])), math.tan(math.radians(sol.beta2[h]))
    axial_chord = g["chord"][h] / math.sqrt(1.0 + (0.5 * (t1 + t2)) ** 2)
    t_static = sol.t01 - sol.cx1[h] ** 2 / (2.0 * CP)
    rho1 = P01 * (t_static / sol.t01) ** _EXP / (R_AIR * t_static)
    r_h = 0.5 * (sol.r1[h] + sol.r2[h])
    d_fric = endwall_dissipation_penalty(sol.w1[h], sol.w2[h], d_w_theta(h), sigma[h],
                                         2.0 * math.pi * r_h / Z, axial_chord, rho1,
                                         mass_flow / Z, sol.work)

    eta_span = (eta_new - distribute(sol.span, sol.rho_cx2_r, d_tip)
                - distribute_from_hub(sol.span, sol.rho_cx2_r, d_sec + d_fric))
    eta_avg = mass_averaged_efficiency(sol) - d_tip - d_sec - d_fric
    tr_span = sol.t02 / sol.t01
    tr = sol.mass_average(sol.t02) / sol.t01
    return {
        "sol": sol, "span": sol.span, "incidence": incidence, "d_eq": d_eq,
        "efficiency": eta_span, "temperature_ratio": tr_span,
        "pressure_ratio": (1.0 + eta_span * (tr_span - 1.0)) ** _EXP,
        "efficiency_avg": eta_avg, "temperature_ratio_avg": tr,
        "pressure_ratio_avg": (1.0 + eta_avg * (tr - 1.0)) ** _EXP,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from fan.rig_analysis import process

    PEAK, STALL = 34.573, 32.305          # orifice mass flows [kg/s], TP-2879
    g = blade_on_grid()
    meas = {"peak": process("peak_efficiency"), "stall": process("near_stall")}

    print("Rotor 67, real blade, design speed. Measured values processed from TP-2879.\n")
    print(f"{'':22} {'PR':>7} {'TR':>7} {'eta':>7}   {'PR':>7} {'TR':>7} {'eta':>7}")
    print(f"{'':22} {'--- peak efficiency ---':>23}   {'----- near stall -----':>23}")
    m = meas
    print(f"{'measured':22} {m['peak']['pressure_ratio_avg']:7.3f} "
          f"{m['peak']['temperature_ratio_avg']:7.3f} {m['peak']['efficiency_avg']:7.3f}   "
          f"{m['stall']['pressure_ratio_avg']:7.3f} {m['stall']['temperature_ratio_avg']:7.3f} "
          f"{m['stall']['efficiency_avg']:7.3f}")

    results = {}
    for model in ("normal", "mlh"):
        peak = analyze_point(PEAK, g, None, model)
        try:
            stall = analyze_point(STALL, g, peak["incidence"], model)
        except StallError as err:
            stall = None
            print(f"{model} shock: {err}")
        results[model] = (peak, stall)
        row = f"{'model, ' + model + ' shock':22} {peak['pressure_ratio_avg']:7.3f} " \
              f"{peak['temperature_ratio_avg']:7.3f} {peak['efficiency_avg']:7.3f}   "
        if stall is not None:
            row += (f"{stall['pressure_ratio_avg']:7.3f} {stall['temperature_ratio_avg']:7.3f} "
                    f"{stall['efficiency_avg']:7.3f}")
        print(row)

    peak, stall = results["normal"]
    print("\nIncidence [deg] at peak efficiency (reference) and near stall:")
    for i in (0, 25, 50, 75, 100):
        s_inc = "" if stall is None else f"{stall['incidence'][i]:6.1f}"
        print(f"  span {g['span'][i]:.2f}: {peak['incidence'][i]:6.1f}  {s_inc}")

    print("\nEfficiency along the span (probes):")
    print(f"{'span':>6} {'peak meas':>9} {'normal':>7} {'MLH':>7}   "
          f"{'stall meas':>10} {'normal':>7} {'MLH':>7}")
    for k, s in enumerate(meas["peak"]["span"]):
        cells = [f"{s:6.2f}", f"{meas['peak']['efficiency'][k]:9.3f}"]
        for model in ("normal", "mlh"):
            cells.append(f"{np.interp(s, results[model][0]['span'], results[model][0]['efficiency']):7.3f}")
        cells.append(f"  {meas['stall']['efficiency'][k]:10.3f}")
        for model in ("normal", "mlh"):
            st = results[model][1]
            cells.append(f"{np.interp(s, st['span'], st['efficiency']):7.3f}" if st else "    -  ")
        print(" ".join(cells))

    # Speed line with the normal shock model
    flows = np.linspace(31.0, 35.0, 9)
    line = []
    print("\nSpeed line (normal shock):")
    for f in flows:
        try:
            r = analyze_point(f, g, peak["incidence"], "normal")
            line.append((f, r["pressure_ratio_avg"], r["efficiency_avg"]))
            print(f"  {f:5.2f} kg/s: PR {r['pressure_ratio_avg']:.3f}, eta {r['efficiency_avg']:.3f}")
        except StallError as err:
            print(f"  {err}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    if line:
        f_, pr_, eta_ = zip(*line)
        ax1.plot(f_, pr_, "-", label="model (normal shock)")
        ax2.plot(f_, eta_, "-")
    for key, flow, mk in (("peak", PEAK, "o"), ("stall", STALL, "s")):
        ax1.plot(flow, meas[key]["pressure_ratio_avg"], "k" + mk, label=f"measured, {key}")
        ax2.plot(flow, meas[key]["efficiency_avg"], "k" + mk)
    ax1.set_ylabel("total pressure ratio")
    ax2.set_ylabel("adiabatic efficiency")
    ax2.set_xlabel("mass flow [kg/s]")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("Rotor 67 design speed line: model with the real blade vs rig")
    fig.tight_layout()
    fig.savefig("figures/speedline.png", dpi=150)
    print("\nSaved figures/speedline.png")