"""
Validation: design model against the NASA Rotor 67 measurements, with the
measurement uncertainty estimated by Monte Carlo.

Every probe reading is perturbed by a random error, and the temperature and
pressure instruments of each station by one common (bias) error; the whole
processing chain is then rerun. A bias does not average out over the probes,
a random error partly does. The assumed accuracies are typical values for a
research rig, not numbers from the report.
"""

from dataclasses import dataclass

import numpy as np

from fan.rig_analysis import performance, streamline_surveys


@dataclass(frozen=True)
class Instrumentation:
    """Assumed instrument accuracy, one standard deviation."""
    t_random: float = 0.25       # [K] per reading
    t_bias: float = 0.25         # [K] common to all probes of one station
    p_random: float = 0.001      # relative, per reading
    p_bias: float = 0.0005       # relative, common to one station
    angle_random: float = 0.5    # [deg] per reading


def _perturb(s: dict, inst: Instrumentation, rng: np.random.Generator) -> dict:
    n = len(s["t0"])
    t_err = inst.t_bias * rng.standard_normal() + inst.t_random * rng.standard_normal(n)
    p_bias = inst.p_bias * rng.standard_normal()
    return {
        "radius": s["radius"],
        "t0": s["t0"] + t_err,
        "p0": s["p0"] * (1.0 + p_bias + inst.p_random * rng.standard_normal(n)),
        "ps": s["ps"] * (1.0 + p_bias + inst.p_random * rng.standard_normal(n)),
        "flow_angle": s["flow_angle"] + inst.angle_random * rng.standard_normal(n),
    }


def monte_carlo(point: str, inst: Instrumentation = Instrumentation(),
                n_samples: int = 2000, seed: int = 0) -> dict:
    """Distribution of the processed results under instrument errors."""
    s1, s2, sl = streamline_surveys(point)
    rng = np.random.default_rng(seed)
    keys = ("efficiency_avg", "pressure_ratio_avg", "temperature_ratio_avg",
            "flow_station2")
    out = {k: np.empty(n_samples) for k in keys}
    out["efficiency"] = np.empty((n_samples, len(sl["r2"])))
    for i in range(n_samples):
        r = performance(_perturb(s1, inst, rng), _perturb(s2, inst, rng), sl)
        for k in keys:
            out[k][i] = r[k]
        out["efficiency"][i] = r["efficiency"]
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from fan.annulus import ROTOR67, Annulus
    from fan.losses import analyze, mass_averaged_efficiency
    from fan.radial_equilibrium import design_for_pressure_ratio
    from fan.rig_analysis import process

    point = "peak_efficiency"
    meas = process(point)

    print("Measured rotor efficiency at peak efficiency, 2-sigma uncertainty:")
    scenarios = {"random errors only": Instrumentation(t_bias=0.0, p_bias=0.0),
                 "random + bias": Instrumentation()}
    for name, inst in scenarios.items():
        mc = monte_carlo(point, inst)
        print(f"  {name:20} {meas['efficiency_avg']:.3f} +/- "
              f"{2.0 * mc['efficiency_avg'].std():.3f}")
    eta_band = 2.0 * mc["efficiency"].std(axis=0)        # random + bias

    # Design model from the previous modules: free vortex, no endwall losses
    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, eta_model, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)
    print(f"Model (free vortex, no endwall or tip clearance loss): "
          f"{mass_averaged_efficiency(sol):.3f}")

    model_at_probes = np.interp(meas["span"], sol.span, eta_model)
    print(f"\n{'span':>6} {'measured':>9} {'+/-2s':>7} {'model':>7} {'model-meas':>11}")
    for s, e, b, m in zip(meas["span"], meas["efficiency"], eta_band, model_at_probes):
        print(f"{s:6.2f} {e:9.3f} {b:7.3f} {m:7.3f} {m - e:11.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    axes[0].plot(sol.pr, sol.span, label="model (free vortex)")
    axes[0].plot(meas["pressure_ratio"], meas["span"], "ko", label="NASA TP-2879")
    axes[0].set_xlabel("total pressure ratio")
    axes[1].plot(sol.t02 / sol.t01, sol.span)
    axes[1].plot(meas["temperature_ratio"], meas["span"], "ko")
    axes[1].set_xlabel("total temperature ratio")
    axes[2].plot(eta_model, sol.span)
    axes[2].errorbar(meas["efficiency"], meas["span"], xerr=eta_band, fmt="ko",
                     capsize=3, label="measured, 2-sigma")
    axes[2].set_xlabel("isentropic efficiency")
    axes[0].set_ylabel("span fraction (0 = hub, 1 = tip)")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Design model vs NASA Rotor 67 rig data "
                 "(model 33.25 kg/s design point, test 34.57 kg/s peak efficiency)")
    fig.tight_layout()
    fig.savefig("figures/model_vs_rig.png", dpi=150)
    print("\nSaved figures/model_vs_rig.png")