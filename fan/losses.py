"""
Rotor loss model and predicted efficiency along the span.

Two loss sources, each as a relative total pressure loss coefficient
omega = (p01_rel - p02_rel_actual) / (p01_rel - p1), defined on the inlet
relative dynamic head:

1. Profile loss (blade boundary layers), Lieblein's correlation in terms of
   the equivalent diffusion ratio D_eq = W_max / W2 (NACA-65 / C.4 cascades):

       D_eq = (W1/W2) [1.12 + 0.61 (cos^2 beta1 / sigma)(tan beta1 - tan beta2)]

       omega cos(beta2) / (2 sigma) (W1/W2)^2 = 0.004 [1 + 3.1 (D_eq - 1)^2
                                                       + 0.4 (D_eq - 1)^8]

   With a change of radius the tangential term is written with the change
   of angular momentum, as in the diffusion factor.

2. Shock loss (shock.py): a normal shock at the inlet relative Mach number
   ("normal"), or the Miller-Lewis-Hartmann model ("mlh"), which needs the
   blade geometry and accounts for the expansion along the suction surface.

The rotor is then followed along each streamline: rothalpy fixes the ideal
relative total pressure at exit, the loss reduces it, and the absolute
total pressure and efficiency follow. The efficiency feeds back into the
radial equilibrium (through the entropy gradient), so the two are iterated.

Units: SI.
"""

import numpy as np

from fan.annulus import Annulus
from fan.radial_equilibrium import SpanwiseSolution, solve_exit
from fan.shock import mlh_shock_loss, normal_shock_p0_ratio, shock_loss  # noqa: F401
from fan.velocity_triangles import CP, GAMMA

_EXP = GAMMA / (GAMMA - 1.0)
SHOCK_MODELS = ("normal", "mlh")


def equivalent_diffusion_ratio(sol: SpanwiseSolution, solidity: np.ndarray) -> np.ndarray:
    """Lieblein D_eq = W_max / W2 at minimum-loss incidence."""
    cos_b1 = sol.cx1 / sol.w1
    d_rc = np.abs(sol.r2 * sol.c_theta2) / (0.5 * (sol.r1 + sol.r2))   # no inlet swirl
    return (sol.w1 / sol.w2) * (1.12 + 0.61 * cos_b1 * d_rc / (solidity * sol.w1))


def profile_loss(sol: SpanwiseSolution, solidity: np.ndarray) -> np.ndarray:
    """Profile loss coefficient from Lieblein's D_eq correlation."""
    d = equivalent_diffusion_ratio(sol, solidity) - 1.0
    loss_parameter = 0.004 * (1.0 + 3.1 * d ** 2 + 0.4 * d ** 8)
    cos_b2 = sol.cx2 / sol.w2
    return loss_parameter * 2.0 * solidity / cos_b2 * (sol.w2 / sol.w1) ** 2


def efficiency_from_loss(sol: SpanwiseSolution, omega: np.ndarray,
                         p01: float = 101325.0) -> np.ndarray:
    """Isentropic efficiency of each streamline for a given loss coefficient.
    A loss larger than the available relative total pressure gives NaN."""
    t01 = sol.t01
    t1 = t01 - sol.cx1 ** 2 / (2.0 * CP)
    p1 = p01 * (t1 / t01) ** _EXP
    t01r = t1 + sol.w1 ** 2 / (2.0 * CP)
    p01r = p1 * (t01r / t1) ** _EXP

    # Rothalpy is conserved: T0rel changes only with blade speed (radius change)
    t02r = t01r + (sol.u2 ** 2 - sol.u1 ** 2) / (2.0 * CP)
    p02r = p01r * (t02r / t01r) ** _EXP - omega * (p01r - p1)

    with np.errstate(invalid="ignore"):        # runaway loss: negative p02r -> NaN
        t2 = t02r - sol.w2 ** 2 / (2.0 * CP)
        p2 = p02r * (t2 / t02r) ** _EXP
        p02 = p2 * (sol.t02 / t2) ** _EXP
        return ((p02 / p01) ** (1.0 / _EXP) - 1.0) / (sol.t02 / t01 - 1.0)


# Rotor 67 (NASA TP-2879): solidity 3.11 at the hub, 1.29 at the tip, 22 blades.
# Chord = sigma * 2 pi r / Z is then almost the same at hub and tip (94 and 92 mm):
# the blade has a nearly constant chord, so the solidity falls as 1/r.
_R_HUB_MEAN = 0.5 * (0.193 + 0.232) / 2     # mean of inlet and exit hub radius [m]
_R_TIP_MEAN = 0.5 * (0.514 + 0.485) / 2     # mean of inlet and exit tip radius [m]
_Z = 22
_C_HUB = 3.11 * 2.0 * np.pi * _R_HUB_MEAN / _Z
_C_TIP = 1.29 * 2.0 * np.pi * _R_TIP_MEAN / _Z


def rotor67_solidity(span: np.ndarray) -> np.ndarray:
    """Rotor 67 solidity: chord linear between the hub and tip values implied by
    the published solidities (3.11 and 1.29), divided by the local pitch."""
    r = _R_HUB_MEAN + (_R_TIP_MEAN - _R_HUB_MEAN) * span
    chord = _C_HUB + (_C_TIP - _C_HUB) * span
    return chord * _Z / (2.0 * np.pi * r)


def rotor67_thickness(span: np.ndarray) -> np.ndarray:
    """Maximum thickness / chord: assumed 10 % at the hub to 3 % at the tip."""
    return 0.10 + (0.03 - 0.10) * np.asarray(span)


class HubStallError(RuntimeError):
    """The loss iteration ran away: diffusion beyond Lieblein's data range."""


def analyze(inlet: Annulus, exit_: Annulus, rpm: float, mass_flow: float,
            c_theta_m: float, n: float, solidity_fn=rotor67_solidity,
            p01: float = 101325.0, t01: float = 288.15, eta_start: float = 0.92,
            relaxation: float = 0.5, tol: float = 1e-7, max_iter: int = 200,
            shock_model: str = "normal", thickness_fn=rotor67_thickness,
            n_blades: int = 22, front_fraction: float = 0.5):
    """Couple radial equilibrium and the loss model at fixed exit swirl (fixed work).

    Loss raises entropy, the entropy gradient changes the axial velocity
    through radial equilibrium, and that changes the diffusion and the loss.
    Near the hub this feedback can run away: lower axial velocity, more
    diffusion, more loss. Lieblein's data end near D_eq = 2, beyond which the
    blade section is considered stalled; a runaway is reported as HubStallError.
    front_fraction sets the camber line used by the MLH shock model.

    Returns (solution, efficiency, profile loss, shock loss), all along the span.
    """
    if shock_model not in SHOCK_MODELS:
        raise ValueError(f"unknown shock model {shock_model!r}, expected one of {SHOCK_MODELS}")
    eta = np.full(1, eta_start)
    d_eq = np.array([np.nan])
    for _ in range(max_iter):
        try:
            sol = solve_exit(inlet, exit_, rpm, mass_flow, c_theta_m, n,
                             eta if eta.size > 1 else float(eta[0]), p01, t01)
        except ValueError:
            break
        sigma = solidity_fn(sol.span)
        d_eq = equivalent_diffusion_ratio(sol, sigma)
        w_p = profile_loss(sol, sigma)
        if shock_model == "mlh":
            w_s = mlh_shock_loss(sol, sigma, thickness_fn(sol.span), n_blades,
                                 front_fraction)
        else:
            w_s = shock_loss(sol.mach_rel1())
        eta_new = efficiency_from_loss(sol, w_p + w_s, p01)
        if not np.all(np.isfinite(eta_new)):
            break                              # loss exceeds the available pressure
        if np.max(np.abs(eta_new - eta)) < tol:
            return sol, eta_new, w_p, w_s
        eta = relaxation * eta_new + (1.0 - relaxation) * eta
    raise HubStallError(f"loss iteration ran away, D_eq reached {np.nanmax(d_eq):.2f} "
                        "(Lieblein's data end near 2): the hub is predicted to stall")


def mass_averaged_efficiency(sol: SpanwiseSolution) -> float:
    """Efficiency from mass-averaged exit total temperature and pressure ratio."""
    pr = sol.mass_average(sol.pr)
    tr = sol.mass_average(sol.t02) / sol.t01
    return (pr ** (1.0 / _EXP) - 1.0) / (tr - 1.0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from fan.annulus import ROTOR67
    from fan.radial_equilibrium import design_for_pressure_ratio

    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mass_flow = ROTOR67["rpm"], ROTOR67["mass_flow"]

    print("Rotor 67 annulus and solidity. Each vortex is first sized for PR 1.63 at an")
    print("assumed efficiency of 0.92; the loss model then predicts the real efficiency.\n")
    results = {}
    for n, name in {-1: "free vortex", 0: "constant swirl", 1: "forced vortex"}.items():
        guess = design_for_pressure_ratio(inlet, exit_, rpm, mass_flow, 1.63, n, 0.92)
        c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** n
        try:
            sol, eta, w_p, w_s = analyze(inlet, exit_, rpm, mass_flow, c_theta_m, n)
        except HubStallError as err:
            print(f"{name:15} {err}")
            continue
        results[name] = (sol, eta, w_p, w_s)
        d_eq = equivalent_diffusion_ratio(sol, rotor67_solidity(sol.span))
        print(f"{name:15} efficiency {mass_averaged_efficiency(sol):.4f}, "
              f"PR {sol.mass_average(sol.pr):.3f}, max D_eq {d_eq.max():.2f}")

    sol, eta, w_p, w_s = results["free vortex"]
    span = sol.span
    d_eq = equivalent_diffusion_ratio(sol, rotor67_solidity(span))
    print("\nFree vortex along the span:")
    print(f"{'span':>5} {'M_rel1':>7} {'D_eq':>6} {'profile':>8} {'shock':>7} {'eta':>6}")
    for i in range(0, len(span), 20):
        print(f"{span[i]:5.2f} {sol.mach_rel1()[i]:7.3f} {d_eq[i]:6.2f} "
              f"{w_p[i]:8.4f} {w_s[i]:7.4f} {eta[i]:6.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    axes[0].plot(w_p, span, label="profile loss")
    axes[0].plot(w_s, span, label="shock loss")
    axes[0].plot(w_p + w_s, span, "k", label="total")
    axes[0].set_xlabel("loss coefficient")
    axes[0].legend(fontsize=8)
    axes[1].plot(d_eq, span)
    axes[1].axvline(2.0, color="k", ls="--", lw=1)
    axes[1].text(1.96, 0.5, "end of Lieblein data", rotation=90, fontsize=8, ha="right")
    axes[1].set_xlabel("equivalent diffusion ratio D_eq")
    axes[2].plot(eta, span)
    axes[2].set_xlabel("isentropic efficiency")
    axes[0].set_ylabel("span fraction (0 = hub, 1 = tip)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Free-vortex rotor in the Rotor 67 annulus: predicted losses")
    fig.tight_layout()
    fig.savefig("figures/loss_breakdown.png", dpi=150)
    print("\nSaved figures/loss_breakdown.png")