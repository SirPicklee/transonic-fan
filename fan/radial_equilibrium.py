"""
Simple radial equilibrium at the rotor exit.

In a rotating flow the static pressure must rise with radius to hold the
swirling air on its circular path. Combined with the energy equation this
gives the simple radial equilibrium (SRE) equation, for axisymmetric flow
with straight, cylindrical streamlines:

    d(Cx^2 / 2)/dr = dh0/dr - T ds/dr - (C_theta / r) d(r C_theta)/dr

The designer chooses the exit swirl distribution ("vortex type"). The work
follows from the Euler equation, dh0 = omega r C_theta2 (no inlet swirl), the
entropy from an assumed rotor efficiency, and SRE then dictates how the axial
velocity must vary along the span. Its level is set by continuity.

Vortex family used in design:  C_theta2(r) = C_theta_m * (r / r_m)^n
    n = -1  free vortex      (r C_theta = const, uniform work, uniform Cx)
    n =  0  constant swirl
    n = +1  forced vortex    (work grows with r^2)

solve_exit_swirl takes any swirl distribution; the off-design analysis uses
it with a swirl that follows from the blade exit angle.

Streamlines are matched between the rotor inlet and exit by equal mass
fraction, so the annulus may contract through the rotor.

Units: SI.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from fan.annulus import Annulus, axial_mach, axial_velocity
from fan.velocity_triangles import CP, GAMMA, R_AIR

_EXP = GAMMA / (GAMMA - 1.0)


@dataclass
class SpanwiseSolution:
    """Rotor inlet and exit conditions along the span (index 0 = hub)."""
    r1: np.ndarray        # inlet radius of each streamline [m]
    r2: np.ndarray        # exit radius [m]
    cx1: np.ndarray       # inlet axial velocity [m/s]
    cx2: np.ndarray       # exit axial velocity [m/s]
    c_theta2: np.ndarray  # exit swirl [m/s]
    u1: np.ndarray        # blade speed at inlet [m/s]
    u2: np.ndarray        # blade speed at exit [m/s]
    t01: float            # inlet total temperature [K]
    t02: np.ndarray       # exit total temperature [K]
    pr: np.ndarray        # total pressure ratio
    mass_flow: float      # [kg/s]
    rho_cx2_r: np.ndarray  # mass flux times radius at exit (for mass averaging)

    @property
    def span(self) -> np.ndarray:
        """Exit span fraction, 0 at hub, 1 at tip."""
        return (self.r2 - self.r2[0]) / (self.r2[-1] - self.r2[0])

    @property
    def w1(self) -> np.ndarray:
        return np.hypot(self.cx1, self.u1)

    @property
    def w2(self) -> np.ndarray:
        return np.hypot(self.cx2, self.c_theta2 - self.u2)

    @property
    def beta1(self) -> np.ndarray:
        """Inlet relative flow angle from axial [deg]."""
        return np.degrees(np.arctan2(-self.u1, self.cx1))

    @property
    def beta2(self) -> np.ndarray:
        """Exit relative flow angle from axial [deg]."""
        return np.degrees(np.arctan2(self.c_theta2 - self.u2, self.cx2))

    @property
    def de_haller(self) -> np.ndarray:
        return self.w2 / self.w1

    def diffusion_factor(self, solidity: np.ndarray) -> np.ndarray:
        """Lieblein diffusion factor with radius change:
        DF = 1 - W2/W1 + |r2 C_theta2 - r1 C_theta1| / ((r1 + r2) sigma W1)."""
        return (1.0 - self.de_haller
                + np.abs(self.r2 * self.c_theta2) / ((self.r1 + self.r2) * solidity * self.w1))

    def mach_rel1(self) -> np.ndarray:
        t1 = self.t01 - self.cx1 ** 2 / (2.0 * CP)
        return self.w1 / np.sqrt(GAMMA * R_AIR * t1)

    def mass_average(self, q: np.ndarray) -> float:
        w = self.rho_cx2_r
        return float(np.trapezoid(q * w, self.r2) / np.trapezoid(w, self.r2))

    @property
    def work(self) -> float:
        """Mass-averaged specific work [J/kg]."""
        return CP * (self.mass_average(self.t02) - self.t01)


def solve_exit_swirl(inlet: Annulus, exit_: Annulus, rpm: float, mass_flow: float,
                     c_theta: np.ndarray, efficiency, p01: float = 101325.0,
                     t01: float = 288.15, picard_iterations: int = 8) -> SpanwiseSolution:
    """Exit flow field for a given exit swirl distribution, on the grid
    r2 = linspace(hub, tip, len(c_theta))."""
    omega = rpm * 2.0 * math.pi / 60.0
    c_theta = np.asarray(c_theta, dtype=float)
    r2 = np.linspace(exit_.r_hub, exit_.r_tip, len(c_theta))

    # Work, total conditions and entropy along the span (uniform inlet)
    dh0 = omega * r2 * c_theta
    t02 = t01 + dh0 / CP
    pr = (1.0 + efficiency * dh0 / (CP * t01)) ** _EXP
    p02 = p01 * pr
    s = CP * np.log(t02 / t01) - R_AIR * np.log(pr)

    dh0_dr = np.gradient(dh0, r2)
    ds_dr = np.gradient(s, r2)
    swirl_term = c_theta / r2 * np.gradient(r2 * c_theta, r2)

    def profile(cx_hub: float):
        """Integrate SRE from the hub; return Cx and density along the span."""
        t2 = t02.copy()
        for _ in range(picard_iterations):        # T depends on Cx: fixed point
            source = dh0_dr - t2 * ds_dr - swirl_term
            cx_sq = cx_hub ** 2 + 2.0 * cumulative_trapezoid(source, r2, initial=0.0)
            cx = np.sqrt(np.maximum(cx_sq, 1.0))
            t2 = t02 - (cx ** 2 + c_theta ** 2) / (2.0 * CP)
        rho = p02 * (t2 / t02) ** _EXP / (R_AIR * t2)
        return cx, rho, cx_sq.min()

    def flow_error(cx_hub: float) -> float:
        cx, rho, _ = profile(cx_hub)
        return np.trapezoid(rho * cx * 2.0 * math.pi * r2, r2) - mass_flow

    # Continuity sets the level of Cx. Search the subsonic, physical branch.
    cx_hub = brentq(flow_error, 20.0, _max_flow_cx(flow_error))
    cx2, rho2, min_cx_sq = profile(cx_hub)
    if min_cx_sq <= 1.0:
        raise ValueError("axial velocity collapses somewhere on the span: vortex not feasible")

    # Map exit streamlines to inlet radii by equal mass fraction (uniform inlet)
    flux = rho2 * cx2 * r2
    frac = cumulative_trapezoid(flux, r2, initial=0.0)
    frac /= frac[-1]
    r1 = np.sqrt(inlet.r_hub ** 2 + frac * (inlet.r_tip ** 2 - inlet.r_hub ** 2))
    cx1 = np.full_like(r2, axial_velocity(axial_mach(inlet, mass_flow, p01, t01), t01))

    return SpanwiseSolution(r1=r1, r2=r2, cx1=cx1, cx2=cx2, c_theta2=c_theta,
                            u1=omega * r1, u2=omega * r2, t01=t01, t02=t02, pr=pr,
                            mass_flow=mass_flow, rho_cx2_r=flux)


def solve_exit(inlet: Annulus, exit_: Annulus, rpm: float, mass_flow: float,
               c_theta_m: float, n: float, efficiency: float,
               p01: float = 101325.0, t01: float = 288.15,
               n_points: int = 101, picard_iterations: int = 8) -> SpanwiseSolution:
    """Exit flow field for the vortex C_theta2 = c_theta_m (r / r_m)^n."""
    r2 = np.linspace(exit_.r_hub, exit_.r_tip, n_points)
    c_theta = c_theta_m * (r2 / exit_.r_mean) ** n
    return solve_exit_swirl(inlet, exit_, rpm, mass_flow, c_theta, efficiency,
                            p01, t01, picard_iterations)


def _max_flow_cx(flow_error) -> float:
    """Hub axial velocity of maximum mass flow (exit choke): upper bracket for brentq."""
    grid = np.linspace(20.0, 600.0, 117)
    flows = [flow_error(c) for c in grid]
    return float(grid[int(np.argmax(flows))])


def design_for_pressure_ratio(inlet: Annulus, exit_: Annulus, rpm: float,
                              mass_flow: float, pressure_ratio: float, n: float,
                              efficiency: float, p01: float = 101325.0,
                              t01: float = 288.15, **kw) -> SpanwiseSolution:
    """Choose the swirl level so that the mass-averaged work matches the
    work needed for `pressure_ratio` at `efficiency`."""
    target = CP * t01 * (pressure_ratio ** (1.0 / _EXP) - 1.0) / efficiency

    def error(c_theta_m):
        try:
            return solve_exit(inlet, exit_, rpm, mass_flow, c_theta_m, n, efficiency,
                              p01, t01, **kw).work - target
        except ValueError:
            return 1e9      # too much swirl: the axial velocity collapsed

    c_theta_m = brentq(error, 10.0, 300.0, xtol=1e-6)
    return solve_exit(inlet, exit_, rpm, mass_flow, c_theta_m, n, efficiency, p01, t01, **kw)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from fan.annulus import ROTOR67

    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)   # Rotor 67 exit annulus
    rpm, mass_flow, pr_target = ROTOR67["rpm"], ROTOR67["mass_flow"], 1.63
    efficiency = 0.92          # assumed rotor efficiency, uniform along the span

    vortices = {-1: "free vortex (n = -1)", 0: "constant swirl (n = 0)",
                1: "forced vortex (n = +1)"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)

    print(f"Rotor 67 annulus, {mass_flow} kg/s, mass-averaged PR {pr_target}, "
          f"eta {efficiency}\n")
    print(f"{'vortex':24} {'PR hub':>7} {'PR tip':>7} {'Cx2 hub':>8} {'Cx2 tip':>8} "
          f"{'DF hub':>7} {'DF tip':>7} {'turn hub':>9}")
    for n, name in vortices.items():
        sol = design_for_pressure_ratio(inlet, exit_, rpm, mass_flow, pr_target, n,
                                        efficiency)
        solidity = 1.288 * exit_.r_tip / sol.r2      # tip solidity 1.288, constant chord
        df = sol.diffusion_factor(solidity)
        turning = sol.beta2 - sol.beta1
        print(f"{name:24} {sol.pr[0]:7.3f} {sol.pr[-1]:7.3f} {sol.cx2[0]:8.1f} "
              f"{sol.cx2[-1]:8.1f} {df[0]:7.3f} {df[-1]:7.3f} {turning[0]:8.1f}°")

        span = sol.span
        axes[0].plot(sol.pr, span, label=name)
        axes[1].plot(sol.cx2, span)
        axes[2].plot(df, span)
        axes[3].plot(turning, span)

    axes[2].axvline(0.45, color="k", ls="--", lw=1)
    axes[2].text(0.455, 0.05, "DF limit", fontsize=8)
    for ax, label in zip(axes, ["total pressure ratio", "exit axial velocity [m/s]",
                                "diffusion factor", "relative flow turning [deg]"]):
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("span fraction (0 = hub, 1 = tip)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Choice of vortex at the same mass-averaged pressure ratio "
                 "(Rotor 67 annulus, simple radial equilibrium)")
    fig.tight_layout()
    fig.savefig("figures/vortex_comparison.png", dpi=150)
    print("\nSaved figures/vortex_comparison.png")