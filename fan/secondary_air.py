"""
Secondary air system: a flow network solver.

Air bled from the compressor flows through orifices and seals to cool
discs, pressurise bearing chambers and purge rim cavities against hot gas
ingestion. The network is a set of chambers (nodes) joined by flow
elements. Boundary nodes have a known pressure and temperature; the
pressures of the internal chambers follow from mass conservation, a system
of nonlinear equations.

Elements
  orifice    compressible isentropic nozzle flow with a discharge coefficient,
             choked below the critical pressure ratio x_crit = 0.528:
               m = Cd A p_u / sqrt(R T_u) sqrt(2g/(g-1) (x^(2/g) - x^((g+1)/g))),
               x = max(p_d / p_u, x_crit)
  labyrinth  Martin's equation for a straight-through labyrinth with N teeth
             and a kinetic energy carry-over factor k (choking of the last
             tooth is not modelled):
               m = k Cd A p_u / sqrt(R T_u) sqrt((1 - x^2) / (N - ln x))

Flow reverses when the downstream pressure is higher. Each chamber takes the
mass-weighted temperature of the flows entering it (no heat transfer, no
windage): an assumption. Values in example_network are illustrative, not
those of a real engine.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import root

from fan.velocity_triangles import GAMMA, R_AIR

X_CRIT = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))


@dataclass
class Element:
    name: str
    kind: str              # "orifice" or "labyrinth"
    a: str                 # node at one end
    b: str                 # node at the other end; positive flow is a -> b
    area: float            # [m^2]
    cd: float = 0.8        # discharge coefficient
    teeth: int = 1         # labyrinth only
    carry_over: float = 1.0

    def _forward(self, p_up, t_up, p_down):
        x = p_down / p_up
        base = self.cd * self.area * p_up / math.sqrt(R_AIR * t_up)
        if self.kind == "orifice":
            x = max(x, X_CRIT)
            phi = math.sqrt(2.0 * GAMMA / (GAMMA - 1.0)
                            * (x ** (2.0 / GAMMA) - x ** ((GAMMA + 1.0) / GAMMA)))
            return base * phi
        if self.kind == "labyrinth":
            return self.carry_over * base * math.sqrt((1.0 - x * x) / (self.teeth - math.log(x)))
        raise ValueError(f"unknown element kind {self.kind!r}")

    def flow(self, p_a, t_a, p_b, t_b):
        """Mass flow [kg/s], positive from a to b."""
        if p_a >= p_b:
            return self._forward(p_a, t_a, p_b)
        return -self._forward(p_b, t_b, p_a)


class Network:
    def __init__(self, boundaries: dict, chambers: list, elements: list):
        self.boundaries = boundaries          # name -> (pressure [Pa], temperature [K])
        self.chambers = list(chambers)
        self.elements = elements

    def _state(self, p_ch, t_ch):
        p = {n: v[0] for n, v in self.boundaries.items()}
        t = {n: v[1] for n, v in self.boundaries.items()}
        p.update(zip(self.chambers, p_ch))
        t.update(zip(self.chambers, t_ch))
        return p, t

    def _flows(self, p, t):
        return [e.flow(p[e.a], t[e.a], p[e.b], t[e.b]) for e in self.elements]

    def solve(self, max_iter: int = 50, tol: float = 1e-6) -> dict:
        """Chamber pressures and temperatures, and the flow in every element."""
        idx = {name: k for k, name in enumerate(self.chambers)}
        n = len(self.chambers)
        u = np.full(n, math.log(np.mean([v[0] for v in self.boundaries.values()])))
        t_ch = np.full(n, float(np.mean([v[1] for v in self.boundaries.values()])))

        for _ in range(max_iter):
            def residual(u_):
                p, t = self._state(np.exp(u_), t_ch)
                res = np.zeros(n)
                for e, f in zip(self.elements, self._flows(p, t)):
                    if e.b in idx:
                        res[idx[e.b]] += f
                    if e.a in idx:
                        res[idx[e.a]] -= f
                return res

            sol = root(residual, u, method="hybr", options={"xtol": 1e-12})
            if not sol.success:
                raise RuntimeError(f"network did not converge: {sol.message}")
            u = sol.x

            # Mixing: each chamber takes the temperature of the air entering it
            p, t = self._state(np.exp(u), t_ch)
            t_new = t_ch.copy()
            for k, name in enumerate(self.chambers):
                num = den = 0.0
                for e, f in zip(self.elements, self._flows(p, t)):
                    if e.b == name and f > 0.0:
                        num, den = num + f * t[e.a], den + f
                    if e.a == name and f < 0.0:
                        num, den = num - f * t[e.b], den - f
                if den > 0.0:
                    t_new[k] = num / den
            converged = np.max(np.abs(t_new - t_ch)) < tol
            t_ch = t_new
            if converged:
                break

        p, t = self._state(np.exp(u), t_ch)
        flows = dict(zip((e.name for e in self.elements), self._flows(p, t)))
        return {"p": p, "t": t, "flows": flows}


def example_network(seal_clearance: float = 0.4e-3) -> Network:
    """Illustrative turbine disc cavity: bleed air reaches the disc cavity
    through supply holes and pre-swirl nozzles, and leaves through an inner
    labyrinth (to the bearing chamber) and the rim seal (into the hot gas)."""
    boundaries = {
        "compressor bleed": (8.0e5, 650.0),
        "rim gas": (5.5e5, 1200.0),
        "bearing chamber": (1.2e5, 400.0),
    }
    elements = [
        Element("supply holes", "orifice", "compressor bleed", "supply cavity", 6.0e-4, 0.8),
        Element("pre-swirl nozzles", "orifice", "supply cavity", "disc cavity", 4.0e-4, 0.9),
        Element("inner labyrinth", "labyrinth", "disc cavity", "bearing chamber",
                math.pi * 0.25 * seal_clearance, 0.7, teeth=4),
        Element("rim seal", "orifice", "disc cavity", "rim gas", math.pi * 0.5 * 1.5e-3, 0.6),
    ]
    return Network(boundaries, ["supply cavity", "disc cavity"], elements)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    res = example_network().solve()
    print("Illustrative disc cavity network, new seal (0.4 mm clearance)\n")
    for name in ("supply cavity", "disc cavity"):
        print(f"  {name:15} p = {res['p'][name] / 1e5:6.3f} bar, T = {res['t'][name]:6.1f} K")
    for name, f in res["flows"].items():
        print(f"  {name:18} {f:7.4f} kg/s")
    margin = res["p"]["disc cavity"] / res["p"]["rim gas"] - 1.0
    print(f"  rim seal pressure margin {margin * 100:.2f} %")

    clearances = np.arange(0.2e-3, 1.81e-3, 0.1e-3)
    purge, t_cav, leak = [], [], []
    print(f"\n{'clearance':>10} {'labyrinth':>10} {'rim seal':>9} {'margin':>8} {'T cavity':>9}")
    for c in clearances:
        r = example_network(c).solve()
        purge.append(r["flows"]["rim seal"])
        leak.append(r["flows"]["inner labyrinth"])
        t_cav.append(r["t"]["disc cavity"])
        m = r["p"]["disc cavity"] / r["p"]["rim gas"] - 1.0
        flag = "  <- hot gas ingestion" if purge[-1] < 0 else ""
        print(f"{c * 1000:8.1f} mm {leak[-1]:10.4f} {purge[-1]:9.4f} {m * 100:7.2f}% "
              f"{t_cav[-1]:8.1f} K{flag}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True)
    ax1.plot(clearances * 1000, leak, label="inner labyrinth leakage")
    ax1.plot(clearances * 1000, purge, label="rim seal purge flow")
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_ylabel("mass flow [kg/s]")
    ax1.legend(fontsize=8)
    ax2.plot(clearances * 1000, t_cav)
    ax2.set_ylabel("disc cavity temperature [K]")
    ax2.set_xlabel("labyrinth seal clearance [mm]")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    ax1.set_title("Seal wear and rim purge (illustrative network)")
    fig.tight_layout()
    fig.savefig("figures/secondary_air.png", dpi=150)
    print("\nSaved figures/secondary_air.png")