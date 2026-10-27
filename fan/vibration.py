"""
Blade vibration and flutter screening.

Structural model: the blade is a cantilever beam along the radius, clamped
at the hub, with the section properties of the designed sections:

  flap and edgewise bending  Euler-Bernoulli beam elements (Hermite cubics)
                             about the minimum and maximum principal axes,
                             stiffened by the centrifugal tension
                             T(r) = Omega^2 rho * integral from r to tip of A r dr
  torsion                    St Venant torsion, linear elements,
                             J = (64/420) c t_max^3 for the lens-shaped section,
                             no centrifugal stiffening

Simplifications: modes are uncoupled (shear centre at the centroid, no
bending-torsion coupling from pretwist), no rotating-frame softening of the
edgewise mode, rigid disc and root.

Campbell diagram: blade frequencies against rotor speed with the engine
order lines f = N rpm / 60. A crossing is a resonance with a forcing that
repeats N times per revolution (struts, inlet distortion, ...).

Flutter screening: reduced frequency k = omega b / W1, with b the semichord
and W1 the inlet relative velocity. A low k means the air has time to follow
the blade motion and can feed energy into it. The screening value used here
is an assumption; a real assessment computes the aerodynamic damping.

Material: titanium alloy Ti-6Al-4V (assumed, typical for solid fan blades).
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

TI64 = {"E": 113.8e9, "G": 42.4e9, "rho": 4430.0}     # [Pa], [Pa], [kg/m^3]
K_SCREEN = 0.3      # assumed screening value of the reduced frequency (semichord)


def section_properties(x, y):
    """Area, minimum and maximum principal second moment of area, and polar
    second moment about the centroid, of a closed polygon."""
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * cross.sum()
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    ixx = ((y ** 2 + y * y1 + y1 ** 2) * cross).sum() / 12.0      # integral of y^2 dA
    iyy = ((x ** 2 + x * x1 + x1 ** 2) * cross).sum() / 12.0      # integral of x^2 dA
    ixy = ((x * y1 + 2.0 * x * y + 2.0 * x1 * y1 + x1 * y) * cross).sum() / 24.0
    s = np.sign(a)                                  # clockwise or anticlockwise loop
    a, ixx, iyy, ixy = abs(a), s * ixx, s * iyy, s * ixy
    ixx -= a * cy ** 2                              # parallel axis to the centroid
    iyy -= a * cx ** 2
    ixy -= a * cx * cy
    mean = 0.5 * (ixx + iyy)
    dev = math.hypot(0.5 * (ixx - iyy), ixy)
    return float(a), float(mean - dev), float(mean + dev), float(ixx + iyy)


def _beam_matrices(r, ei, rho_a, tension):
    """Bending stiffness, geometric (tension) stiffness and mass, clamped at r[0]."""
    n = 2 * len(r)
    k_b, k_g, m = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for e in range(len(r) - 1):
        L = r[e + 1] - r[e]
        EI = 0.5 * (ei[e] + ei[e + 1])
        P = 0.5 * (tension[e] + tension[e + 1])
        mu = 0.5 * (rho_a[e] + rho_a[e + 1])
        kb = EI / L ** 3 * np.array([[12, 6 * L, -12, 6 * L],
                                     [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                                     [-12, -6 * L, 12, -6 * L],
                                     [6 * L, 2 * L * L, -6 * L, 4 * L * L]])
        kg = P / (30.0 * L) * np.array([[36, 3 * L, -36, 3 * L],
                                        [3 * L, 4 * L * L, -3 * L, -L * L],
                                        [-36, -3 * L, 36, -3 * L],
                                        [3 * L, -L * L, -3 * L, 4 * L * L]])
        me = mu * L / 420.0 * np.array([[156, 22 * L, 54, -13 * L],
                                        [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                        [54, 13 * L, 156, -22 * L],
                                        [-13 * L, -3 * L * L, -22 * L, 4 * L * L]])
        i = slice(2 * e, 2 * e + 4)
        k_b[i, i] += kb
        k_g[i, i] += kg
        m[i, i] += me
    free = slice(2, n)                              # clamp displacement and slope at the hub
    return k_b[free, free], k_g[free, free], m[free, free]


def _torsion_matrices(r, gj, rho_ip):
    """Torsional stiffness and inertia, clamped at r[0]."""
    n = len(r)
    k, m = np.zeros((n, n)), np.zeros((n, n))
    for e in range(n - 1):
        L = r[e + 1] - r[e]
        GJ = 0.5 * (gj[e] + gj[e + 1])
        I = 0.5 * (rho_ip[e] + rho_ip[e + 1])
        k[e:e + 2, e:e + 2] += GJ / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        m[e:e + 2, e:e + 2] += I * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    return k[1:, 1:], m[1:, 1:]


def _frequencies(k, m, n_modes):
    w2 = eigh(k, m, eigvals_only=True)
    return np.sqrt(np.maximum(w2[:n_modes], 0.0)) / (2.0 * math.pi)


@dataclass
class BladeStructure:
    r: np.ndarray          # radius of the stations, hub first [m]
    area: np.ndarray       # [m^2]
    i_min: np.ndarray      # minimum principal second moment [m^4] (flap)
    i_max: np.ndarray      # maximum principal second moment [m^4] (edgewise)
    j_torsion: np.ndarray  # St Venant torsion constant [m^4]
    i_polar: np.ndarray    # polar second moment about the centroid [m^4]
    material: dict

    def tension_per_omega2(self) -> np.ndarray:
        """Centrifugal tension divided by Omega^2 [N s^2]."""
        f = self.material["rho"] * self.area * self.r
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(self.r))])
        return cum[-1] - cum

    def frequencies(self, rpm: float) -> dict:
        """Natural frequencies [Hz] of the first modes at a rotor speed."""
        E, G, rho = self.material["E"], self.material["G"], self.material["rho"]
        omega2 = (rpm * 2.0 * math.pi / 60.0) ** 2
        t1 = self.tension_per_omega2()
        out = {}
        for name, inertia in (("F", self.i_min), ("E", self.i_max)):
            k, kg, m = _beam_matrices(self.r, E * inertia, rho * self.area, t1)
            f = _frequencies(k + omega2 * kg, m, 2)
            out[f"1{name}"], out[f"2{name}"] = f
        k, m = _torsion_matrices(self.r, G * self.j_torsion, rho * self.i_polar)
        out["1T"] = _frequencies(k, m, 1)[0]
        return {key: float(out[key]) for key in ("1F", "2F", "1E", "1T")}


def rotor67_structure(n_stations: int = 21, material: dict = TI64):
    """Beam model of the free-vortex blade of the previous modules.
    Returns (structure, aerodynamic solution, chord and indices of the stations)."""
    from fan.annulus import ROTOR67, Annulus
    from fan.blade3d import section_loop
    from fan.blade_angles import metal_angles
    from fan.blade_sections import chord_from_solidity, circular_arc_section
    from fan.losses import analyze, rotor67_solidity
    from fan.radial_equilibrium import design_for_pressure_ratio

    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)

    sigma = rotor67_solidity(sol.span)
    _, _, camber, stagger, _ = metal_angles(sol.beta1, sol.beta2, sigma)
    radius = 0.5 * (sol.r1 + sol.r2)
    chord = chord_from_solidity(sigma, radius, 22)
    t_ratio = 0.10 + (0.03 - 0.10) * sol.span       # assumed, as in blade_sections

    idx = np.round(np.linspace(0, len(sol.span) - 1, n_stations)).astype(int)
    props = [section_properties(*section_loop(
        circular_arc_section(camber[i], stagger[i], chord[i], t_ratio[i]))) for i in idx]
    area, i_min, i_max, i_polar = (np.array(p) for p in zip(*props))
    j = 64.0 / 420.0 * chord[idx] * (t_ratio[idx] * chord[idx]) ** 3
    structure = BladeStructure(radius[idx], area, i_min, i_max, j, i_polar, material)
    return structure, sol, chord, idx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    blade, sol, chord, idx = rotor67_structure()
    design_rpm = 16043.0

    print("Section properties (hub, mid, tip):")
    for k in (0, len(idx) // 2, len(idx) - 1):
        print(f"  r = {blade.r[k] * 1000:6.1f} mm: area {blade.area[k] * 1e6:6.1f} mm^2, "
              f"I_min {blade.i_min[k] * 1e12:8.0f} mm^4, J {blade.j_torsion[k] * 1e12:7.0f} mm^4")

    f0, fd = blade.frequencies(0.0), blade.frequencies(design_rpm)
    print(f"\n{'mode':>5} {'0 rpm [Hz]':>11} {'design [Hz]':>12}")
    for mode in f0:
        print(f"{mode:>5} {f0[mode]:11.0f} {fd[mode]:12.0f}")

    rpms = np.linspace(0.0, 1.1 * design_rpm, 45)
    table = {mode: np.array([blade.frequencies(n)[mode] for n in rpms]) for mode in f0}

    print("\nCampbell crossings (engine orders 1-8, up to 110 % speed):")
    for mode, f in table.items():
        for eo in range(1, 9):
            g = f - eo * rpms / 60.0
            for i in np.nonzero(np.sign(g[:-1]) != np.sign(g[1:]))[0]:
                n_cross = rpms[i] - g[i] * (rpms[i + 1] - rpms[i]) / (g[i + 1] - g[i])
                print(f"  {mode} x {eo}EO at {n_cross:6.0f} rpm "
                      f"({n_cross / design_rpm * 100:5.1f} % speed)")

    # Flutter screening at design speed
    w1 = sol.w1[idx]
    print(f"\nReduced frequency k = 2 pi f (c/2) / W1 at design speed "
          f"(screening value {K_SCREEN}, assumed):")
    print(f"{'span':>6} {'W1 [m/s]':>9} {'k 1F':>7} {'k 1T':>7}")
    for k in range(0, len(idx), 5):
        b = 0.5 * chord[idx][k]
        k_f = 2.0 * math.pi * fd["1F"] * b / w1[k]
        k_t = 2.0 * math.pi * fd["1T"] * b / w1[k]
        flag = "  <- below screening value" if min(k_f, k_t) < K_SCREEN else ""
        print(f"{sol.span[idx][k]:6.2f} {w1[k]:9.1f} {k_f:7.3f} {k_t:7.3f}{flag}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for mode, f in table.items():
        ax.plot(rpms / design_rpm * 100, f, lw=2, label=mode)
    for eo in range(1, 9):
        ax.plot(rpms / design_rpm * 100, eo * rpms / 60.0, "k:", lw=0.8)
        ax.text(110, eo * 1.1 * design_rpm / 60.0, f"{eo}EO", fontsize=7, va="center")
    ax.axvspan(80, 105, color="0.9", zorder=0, label="operating range")
    ax.set_xlabel("rotor speed [% design]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_ylim(0, 1.2 * max(f.max() for f in table.values()))
    ax.set_xlim(0, 115)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Campbell diagram, free-vortex Rotor 67 blade (Ti-6Al-4V)")
    fig.tight_layout()
    fig.savefig("figures/campbell.png", dpi=150)
    print("\nSaved figures/campbell.png")