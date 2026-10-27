"""
Blade sections: circular-arc camber line with a lens-shaped thickness.

Each section is built in its own chord frame and then rotated by the
stagger angle:

  camber line   circular arc, direction angle phi from -theta/2 (leading
                edge) to +theta/2 (trailing edge), radius R = c / (2 sin(theta/2))
                    x = c/2 + R sin(phi),   y = R (cos(theta/2) - cos(phi))
  thickness     t(s) = 4 t_max s (1 - s), s = x / c, laid off normal to
                the camber line on both sides. For a thin section this is
                the double-circular-arc (DCA) profile, a classic transonic
                fan section. The edges are sharp; edge radii come later.

Coordinates: X axial, Y tangential (direction of blade motion), metres,
origin at mid-chord. Direction angles from axial, so the camber line
leaves the leading edge at kappa1 = stagger - theta/2 and the trailing
edge at kappa2 = stagger + theta/2, as in blade_angles.py.
"""

import math

import numpy as np


def chord_from_solidity(solidity, radius, n_blades: int):
    """Chord [m] from solidity sigma = c / pitch, pitch = 2 pi r / Z."""
    return solidity * 2.0 * np.pi * radius / n_blades


def circular_arc_section(camber_deg: float, stagger_deg: float, chord: float,
                         thickness_ratio: float, n_points: int = 101) -> dict:
    """Camber line, upper and lower surface of one section.

    Returns {"camber": (X, Y), "upper": (X, Y), "lower": (X, Y)}.
    """
    theta = math.radians(camber_deg)
    gamma = math.radians(stagger_deg)

    if abs(theta) < 1e-9:                       # flat plate
        phi = np.zeros(n_points)
        x = np.linspace(0.0, chord, n_points)
        y = np.zeros(n_points)
    else:
        radius = chord / (2.0 * math.sin(0.5 * theta))
        phi = np.linspace(-0.5 * theta, 0.5 * theta, n_points)
        x = 0.5 * chord + radius * np.sin(phi)
        y = radius * (math.cos(0.5 * theta) - np.cos(phi))

    s = np.clip(x / chord, 0.0, 1.0)
    half = 2.0 * thickness_ratio * chord * s * (1.0 - s)    # half thickness
    nx, ny = -np.sin(phi), np.cos(phi)                      # unit normal

    def to_blade_frame(xc, yc):
        xc = xc - 0.5 * chord                                # origin at mid-chord
        return (xc * math.cos(gamma) - yc * math.sin(gamma),
                xc * math.sin(gamma) + yc * math.cos(gamma))

    return {
        "camber": to_blade_frame(x, y),
        "upper": to_blade_frame(x + half * nx, y + half * ny),
        "lower": to_blade_frame(x - half * nx, y - half * ny),
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from fan.annulus import ROTOR67, Annulus
    from fan.blade_angles import metal_angles
    from fan.losses import analyze, rotor67_solidity
    from fan.radial_equilibrium import design_for_pressure_ratio

    N_BLADES = 22
    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)

    sigma = rotor67_solidity(sol.span)
    _, _, camber, stagger, _ = metal_angles(sol.beta1, sol.beta2, sigma)
    radius = 0.5 * (sol.r1 + sol.r2)
    chord = chord_from_solidity(sigma, radius, N_BLADES)
    t_max = 0.10 + (0.03 - 0.10) * sol.span      # assumed: 10 % at hub, 3 % at tip

    picks = [0, 25, 50, 75, 100]
    print(f"{'span':>5} {'r [mm]':>7} {'chord [mm]':>11} {'t_max [mm]':>11} "
          f"{'camber':>7} {'stagger':>8}")
    fig, axes = plt.subplots(1, len(picks), figsize=(15, 4.5))
    for ax, i in zip(axes, picks):
        print(f"{sol.span[i]:5.2f} {radius[i] * 1000:7.1f} {chord[i] * 1000:11.1f} "
              f"{t_max[i] * chord[i] * 1000:11.2f} {camber[i]:7.1f} {stagger[i]:8.1f}")
        sec = circular_arc_section(camber[i], stagger[i], chord[i], t_max[i])
        for key, style in (("upper", "b-"), ("lower", "b-"), ("camber", "k--")):
            xs, ys = sec[key]
            ax.plot(xs * 1000, ys * 1000, style, lw=1 if key == "camber" else 1.5)
        ax.set_title(f"span {sol.span[i]:.2f}\ncamber {camber[i]:.0f}°, "
                     f"stagger {stagger[i]:.0f}°", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("axial [mm]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("tangential [mm] (blade motion +)")

    span_len = (exit_.r_tip + inlet.r_tip) / 2 - (exit_.r_hub + inlet.r_hub) / 2
    print(f"\nMean span {span_len * 1000:.0f} mm, span / mean chord = "
          f"{span_len / chord.mean():.2f} (NASA aspect ratio 1.56, based on root axial chord)")
    fig.suptitle("Rotor sections (free vortex, circular-arc camber, DCA-like thickness)")
    fig.tight_layout()
    fig.savefig("figures/blade_sections.png", dpi=150)
    print("Saved figures/blade_sections.png")