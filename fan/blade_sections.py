"""
Blade sections: camber line with a lens-shaped thickness.

Circular-arc section (circular_arc_section): built in its own chord frame
and rotated by the stagger angle:

  camber line   circular arc, direction angle phi from -theta/2 (leading
                edge) to +theta/2 (trailing edge), radius R = c / (2 sin(theta/2))
                    x = c/2 + R sin(phi),   y = R (cos(theta/2) - cos(phi))
  thickness     t(s) = 4 t_max s (1 - s), s = x / c, laid off normal to
                the camber line on both sides (double-circular-arc-like
                for a thin section). The edges are sharp.

Two-arc section (mca_section): the camber line is made of two tangent
circular arcs, with `front_fraction` of the camber in the front part of the
blade. Transonic fan blades put little camber ahead of the passage shock to
limit the supersonic expansion along the suction surface (see shock.py).
front_fraction = 0.5 gives back the circular arc.

Coordinates: X axial, Y tangential (direction of blade motion), metres,
origin at mid-chord. Direction angles from axial, so the camber line leaves
the leading edge at kappa1 and the trailing edge at kappa2.
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


def two_arc_camber(camber_deg: float, front_fraction: float = 0.5,
                   transition: float = 0.5, n_points: int = 101):
    """Unit-chord camber line of two tangent circular arcs: the direction
    angle grows linearly with arc length, by front_fraction * camber up to
    `transition` and by the rest after it. Returns x, y and the direction
    angle phi [rad], all in the chord frame (leading edge at the origin,
    trailing edge at (1, 0))."""
    theta = math.radians(camber_deg)
    xi = np.linspace(0.0, 1.0, n_points)                     # arc length fraction
    t_front = front_fraction * theta
    phi = np.where(xi <= transition, t_front * xi / transition,
                   t_front + (theta - t_front) * (xi - transition) / (1.0 - transition))
    phi_mid = 0.5 * (phi[1:] + phi[:-1])
    ds = 1.0 / (n_points - 1)
    x = np.concatenate([[0.0], np.cumsum(np.cos(phi_mid) * ds)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(phi_mid) * ds)])
    alpha = math.atan2(y[-1], x[-1])                         # chord direction
    length = math.hypot(x[-1], y[-1])
    ca, sa = math.cos(alpha), math.sin(alpha)
    return (x * ca + y * sa) / length, (-x * sa + y * ca) / length, phi - alpha


def max_camber_position(front_fraction: float, transition: float = 0.5,
                        camber_deg: float = 10.0) -> float:
    """Chordwise position a/c of maximum camber of the two-arc camber line."""
    x, y, _ = two_arc_camber(camber_deg, front_fraction, transition, 401)
    return float(x[np.argmin(y)])


def mca_section(kappa1_deg: float, kappa2_deg: float, chord: float,
                thickness_ratio: float, front_fraction: float = 0.5,
                transition: float = 0.5, n_points: int = 101) -> dict:
    """Section with a two-arc camber line, placed by its metal angles.

    Returns {"camber", "upper", "lower": (X, Y), "stagger": chord angle [deg]}.
    """
    x, y, phi = two_arc_camber(kappa2_deg - kappa1_deg, front_fraction,
                               transition, n_points)
    gamma = math.radians(kappa1_deg) - phi[0]                # chord (stagger) angle
    s = np.clip(x, 0.0, 1.0)
    half = 2.0 * thickness_ratio * s * (1.0 - s)             # unit chord
    nx, ny = -np.sin(phi), np.cos(phi)

    def to_blade_frame(xc, yc):
        xc, yc = (xc - 0.5) * chord, yc * chord
        return (xc * math.cos(gamma) - yc * math.sin(gamma),
                xc * math.sin(gamma) + yc * math.cos(gamma))

    return {
        "camber": to_blade_frame(x, y),
        "upper": to_blade_frame(x + half * nx, y + half * ny),
        "lower": to_blade_frame(x - half * nx, y - half * ny),
        "stagger": math.degrees(gamma),
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