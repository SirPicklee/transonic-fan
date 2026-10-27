"""
Three-dimensional blade: stacking of the sections and export.

Each section lies on a streamline surface. Its radius varies linearly from
the inlet radius r1 at the leading edge to the exit radius r2 at the
trailing edge, so the blade fills the contracting Rotor 67 annulus.

Stacking: the area centroids of all sections are placed on one radial line
(the stacking axis), as is usual for fan blades, so that centrifugal force
does not bend the blade.

Coordinates: x axial (flow direction), rotation about the x axis, blade
moving in +theta. Exports: a CSV of section points and an ASCII STL surface
(opens in 3D viewers; GitHub shows STL files as an interactive model).
Units in the files: mm.
"""

from dataclasses import dataclass

import numpy as np

from fan.annulus import ROTOR67, Annulus
from fan.blade_angles import metal_angles
from fan.blade_sections import chord_from_solidity, circular_arc_section
from fan.losses import analyze, rotor67_solidity
from fan.radial_equilibrium import design_for_pressure_ratio


def section_loop(section: dict) -> tuple[np.ndarray, np.ndarray]:
    """Closed outline: upper surface LE -> TE, then lower surface TE -> LE,
    without repeating the shared edge points."""
    xu, yu = section["upper"]
    xl, yl = section["lower"]
    x = np.concatenate([xu, xl[-2:0:-1]])
    y = np.concatenate([yu, yl[-2:0:-1]])
    return x, y


def polygon_centroid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Area centroid of a closed polygon (shoelace formula)."""
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * cross.sum()
    cx = ((x + x1) * cross).sum() / (6.0 * area)
    cy = ((y + y1) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


@dataclass
class Blade:
    sections: list        # one (M, 3) array of x, y, z [m] per section, closed loop
    span: np.ndarray      # span fraction of each section


def stack_blade(span, r1, r2, camber, stagger, chord, thickness_ratio,
                n_points: int = 101) -> Blade:
    """Build the sections, put their centroids on the stacking axis and wrap
    them onto their streamline surfaces."""
    sections = []
    for k in range(len(span)):
        sec = circular_arc_section(camber[k], stagger[k], chord[k],
                                   thickness_ratio[k], n_points)
        x, y = section_loop(sec)
        cx, cy = polygon_centroid(x, y)
        x, y = x - cx, y - cy                        # centroid on the stacking axis

        x_cam = sec["camber"][0]
        x_le, x_te = x_cam[0] - cx, x_cam[-1] - cx
        frac = np.clip((x - x_le) / (x_te - x_le), 0.0, 1.0)
        r = r1[k] + (r2[k] - r1[k]) * frac           # streamline radius
        theta = y / r                                # unrolled y = r * theta
        sections.append(np.column_stack([x, r * np.sin(theta), r * np.cos(theta)]))
    return Blade(sections, np.asarray(span, dtype=float))


def rotor67_free_vortex_blade(n_sections: int = 11, n_points: int = 101) -> Blade:
    """The free-vortex rotor designed in the previous modules, as a 3D blade."""
    inlet = ROTOR67["annulus_inlet"]
    exit_ = Annulus(r_tip=0.485 / 2, r_hub=0.232 / 2)
    rpm, mdot = ROTOR67["rpm"], ROTOR67["mass_flow"]
    guess = design_for_pressure_ratio(inlet, exit_, rpm, mdot, 1.63, -1, 0.92)
    c_theta_m = guess.c_theta2[0] * (exit_.r_mean / guess.r2[0]) ** -1
    sol, *_ = analyze(inlet, exit_, rpm, mdot, c_theta_m, -1)

    sigma = rotor67_solidity(sol.span)
    _, _, camber, stagger, _ = metal_angles(sol.beta1, sol.beta2, sigma)
    chord = chord_from_solidity(sigma, 0.5 * (sol.r1 + sol.r2), 22)
    t_max = 0.10 + (0.03 - 0.10) * sol.span      # assumed: 10 % at hub, 3 % at tip

    idx = np.round(np.linspace(0, len(sol.span) - 1, n_sections)).astype(int)
    return stack_blade(sol.span[idx], sol.r1[idx], sol.r2[idx], camber[idx],
                       stagger[idx], chord[idx], t_max[idx], n_points)


def write_stl(blade: Blade, path: str, name: str = "rotor_blade") -> int:
    """ASCII STL of the blade surface with hub and tip caps, in mm.
    Returns the number of triangles."""
    rings = [s * 1000.0 for s in blade.sections]
    m = len(rings[0])
    tris = []
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(m):
            j = (i + 1) % m
            tris.append((a[i], a[j], b[j]))
            tris.append((a[i], b[j], b[i]))
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        c = ring.mean(axis=0)
        for i in range(m):
            j = (i + 1) % m
            tris.append((c, ring[j], ring[i]) if flip else (c, ring[i], ring[j]))

    with open(path, "w") as f:
        f.write(f"solid {name}\n")
        for p0, p1, p2 in tris:
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
            for p in (p0, p1, p2):
                f.write(f"      vertex {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            f.write("    endloop\n  endfacet\n")
        f.write(f"endsolid {name}\n")
    return len(tris)


def write_sections_csv(blade: Blade, path: str) -> None:
    """Section points in mm: section index, span fraction, point index, x, y, z."""
    with open(path, "w") as f:
        f.write("section,span,point,x_mm,y_mm,z_mm\n")
        for k, (s, pts) in enumerate(zip(blade.span, blade.sections)):
            for i, p in enumerate(pts * 1000.0):
                f.write(f"{k},{s:.4f},{i},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    blade = rotor67_free_vortex_blade()
    n_tri = write_stl(blade, "figures/rotor67_free_vortex_blade.stl")
    write_sections_csv(blade, "figures/rotor67_free_vortex_sections.csv")
    print(f"{len(blade.sections)} sections, {len(blade.sections[0])} points each, "
          f"{n_tri} triangles")
    for k in (0, len(blade.sections) // 2, len(blade.sections) - 1):
        pts = blade.sections[k]
        r = np.hypot(pts[:, 1], pts[:, 2])
        print(f"span {blade.span[k]:.2f}: radius {r.min() * 1000:6.1f} to "
              f"{r.max() * 1000:6.1f} mm, axial {pts[:, 0].min() * 1000:6.1f} to "
              f"{pts[:, 0].max() * 1000:6.1f} mm")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    for pts in blade.sections:
        loop = np.vstack([pts, pts[:1]]) * 1000.0
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], "b-", lw=0.8)
    for i in range(0, len(blade.sections[0]), 10):
        line = np.array([s[i] for s in blade.sections]) * 1000.0
        ax.plot(line[:, 0], line[:, 1], line[:, 2], "k-", lw=0.4)
    all_pts = np.vstack(blade.sections) * 1000.0
    ax.set_box_aspect(np.ptp(all_pts, axis=0))
    ax.set_xlabel("axial [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=15, azim=-60)
    ax.set_title("Free-vortex rotor blade in the Rotor 67 annulus")
    fig.savefig("figures/blade_3d.png", dpi=150)
    print("Saved figures/blade_3d.png, figures/rotor67_free_vortex_blade.stl, "
          "figures/rotor67_free_vortex_sections.csv")