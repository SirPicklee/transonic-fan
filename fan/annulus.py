"""
Annulus sizing: how big must the fan be to pass a given mass flow?

Compressible continuity through the inlet annulus, with uniform axial inflow
(no inlet swirl) and an optional blockage factor for the wall boundary layers:

    m = B * A * p0 / sqrt(R T0) * Q(M),
    Q(M) = sqrt(gamma) M (1 + (gamma-1)/2 M^2)^(-(gamma+1) / (2 (gamma-1)))

    A = pi r_tip^2 (1 - nu^2),   nu = r_hub / r_tip  (hub-to-tip ratio)

Units: SI (kg/s, Pa, K, m, rad/s).
"""

from dataclasses import dataclass
import math

from scipy.optimize import brentq

from fan.velocity_triangles import GAMMA, R_AIR, Triangle

# Exponent of the mass flow function
_EXP = -(GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))


def flow_function(mach: float) -> float:
    """Non-dimensional mass flow Q(M) = m sqrt(R T0) / (A p0). Maximum (choke) at M = 1."""
    return math.sqrt(GAMMA) * mach * (1.0 + 0.5 * (GAMMA - 1.0) * mach ** 2) ** _EXP


Q_CHOKE = flow_function(1.0)


def mach_from_flow_function(q: float) -> float:
    """Subsonic Mach number for a given flow function value."""
    if not 0.0 < q < Q_CHOKE:
        raise ValueError(f"flow function {q:.4f} outside (0, {Q_CHOKE:.4f}): annulus choked")
    return brentq(lambda m: flow_function(m) - q, 1e-9, 1.0)


@dataclass(frozen=True)
class Annulus:
    r_tip: float     # casing radius [m]
    r_hub: float     # hub radius [m]

    @property
    def hub_tip_ratio(self) -> float:
        return self.r_hub / self.r_tip

    @property
    def area(self) -> float:
        return math.pi * (self.r_tip ** 2 - self.r_hub ** 2)

    @property
    def r_mean(self) -> float:
        """Radius that splits the annulus into two equal areas [m]."""
        return math.sqrt(0.5 * (self.r_tip ** 2 + self.r_hub ** 2))


def size_annulus(mass_flow: float, axial_mach: float, hub_tip_ratio: float,
                 p0: float = 101325.0, t0: float = 288.15,
                 blockage: float = 1.0) -> Annulus:
    """Tip and hub radius that pass `mass_flow` at the given inlet axial Mach number."""
    area = mass_flow * math.sqrt(R_AIR * t0) / (blockage * p0 * flow_function(axial_mach))
    r_tip = math.sqrt(area / (math.pi * (1.0 - hub_tip_ratio ** 2)))
    return Annulus(r_tip, hub_tip_ratio * r_tip)


def axial_mach(annulus: Annulus, mass_flow: float, p0: float = 101325.0,
               t0: float = 288.15, blockage: float = 1.0) -> float:
    """Inlet axial Mach number of a given annulus at a given mass flow."""
    q = mass_flow * math.sqrt(R_AIR * t0) / (blockage * annulus.area * p0)
    return mach_from_flow_function(q)


def axial_velocity(mach: float, t0: float = 288.15) -> float:
    """Velocity [m/s] from Mach number and total temperature."""
    t = t0 / (1.0 + 0.5 * (GAMMA - 1.0) * mach ** 2)
    return mach * math.sqrt(GAMMA * R_AIR * t)


def inlet_triangle(annulus: Annulus, radius: float, rpm: float, mass_flow: float,
                   p0: float = 101325.0, t0: float = 288.15,
                   blockage: float = 1.0) -> Triangle:
    """Rotor inlet velocity triangle at a given radius, axial inflow without swirl."""
    cx = axial_velocity(axial_mach(annulus, mass_flow, p0, t0, blockage), t0)
    u = rpm * 2.0 * math.pi / 60.0 * radius
    return Triangle(u, cx, 0.0)


# NASA Rotor 67 design point (Strazisar et al., NASA TP-2879, 1989)
ROTOR67 = {
    "mass_flow": 33.25,             # [kg/s]
    "rpm": 16043.0,
    "pressure_ratio": 1.63,
    "tip_relative_mach": 1.38,
    "annulus_inlet": Annulus(r_tip=0.514 / 2, r_hub=0.193 / 2),
}


if __name__ == "__main__":
    T0 = 288.15
    r67 = ROTOR67
    ann = r67["annulus_inlet"]

    print("NASA Rotor 67 inlet, design point (sea-level standard inlet)")
    print(f"  hub/tip ratio          {ann.hub_tip_ratio:.3f}")
    print(f"  annulus area           {ann.area:.4f} m^2")

    for blockage in (1.00, 0.98):
        tip = inlet_triangle(ann, ann.r_tip, r67["rpm"], r67["mass_flow"], t0=T0,
                             blockage=blockage)
        m_ax = axial_mach(ann, r67["mass_flow"], t0=T0, blockage=blockage)
        print(f"\n  blockage {blockage:.2f}")
        print(f"    axial Mach           {m_ax:.3f}")
        print(f"    axial velocity       {tip.cx:.1f} m/s")
        print(f"    tip speed            {tip.u:.1f} m/s")
        print(f"    tip relative Mach    {tip.mach_rel(T0):.3f}   "
              f"(NASA: {r67['tip_relative_mach']})")

    # Inverse problem: size a new annulus for the same flow and axial Mach
    m_ax = axial_mach(ann, r67["mass_flow"], t0=T0)
    new = size_annulus(r67["mass_flow"], m_ax, ann.hub_tip_ratio, t0=T0)
    print(f"\nInverse check: sized tip diameter {2 * new.r_tip * 100:.2f} cm "
          f"(NASA: {2 * ann.r_tip * 100:.2f} cm)")