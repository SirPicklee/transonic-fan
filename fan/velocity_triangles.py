"""
Velocity triangles of an axial fan or compressor rotor.

Conventions
-----------
- Angles are measured from the axial direction, in degrees.
- The rotor blade moves in the +theta direction with speed U = omega * r.
- Absolute velocity C, relative velocity W (seen by the blade): W = C - U.
- Station 1: rotor inlet, station 2: rotor outlet, at the same radius.

Units: SI (m/s, K, J/kg).
"""

from dataclasses import dataclass
import math

GAMMA = 1.4      # ratio of specific heats for air
R_AIR = 287.05   # gas constant of air [J/(kg K)]
CP = GAMMA * R_AIR / (GAMMA - 1.0)   # specific heat at constant pressure [J/(kg K)]


@dataclass(frozen=True)
class Triangle:
    """Velocity triangle at one station."""
    u: float        # blade speed [m/s]
    cx: float       # axial velocity [m/s]
    c_theta: float  # absolute tangential velocity (swirl) [m/s]

    @property
    def w_theta(self) -> float:
        """Relative tangential velocity [m/s]."""
        return self.c_theta - self.u

    @property
    def c(self) -> float:
        """Absolute velocity magnitude [m/s]."""
        return math.hypot(self.cx, self.c_theta)

    @property
    def w(self) -> float:
        """Relative velocity magnitude [m/s]."""
        return math.hypot(self.cx, self.w_theta)

    @property
    def alpha(self) -> float:
        """Absolute flow angle from axial [deg]."""
        return math.degrees(math.atan2(self.c_theta, self.cx))

    @property
    def beta(self) -> float:
        """Relative flow angle from axial [deg] (negative: against rotation)."""
        return math.degrees(math.atan2(self.w_theta, self.cx))

    def static_temperature(self, t0: float) -> float:
        """Static temperature from total temperature T0 [K]."""
        return t0 - self.c ** 2 / (2.0 * CP)

    def mach_abs(self, t0: float) -> float:
        """Absolute Mach number."""
        return self.c / math.sqrt(GAMMA * R_AIR * self.static_temperature(t0))

    def mach_rel(self, t0: float) -> float:
        """Relative Mach number: what the blade sees. Above 1 at a transonic fan tip."""
        return self.w / math.sqrt(GAMMA * R_AIR * self.static_temperature(t0))


@dataclass(frozen=True)
class RotorStage:
    """Rotor inlet and outlet triangles at one radius."""
    inlet: Triangle
    outlet: Triangle

    @property
    def work(self) -> float:
        """Specific work from the Euler equation, dh0 = U (C_theta2 - C_theta1) [J/kg]."""
        return self.inlet.u * (self.outlet.c_theta - self.inlet.c_theta)

    @property
    def loading(self) -> float:
        """Stage loading coefficient psi = dh0 / U^2."""
        return self.work / self.inlet.u ** 2

    @property
    def flow_coefficient(self) -> float:
        """Flow coefficient phi = Cx1 / U."""
        return self.inlet.cx / self.inlet.u

    @property
    def de_haller(self) -> float:
        """de Haller ratio W2 / W1. Below about 0.72 the rotor diffuses too much."""
        return self.outlet.w / self.inlet.w

    def diffusion_factor(self, solidity: float) -> float:
        """Lieblein diffusion factor
        DF = 1 - W2/W1 + |W_theta1 - W_theta2| / (2 sigma W1),
        with solidity sigma = chord / pitch. Design limit about 0.45 (0.40 at the tip)."""
        dw = abs(self.inlet.w_theta - self.outlet.w_theta)
        return 1.0 - self.de_haller + dw / (2.0 * solidity * self.inlet.w)

    def total_temperature_rise(self) -> float:
        """dT0 = dh0 / cp [K]."""
        return self.work / CP

    def pressure_ratio(self, t01: float, efficiency: float) -> float:
        """Total pressure ratio p02/p01 for a given isentropic efficiency."""
        tr = 1.0 + efficiency * self.total_temperature_rise() / t01
        return tr ** (GAMMA / (GAMMA - 1.0))


def rotor_from_loading(u: float, cx1: float, loading: float,
                       cx2: float | None = None, c_theta1: float = 0.0) -> RotorStage:
    """Build a rotor from blade speed, axial velocity and loading coefficient psi.

    The outlet swirl follows from the Euler equation: C_theta2 = C_theta1 + psi U.
    Without inlet guide vanes (usual for a fan) the inlet swirl is zero.
    """
    cx2 = cx1 if cx2 is None else cx2
    inlet = Triangle(u, cx1, c_theta1)
    outlet = Triangle(u, cx2, c_theta1 + loading * u)
    return RotorStage(inlet, outlet)


if __name__ == "__main__":
    # Example: a transonic fan rotor tip, sea-level static inlet
    T01 = 288.15
    rotor = rotor_from_loading(u=430.0, cx1=190.0, loading=0.25)
    i, o = rotor.inlet, rotor.outlet

    print("Station      C [m/s]  W [m/s]  alpha [deg]  beta [deg]  M_rel")
    print(f"inlet  1  {i.c:9.1f} {i.w:8.1f} {i.alpha:11.1f} {i.beta:11.1f} {i.mach_rel(T01):6.2f}")
    t02 = T01 + rotor.total_temperature_rise()
    print(f"outlet 2  {o.c:9.1f} {o.w:8.1f} {o.alpha:11.1f} {o.beta:11.1f} {o.mach_rel(t02):6.2f}")
    print()
    print(f"Specific work         {rotor.work / 1000:7.2f} kJ/kg")
    print(f"Total temp. rise      {rotor.total_temperature_rise():7.2f} K")
    print(f"Flow coefficient phi  {rotor.flow_coefficient:7.3f}")
    print(f"Loading psi           {rotor.loading:7.3f}")
    print(f"de Haller W2/W1       {rotor.de_haller:7.3f}   (limit ~0.72)")
    print(f"Diffusion factor      {rotor.diffusion_factor(solidity=1.3):7.3f}   (limit ~0.40 at tip)")
    print(f"Pressure ratio        {rotor.pressure_ratio(T01, efficiency=0.88):7.3f}   (eta = 0.88)")