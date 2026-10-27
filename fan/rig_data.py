"""
NASA Rotor 67 rig test data (Strazisar et al., NASA TP-2879, 1989).

Transcribed by hand from the page images of tables III(a), III(b), IV and V
of the report (printed pages 34-37), because the text layer of the scanned
report contains OCR errors. The CSV files are in the data/ folder.

Two operating points at design speed: "peak_efficiency" and "near_stall".
"""

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
P_REF = 101325.0     # [Pa]
T_REF = 288.15       # [K]
POINTS = ("peak_efficiency", "near_stall")


def _read_csv(name: str) -> list[dict]:
    """Rows of a CSV file as dictionaries, skipping lines that start with '#'."""
    with open(DATA_DIR / name, newline="") as f:
        lines = [line for line in f if line.strip() and not line.startswith("#")]
    rows = list(csv.reader(lines))
    header, body = rows[0], rows[1:]
    return [dict(zip(header, row)) for row in body]


def load_survey(station: int, point: str) -> dict:
    """Radial survey at station 1 (rotor inlet) or 2 (rotor exit), SI units.
    Keys: radius [m], p0 [Pa], t0 [K], ps [Pa], flow_angle [deg from meridional]."""
    if point not in POINTS:
        raise ValueError(f"unknown operating point {point!r}, expected one of {POINTS}")
    rows = [r for r in _read_csv(f"rotor67_station{station}.csv") if r["point"] == point]

    def col(key):
        return np.array([float(r[key]) for r in rows])

    return {
        "radius": col("radius_cm") / 100.0,
        "p0": col("p0_ratio") * P_REF,
        "t0": col("t0_ratio") * T_REF,
        "ps": col("ps_ratio") * P_REF,
        "flow_angle": col("flow_angle_deg"),
    }


def load_streamlines() -> dict:
    """The 9 design streamlines of table IV, SI units."""
    rows = _read_csv("rotor67_streamlines.csv")

    def col(key):
        return np.array([float(r[key]) for r in rows])

    return {
        "r1": col("r1_cm") / 100.0, "r2": col("r2_cm") / 100.0,
        "area1": col("area1_cm2") / 1e4, "area2": col("area2_cm2") / 1e4,
        "slope1": col("slope1_deg"), "slope2": col("slope2_deg"),
    }


def load_overall(point: str) -> dict:
    """Published overall performance of one operating point."""
    for row in _read_csv("rotor67_overall.csv"):
        if row["point"] == point:
            return {k: float(v) for k, v in row.items() if k != "point"}
    raise ValueError(f"unknown operating point {point!r}, expected one of {POINTS}")