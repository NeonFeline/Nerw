"""Write DATA/STATIONS for a horizontal DAS fibre, one receiver per channel.

SPECFEM2D reads ``station network x z elevation burial`` with z measured up from
the bottom of the mesh (burial must be 0), so a fibre d metres below a flat free
surface at z_top sits at z = z_top - d.
"""

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--x-start", type=float, default=10.0, help="first channel x (m)")
parser.add_argument("--x-end", type=float, default=90.0, help="last channel x (m)")
parser.add_argument("--spacing", type=float, default=1.0, help="channel spacing (m)")
parser.add_argument("--depth", type=float, default=1.0, help="fibre depth below the surface (m)")
parser.add_argument("--surface-z", type=float, default=50.0, help="z of the free surface (m)")
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "DATA" / "STATIONS")
args = parser.parse_args()

n = int(round((args.x_end - args.x_start) / args.spacing)) + 1
x = args.x_start + args.spacing * np.arange(n)
z = args.surface_z - args.depth
args.out.write_text("".join(
    f"S{i + 1:04d}    AA {xi:12.4f} {z:12.4f}    0.0    0.0\n" for i, xi in enumerate(x)
))
print(f"{args.out}: {n} channels, x = {x[0]:g}..{x[-1]:g} m, z = {z:g} m")
