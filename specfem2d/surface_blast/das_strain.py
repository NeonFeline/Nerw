"""Turn SPECFEM2D displacement seismograms into a DAS strain gather.

A straight horizontal fibre measures the axial strain eps_xx averaged over the
gauge length L.  From the displacement UX sampled at the channels,

    eps(x) = (u(x + L/2) - u(x - L/2)) / L = (1/L) * integral of eps_xx over the gauge,

so the gauge difference is not an approximation of the average strain, it *is*
the average.  Its wavenumber response is sinc(k L / 2): zero wherever the
apparent wavelength along the fibre is L / n.

Channels whose gauge would run off either end of the fibre are dropped.  The
result is exact when L/2 is a multiple of the channel spacing.  Otherwise u is
interpolated linearly between channels, which is only accurate for wavelengths
many channel spacings long, so the script warns; record at a spacing that
divides L/2 instead.

Reads DATA/STATIONS, DATA/SOURCE and OUTPUT_FILES/<net>.<sta>.?XX.semd under
--run-dir; writes the waterfall figure and an .npz with the gather.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Optical phase per unit strain per metre of gauge for single-mode fibre read
# at 1550 nm: dphi = 4 pi n xi L eps / lambda, with effective index n = 1.468
# and photo-elastic scaling factor xi = 0.78.
PHASE_PER_STRAIN_METRE = 4 * np.pi * 1.468 * 0.78 / 1550e-9

SURFACE, INK, INK_SECONDARY, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
# blue (compression) <-> neutral grey <-> red (extension)
DIVERGING = LinearSegmentedColormap.from_list("strain", ["#2a78d6", "#f0efec", "#e34948"])


def read_stations(path: Path) -> tuple[list[tuple[str, str]], np.ndarray]:
    """(network, station) and x of every receiver in a SPECFEM2D STATIONS file."""
    ids, x = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0][0] in "#!":
            continue
        ids.append((fields[1], fields[0]))
        x.append(float(fields[2]))
    return ids, np.array(x)


def read_source_x(path: Path) -> float | None:
    """x of the first source in a SPECFEM2D SOURCE file, if there is one."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "xs":
            return float(value.split("#")[0].lower().replace("d", "e"))
    return None


def load_ux(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel x [nch], time t [nt] and UX [nch, nt], ordered along the fibre."""
    ids, x = read_stations(run_dir / "DATA" / "STATIONS")
    traces, t = [], None
    for net, sta in ids:
        # The band letter follows the sample interval (F for <= 1 ms, then C, H, B, ...).
        matches = sorted((run_dir / "OUTPUT_FILES").glob(f"{net}.{sta}.?XX.semd"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one UX trace for {net}.{sta}, found {len(matches)}")
        data = np.loadtxt(matches[0])
        if t is None:
            t = data[:, 0]
        traces.append(data[:, 1])
    order = np.argsort(x, kind="stable")
    return x[order], t, np.stack(traces)[order]


def gauge_strain(ux: np.ndarray, x: np.ndarray, gauge: float) -> tuple[np.ndarray, np.ndarray]:
    """(u(x + L/2) - u(x - L/2)) / L at every channel whose gauge fits on the fibre."""
    half = 0.5 * gauge
    tol = 1e-6 * gauge
    fits = (x - half >= x[0] - tol) & (x + half <= x[-1] + tol)
    if not fits.any():
        raise ValueError(f"gauge length {gauge:g} m does not fit on a {x[-1] - x[0]:g} m fibre")
    centres = x[fits]

    taps = np.concatenate([centres - half, centres + half])
    if np.abs(taps[:, None] - x[None, :]).min(axis=1).max() > tol:
        warnings.warn(f"gauge ends for L = {gauge:g} m fall between channels; u is interpolated "
                      "linearly, which is inaccurate unless the wavelength spans many channels",
                      stacklevel=2)

    def at(xq):
        i = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
        w = ((xq - x[i]) / (x[i + 1] - x[i]))[:, None]
        return ux[i] * (1.0 - w) + ux[i + 1] * w

    return centres, (at(centres + half) - at(centres - half)) / gauge


def strain_units(peak: float) -> tuple[float, str]:
    """Scale and prefix that put the colour limit at 1 or above."""
    for scale, label in ((1e3, "mε"), (1e6, "µε"), (1e9, "nε")):
        if peak * scale >= 1:
            return scale, label
    return 1e12, "pε"


def plot_gather(x, t, gather, title, quantity, unit, source_x, clip, out):
    limit = float(np.percentile(np.abs(gather), clip)) or 1.0
    scale, prefix = strain_units(limit)

    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained", facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    mesh = ax.pcolormesh(x, t, scale * gather.T, shading="nearest", cmap=DIVERGING,
                         vmin=-scale * limit, vmax=scale * limit, rasterized=True)
    ax.set_ylim(t[-1], t[0])
    ax.set_xlabel("Distance along fibre (m)", color=INK_SECONDARY)
    ax.set_ylabel("Time (s)", color=INK_SECONDARY)
    ax.set_title(title, color=INK, loc="left", pad=28)  # clears the source marker
    if source_x is not None and x[0] <= source_x <= x[-1]:
        ax.annotate("source", xy=(source_x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 12), textcoords="offset points", ha="center", color=INK_SECONDARY,
                    arrowprops=dict(arrowstyle="-|>", color=INK_SECONDARY, lw=1))

    # extend arrows: values beyond the clip percentile saturate
    bar = fig.colorbar(mesh, ax=ax, extend="both", pad=0.02)
    bar.set_label(f"{quantity} ({prefix}{unit})", color=INK_SECONDARY)
    for axis in (ax, bar.ax):
        axis.tick_params(colors=INK_SECONDARY)
        for spine in axis.spines.values():
            spine.set_color(AXIS)

    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return fig


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=here, help="directory holding DATA/ and OUTPUT_FILES/")
    parser.add_argument("--gauge", type=float, default=4.0, help="gauge length L in metres")
    parser.add_argument("--rate", action="store_true", help="plot strain rate instead of strain")
    parser.add_argument("--clip", type=float, default=99.5, help="colour limit as a percentile of |value|")
    parser.add_argument("--out", type=Path, help="figure path (default OUTPUT_FILES/das_strain_gather.png)")
    parser.add_argument("--show", action="store_true", help="also open the figure in a window")
    args = parser.parse_args()

    x, t, ux = load_ux(args.run_dir)
    centres, strain = gauge_strain(ux, x, args.gauge)
    gather, quantity, unit = strain, "Strain", ""
    if args.rate:
        gather, quantity, unit = np.gradient(strain, t, axis=1), "Strain rate", "/s"

    outdir = args.run_dir / "OUTPUT_FILES"
    np.savez(outdir / "das_strain.npz", x=centres, t=t, strain=strain, gauge_length=args.gauge,
             phase_rad=PHASE_PER_STRAIN_METRE * args.gauge * strain)
    out = args.out or outdir / "das_strain_gather.png"
    fig = plot_gather(centres, t, gather, f"DAS {quantity.lower()} gather, gauge length {args.gauge:g} m",
                      quantity, unit, read_source_x(args.run_dir / "DATA" / "SOURCE"), args.clip, out)

    peak = np.abs(strain).max()
    print(f"{len(centres)} channels (x = {centres[0]:g}..{centres[-1]:g} m) x {len(t)} samples, "
          f"dt = {t[1] - t[0]:g} s")
    print(f"peak |strain| = {peak:.3e}  ->  {PHASE_PER_STRAIN_METRE * args.gauge * peak:.3e} rad optical phase")
    print(f"wrote {out} and {outdir / 'das_strain.npz'}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
