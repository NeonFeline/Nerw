"""Render one overview figure of a generated run into data/showcase.

Picks the richest run in a dataset (most security classes, then most classes
overall) and draws three panels: the band-passed waterfall an operator would
see, the per-pixel labels, and the per-class spectra the classifier has to
separate.

    uv run python make_showcase.py --data data/synthetic/v3 --out data/showcase
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import label as cc_label
from scipy.signal import butter, filtfilt, welch

import simulate_das as sd

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
# Reference categorical slots 1-3 (documented as all-pairs validated, both modes)
GROUP_COLOR = {"traffic": "#2a78d6", "nuisance": "#eb6834", "security": "#1baf7a"}
GROUP_OF = {"train": "traffic", "road_vehicle": "traffic", "wheel_flat": "traffic",
            "footsteps": "nuisance", "burst": "nuisance"}
for name in sd.SECURITY_CLASSES:
    GROUP_OF[name] = "security"
# slots 1-6 for the spectra (adjacent pairlist)
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--data", type=Path, default=Path("data/synthetic/v3"))
_ap.add_argument("--out", type=Path, default=Path("data/showcase"))
_ap.add_argument("--run", type=Path, default=None, help="use this run instead of the richest")
_args = _ap.parse_args()

root = _args.data
best, best_key = None, (-1, -1)
for d in sorted(root.glob("*/*/")):
    if not (d / "das.h5").exists():
        continue
    with h5py.File(d / "das.h5", "r") as f:
        m = f["labels/class_mask"][:]
    present = {int(c) for c in np.unique(m)} - {0, 255}
    sec = {c for c in present if sd.CLASS_NAMES.get(c) in sd.SECURITY_CLASSES}
    if (len(sec), len(present)) > best_key:
        best, best_key = d, (len(sec), len(present))

run = _args.run or best
with h5py.File(run / "das.h5", "r") as f:
    traces, mask = f["traces"][:], f["labels/class_mask"][:]
    cfg = json.loads(f.attrs["meta_json"])["config"]
with h5py.File(run / "components.h5", "r") as f:
    signal, background = f["signal"][:], f["background"][:]
fs = cfg["fs"]
ch = cfg["x_offset"] + (np.arange(traces.shape[1]) + 0.5) * cfg["channel_spacing"]
t = np.arange(traces.shape[0]) / fs
extent = [t[0], t[-1], ch[0], ch[-1]]

fig = plt.figure(figsize=(12.5, 13.5), facecolor=SURFACE)
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.15, 0.95], hspace=0.28,
                      left=0.085, right=0.975, top=0.945, bottom=0.055)
gs.update(hspace=0.36)

# -- A: what an operator sees ------------------------------------------------
b, a = butter(4, (5 / (fs / 2), min(200.0, 0.45 * fs) / (fs / 2)), btype="band")
bp = filtfilt(b, a, traces, axis=0)
db = 20 * np.log10(np.abs(bp) / (np.percentile(np.abs(bp), 99.5) + 1e-30) + 1e-6)
ax = fig.add_subplot(gs[0], facecolor=SURFACE)
im = ax.imshow(db.T, aspect="auto", origin="lower", extent=extent, cmap="magma",
               vmin=-36, vmax=12, interpolation="nearest")
ax.set_title(f"Band-passed strain rate, 5-200 Hz  ·  {run.parent.name}/{run.name}  ·  "
             f"c₀={cfg['wave_speed']:.0f} m/s, D={cfg['damping_ratio']:.3f}",
             color=INK, fontsize=12, pad=9, loc="left")
ax.set_ylabel("distance along fibre (m)", color=INK2, fontsize=10)
cb = fig.colorbar(im, ax=ax, pad=0.012)
cb.set_label("dB rel. p99.5", color=INK2, fontsize=9)
cb.ax.tick_params(colors=INK2, labelsize=8)
ax.tick_params(labelbottom=False)

# -- B: labels, coloured by group, every region named ------------------------
axm = fig.add_subplot(gs[1], sharex=ax, facecolor=SURFACE)
rgb = np.ones(mask.shape + (3,), dtype=float)
rgb[mask == 255] = matplotlib.colors.to_rgb("#d6d5cf")
for name, group in GROUP_OF.items():
    cid = sd.CLASS_IDS.get(name)
    if cid is not None:
        rgb[mask == cid] = matplotlib.colors.to_rgb(GROUP_COLOR[group])
axm.imshow(np.transpose(rgb, (1, 0, 2)), aspect="auto", origin="lower",
           extent=extent, interpolation="nearest")
for cid in sorted(set(np.unique(mask)) - {0, 255}):
    name = sd.CLASS_NAMES[int(cid)]
    parts, n = cc_label(mask == cid)
    if n == 0:
        continue
    sizes = np.bincount(parts.ravel())
    sizes[0] = 0
    ti, ci = np.nonzero(parts == int(np.argmax(sizes)))   # label each class once
    axm.annotate(name, (t[int(np.median(ti))], ch[int(np.median(ci))]),
                 color=INK, fontsize=10, ha="center", va="center", weight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="#d6d5cf", alpha=0.92))
axm.set_title("Per-pixel labels, derived from signal-over-background (colour = group, "
              "text = class)", color=INK, fontsize=12, pad=9, loc="left")
axm.set_ylabel("distance along fibre (m)", color=INK2, fontsize=10)
axm.set_xlabel("time (s)", color=INK2, fontsize=10)
axm.legend(handles=[Patch(facecolor=GROUP_COLOR[g], label=g) for g in GROUP_COLOR]
           + [Patch(facecolor="#d6d5cf", label="ignore (unknowable)"),
              Patch(facecolor="white", ec=INK2, label="background")],
           loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False,
           fontsize=9, ncol=5)

# -- C: the spectra the classifier has to separate ---------------------------
axs = fig.add_subplot(gs[2], facecolor=SURFACE)
series, k = [], 0
freqs, bgp = welch(background, fs=fs, nperseg=4096, axis=0)
axs.semilogx(freqs, 10 * np.log10(bgp.mean(axis=1) + 1e-40), color=INK2, lw=2,
             label="background", zorder=1)
for cid in sorted(set(np.unique(mask)) - {0, 255}):
    name = sd.CLASS_NAMES[int(cid)]
    cells = mask == cid
    channels = np.flatnonzero(cells.any(axis=0))
    rows = np.flatnonzero(cells.any(axis=1))
    if channels.size < 3 or rows.size < 2048:
        continue
    seg = signal[rows[0]:rows[-1] + 1][:, channels]
    f2, p = welch(seg, fs=fs, nperseg=min(4096, seg.shape[0]), axis=0)
    curve = 10 * np.log10(p.mean(axis=1) + 1e-40)
    color = LINE_COLORS[k % len(LINE_COLORS)]
    axs.semilogx(f2, curve, color=color, lw=2, label=name, zorder=2)
    series.append((name, f2, curve, color))
    k += 1

# Selective direct labels: name only the series the eye can follow to a
# distinctive feature; the legend carries the rest.
for name, anchor in (("cable_cut", 260.0), ("vehicle_stop", 3.0), ("train", 30.0)):
    match = next((s for s in series if s[0] == name), None)
    if match is None:
        continue
    _, f2, curve, color = match
    i = int(np.argmin(np.abs(f2 - anchor)))
    axs.annotate(name, (f2[i], curve[i]), color=INK, fontsize=10, weight="bold",
                 xytext=(6, 10), textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="#d6d5cf", alpha=0.92),
                 arrowprops=dict(arrowstyle="-", color=color, lw=1.4))
axs.set_xlim(0.5, 500)
axs.set_title("Mean spectrum of the clean event field, per class, against the background",
              color=INK, fontsize=12, pad=9, loc="left")
axs.set_xlabel("frequency (Hz)", color=INK2, fontsize=10)
axs.set_ylabel("dB/Hz", color=INK2, fontsize=10)
axs.grid(True, which="both", color="#e8e7e1", lw=0.7, zorder=0)
axs.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor="#d6d5cf",
           fontsize=9, ncol=3)

for axis in (ax, axm, axs):
    axis.tick_params(colors=INK2, labelsize=9)
    for spine in axis.spines.values():
        spine.set_color("#d6d5cf")

_args.out.mkdir(parents=True, exist_ok=True)
out = _args.out / "das_overview.png"
fig.savefig(out, dpi=135, facecolor=SURFACE)
print("run:", run)
print("classes:", sorted(sd.CLASS_NAMES[int(c)] for c in set(np.unique(mask)) - {0, 255}))
print("wrote", out)
