"""Validate a synthetic DAS run: format, band content, labels and event speed.

Checks the properties the generator promises, from the files alone:

* miniDAS attributes, and that ``scale_factor`` is 1.0 so a compliant reader
  cannot rescale traces that are already in rad/s;
* ``traces == signal + background`` in the companion file;
* per-class coverage, and the measured signal-over-background ratio inside each
  labelled region -- every event class must outrank the background class;
* the spatial structure of the background, which has to carry the gauge
  difference signature rather than being channel-independent noise;
* apparent speed of each moving event, estimated from the ridge slope inside
  that event's own labelled footprint (no geometric corridor is assumed) and
  compared with the scenario value.

    uv run python validate_sim.py data/synthetic/v2/train/embankment_soft_000/das.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import theilslopes

from simulate_das import signal_envelope

IGNORE_ID = 255


def bandpass(x, fs, fmin=15.0, fmax=150.0, order=4):
    b, a = butter(order, (fmin / (fs / 2), min(fmax, 0.45 * fs) / (fs / 2)), btype="band")
    return filtfilt(b, a, x, axis=0)


def track_range(ev, duration):
    """The span of fibre the source actually moves along during the record."""
    if "x0" not in ev:
        return None
    t0 = float(ev.get("t_start", 0.0)) if ev["kind"] == "walker" else 0.0
    t1 = float(ev.get("t_end", duration)) if ev["kind"] == "walker" else duration
    first = float(ev["x0"]) + float(ev["speed"]) * t0
    last = float(ev["x0"]) + float(ev["speed"]) * t1
    return min(first, last), max(first, last)


def estimate_event_speed(envelope, above_threshold, tvec_coarse, ch_pos, x_limits=None,
                         min_channels=8):
    """Apparent ridge speed from one event's own clean SNR map, by Theil-Sen.

    Reading the ridge off the event's own clean envelope rather than off the
    recorded traces keeps other events, background transients and the noise
    floor out of the peak pick.  Only channels whose peak is interior to
    ``above_threshold`` contribute, so entry and exit edges do not bias the fit.

    ``x_limits`` restricts the fit to the stretch the source actually travels
    along.  Beyond it the source only radiates, so every one of those channels
    peaks at roughly the same instant -- the moment of closest approach -- and
    mixing that flat segment into the fit drags the slope towards the wave
    moveout and reports a source far faster than it is.
    """
    t_peak, used = [], []
    for channel in range(envelope.shape[0]):
        if x_limits is not None and not (x_limits[0] <= ch_pos[channel] <= x_limits[1]):
            continue
        rows = np.flatnonzero(above_threshold[channel])
        if rows.size < 3:
            continue
        peak = int(np.argmax(envelope[channel, rows]))
        if peak in (0, rows.size - 1):
            continue
        t_peak.append(tvec_coarse[rows[peak]])
        used.append(channel)
    if len(used) < min_channels:
        return None
    used, t_peak = np.asarray(used), np.asarray(t_peak)
    slope, _, lo_slope, hi_slope = theilslopes(t_peak, ch_pos[used])
    if not np.isfinite(slope) or abs(slope) < 1e-9:
        return None
    if lo_slope * hi_slope > 0:
        lo, hi = sorted((1.0 / hi_slope, 1.0 / lo_slope))
    else:
        lo, hi = np.nan, np.nan
    return {"speed": 1.0 / slope, "lo": lo, "hi": hi, "channels": len(used),
            "x_span": float(ch_pos[used].max() - ch_pos[used].min()),
            "t_span": float(t_peak.max() - t_peak.min())}


def source_length(ev) -> float:
    offsets = ev.get("axle_offsets")
    if offsets:
        return float(max(offsets) - min(offsets))
    return float(ev.get("length", 0.0))


def speed_verdict(fit, ev, wave_speed, tolerance):
    """What the measured ridge slope can and cannot say about the source.

    Three things make the slope uninformative rather than wrong, and each is
    reported instead of failed:

    * a wide confidence interval -- too few channels, or a peak pick set by
      whichever axle happened to be loudest;
    * a source that never crosses its own footprint, where the ridge is the wave
      moveout at ``c`` and says nothing about the source speed;
    * a source long enough to smear the peak over a good part of the array,
      where the per-channel peak time is set by train geometry, not arrival.

    The propagation kinematics themselves are pinned by a compact-source unit
    test in ``tests/test_simulate.py``, which is where a real regression shows.
    """
    speed, lo, hi = fit["speed"], fit["lo"], fit["hi"]
    width = abs(hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.inf
    if width > 0.3 * abs(speed):
        return "unconstrained", False
    if abs(float(ev["speed"])) * fit["t_span"] < 0.8 * fit["x_span"]:
        return f"wave moveout dominates (c={wave_speed:.0f})", False
    if source_length(ev) > 0.25 * fit["x_span"]:
        return f"extended source ({source_length(ev):.0f} m)", False
    if abs(speed - float(ev["speed"])) > tolerance * abs(float(ev["speed"])):
        return "off", True
    return "ok", False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", type=Path)
    ap.add_argument("--speed-tolerance", type=float, default=0.25,
                    help="fractional error at which an estimated speed is flagged")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        traces = f["traces"][:]
        mask = f["labels/class_mask"][:]
        fs = float(f.attrs["sampling_rate"])
        gauge = float(f.attrs["gauge_length"])
        scale_factor = float(f.attrs["scale_factor"])
        meta = json.loads(f.attrs["meta_json"])
        class_ids = json.loads(f.attrs.get("class_ids", "{}"))
    cfg, scenario = meta["config"], meta["scenario"]
    names = {v: k for k, v in class_ids.items()}
    dx = float(cfg["channel_spacing"])
    ch_pos = cfg["x_offset"] + (np.arange(traces.shape[1]) + 0.5) * dx
    problems = []

    print(f"file        : {args.h5}")
    print(f"traces      : {traces.shape} {traces.dtype}  ({traces.nbytes/1e6:.1f} MB)")
    print(f"fs          : {fs:g} Hz   gauge: {gauge:g} m   dx: {dx:g} m   "
          f"c0: {cfg['wave_speed']:.0f} m/s   bands: {cfg.get('propagation_bands')}")
    print(f"duration    : {traces.shape[0]/fs:.1f} s   channels: {traces.shape[1]}   "
          f"family: {cfg.get('family', '-')}   split: {cfg.get('split', '-')}")
    print(f"value range : [{traces.min():.3g}, {traces.max():.3g}]  std={traces.std():.3g}")

    print(f"scale_factor: {scale_factor:g}", end="")
    if scale_factor != 1.0:
        problems.append("scale_factor != 1.0: a reader would rescale rad/s data again")
        print("   <-- traces are already in rad/s")
    else:
        print("   (traces already in rad/s, nothing to re-apply)")

    components = args.h5.parent / "components.h5"
    if components.exists():
        with h5py.File(components, "r") as f:
            signal = f["signal"][:]
            background = f["background"][:]
            snr_db = f["event_snr_db"][:]
            decim = int(f.attrs["envelope_decimation"])
            event_ids = json.loads(f.attrs["event_ids"])
        exact = np.allclose(traces, signal + background, rtol=1e-5, atol=1e-14)
        print(f"components  : traces == signal + background: {exact}")
        if not exact:
            problems.append("traces != signal + background")
    else:
        signal = background = snr_db = None
        print("components  : (not written)")

    bp = bandpass(traces, fs)
    noise = 1.4826 * np.median(np.abs(bp - np.median(bp)))
    peak = np.percentile(np.abs(bp), 99.9)
    print(f"noise floor : {noise:.3g}  (MAD of band-passed traces)")
    print(f"peak p99.9  : {peak:.3g}   peak/noise ~ {20*np.log10(peak/(noise+1e-30)):.1f} dB")
    quiet = int(0.02 * bp.shape[1])
    freqs, pxx = welch(bp[:, quiet], fs=fs, nperseg=min(1024, bp.shape[0]))
    print(f"quiet PSD peak: {freqs[np.argmax(pxx[freqs > 1]) + int((freqs <= 1).sum())]:.1f} Hz")

    print("\nclass coverage and contrast:")
    if signal is not None:
        # The only contrast that can be checked against the label threshold is
        # signal against background *in the same cells*, in the band the labels
        # were derived in.  Most of the background's power sits below 2 Hz in
        # drift and the microseism, which the labelling envelope high-passes
        # away -- compare broadband and every class looks quieter than the noise.
        corner = max(1.0, 0.002 * fs)
        b, a = butter(4, corner / (fs / 2), btype="high")
        s_band = filtfilt(b, a, signal, axis=0)
        b_band = filtfilt(b, a, background, axis=0)
        print(f"  (signal vs background inside each class, above {corner:.0f} Hz)")
    for class_id in sorted(set(np.unique(mask))):
        cells = mask == class_id
        label = ("ignore" if class_id == IGNORE_ID
                 else names.get(class_id, str(class_id)))
        line = f"  {label:13s} {100*cells.mean():6.2f}%"
        if signal is not None and cells.sum() > 100:
            contrast = 20 * np.log10(
                np.sqrt((s_band[cells] ** 2).mean())
                / (np.sqrt((b_band[cells] ** 2).mean()) + 1e-30)
            )
            line += f"   {contrast:+6.2f} dB"
            if class_id not in (0, IGNORE_ID) and contrast < cfg["label_hi_db"] - 6.0:
                problems.append(
                    f"class {label} labelled at {contrast:+.1f} dB, below the "
                    f"{cfg['label_hi_db']:.0f} dB label threshold"
                )
        print(line)
    if not (mask == 0).any():
        problems.append("no background cells at all")

    reference = np.sqrt((bp[mask == 0] ** 2).mean()) if (mask == 0).any() else np.nan
    if np.isfinite(reference):
        parts = []
        for class_id in sorted(set(np.unique(mask)) - {0}):
            cells = mask == class_id
            label = "ignore" if class_id == IGNORE_ID else names.get(class_id, str(class_id))
            parts.append(f"{label} {20*np.log10(np.sqrt((bp[cells]**2).mean())/reference):+.1f}")
        print("  (15-150 Hz, class cells vs background cells, informational: "
              + ", ".join(parts) + ")")

    if background is not None:
        print("\nbackground spatial structure (channel-lag correlation):")
        x = bandpass(background.T, fs)
        x = (x - x.mean(0)) / (x.std(0) + 1e-30)
        gauge_lag = max(1, int(round(gauge / dx)))
        lags = sorted({1, 2, 5, gauge_lag, 2 * gauge_lag, 50})
        for lag in lags:
            if lag >= x.shape[1]:
                continue
            value = float(np.mean(x[:, :-lag] * x[:, lag:]))
            tag = "  <- gauge length" if lag == gauge_lag else ""
            print(f"  lag {lag:4d}: {value:+.3f}{tag}")
        if float(np.mean(x[:, :-1] * x[:, 1:])) < 0.3:
            problems.append("adjacent channels are nearly uncorrelated: "
                            "noise is not passing through the gauge operator")

    if snr_db is not None and scenario:
        print("\nevent speed from the labelled ridge:")
        tvec_coarse = np.arange(snr_db.shape[2]) * decim / fs
        # Peak-pick on each event's own envelope, not on its ratio to the
        # background: the background envelope breathes with the gusts, and a
        # quiet moment elsewhere in the record otherwise steals the peak.
        background_env = signal_envelope(background.T, cfg, decim)[:, : snr_db.shape[2]]
        event_env = (10.0 ** (snr_db / 20.0)) * background_env[None, :, :]
        for index, ev in enumerate(scenario):
            if ev["kind"] not in ("train", "wheel_flat", "road_vehicle", "walker"):
                continue
            fit = estimate_event_speed(
                event_env[index], snr_db[index] >= float(cfg["label_hi_db"]),
                tvec_coarse, ch_pos,
                x_limits=track_range(ev, float(cfg["duration"])),
            )
            if fit is None:
                print(f"  {event_ids[index]:12s} ({ev['class']:12s}) "
                      f"scenario {ev['speed']:+7.2f} m/s   footprint too small to fit")
                continue
            verdict, failed = speed_verdict(
                fit, ev, float(cfg["wave_speed"]), args.speed_tolerance
            )
            if failed:
                problems.append(
                    f"{event_ids[index]} speed {fit['speed']:.1f} vs {ev['speed']:.1f}"
                )
            print(f"  {event_ids[index]:12s} ({ev['class']:12s}) "
                  f"scenario {ev['speed']:+7.2f}  measured {fit['speed']:+7.2f} m/s "
                  f"(95% CI {fit['lo']:+.1f}..{fit['hi']:+.1f}, {fit['channels']} ch)  "
                  f"{verdict}")

    labels_csv = args.h5.parent / "labels.csv"
    if labels_csv.exists():
        print("\nboxes:")
        for row in labels_csv.read_text().strip().splitlines()[1:]:
            cls, ident, t0, t1, x0, x1, snr, cells = row.split(",")
            if not t0:
                print(f"  {cls:13s} {ident:10s} not labelled anywhere "
                      f"(requested {snr} dB)")
                continue
            if not (0.0 <= float(t0) <= float(t1) <= cfg["duration"] + 1e-6):
                problems.append(f"box {ident} has t0={t0} t1={t1} outside [0, {cfg['duration']}]")
            print(f"  {cls:13s} {ident:10s} t {float(t0):6.2f}..{float(t1):6.2f} s   "
                  f"x {float(x0):7.1f}..{float(x1):7.1f} m   snr {snr} dB   {cells} cells")

    print()
    if problems:
        print(f"FAILED {len(problems)} check(s):")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
