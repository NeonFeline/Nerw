"""Build a synthetic DAS dataset with randomised scenarios and a family holdout.

Every run gets its own scene: how many events there are, what they are, where
they sit, which way they travel, how fast, and how far above the background
they land are all drawn per run.  Nothing but the site profile is shared, so a
model cannot score by memorising coordinates.

The split is by *scenario family*, not by seed.  A family is a site: its wave
speed, attenuation, dispersion, gauge, channel spacing, source stand-offs,
ambient character and event mix.  ``train`` and ``val`` runs come from the
training families (``val`` is for model selection); ``test`` runs come from
families that never appear in training, so the test score measures transfer to
an unseen installation rather than an unseen noise seed.

Everything needed to rebuild the dataset byte-for-byte is written to
``manifest.json``: the command line, library versions, and every run's family,
seed, resolved config and scenario.

    uv run python make_dataset.py --out data/synthetic/v2
    uv run python make_dataset.py --out data/synthetic/v2 --train-runs 24 --test-runs 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import simulate_das as sd

# Ranges are (low, high) and sampled uniformly, or log-uniformly where marked.
SITE_FAMILIES: dict[str, dict] = {
    "embankment_soft": {
        "wave_speed": (170.0, 220.0),
        "attenuation": (1.5e-4, 3.0e-4),
        "dispersion": (0.08, 0.14),
        "gauge_length": (8.0, 12.0),
        "channel_spacing": (0.95, 1.15),
        "track_offset": (5.0, 9.0),
        "road_offset": (11.0, 18.0),
        "noise_correlation": (1.5, 3.0),
        "ambient_sources": (6, 14),
        "ambient_level": (0.08, 0.18),
        "burst_rate": (0.05, 0.4),
        "glitch_rate": (0.03, 0.2),
        "event_mix": {"train": 3.0, "wheel_flat": 1.0, "road_vehicle": 2.0,
                      "walker": 1.5, "digging": 1.0, "burst": 1.5},
    },
    "cutting_stiff": {
        "wave_speed": (280.0, 380.0),
        "attenuation": (0.8e-4, 1.8e-4),
        "dispersion": (0.03, 0.08),
        "gauge_length": (9.0, 11.0),
        "channel_spacing": (1.0, 1.3),
        "track_offset": (4.0, 7.0),
        "road_offset": (20.0, 35.0),
        "noise_correlation": (1.0, 2.5),
        "ambient_sources": (4, 10),
        "ambient_level": (0.05, 0.12),
        "burst_rate": (0.02, 0.2),
        "glitch_rate": (0.02, 0.12),
        "event_mix": {"train": 3.0, "wheel_flat": 1.5, "road_vehicle": 1.0,
                      "walker": 1.0, "digging": 1.5, "burst": 1.0},
    },
    "urban_fill": {
        "wave_speed": (200.0, 260.0),
        "attenuation": (2.5e-4, 5.0e-4),
        "dispersion": (0.10, 0.18),
        "gauge_length": (5.0, 8.0),
        "channel_spacing": (1.8, 2.4),
        "track_offset": (8.0, 14.0),
        "road_offset": (8.0, 14.0),
        "noise_correlation": (2.0, 4.0),
        "ambient_sources": (14, 24),
        "ambient_level": (0.18, 0.30),
        "burst_rate": (0.5, 2.0),
        "glitch_rate": (0.15, 0.45),
        "event_mix": {"train": 2.0, "wheel_flat": 1.0, "road_vehicle": 4.0,
                      "walker": 3.0, "digging": 2.0, "burst": 2.5},
    },
    "coastal_marsh": {
        "wave_speed": (120.0, 170.0),
        "attenuation": (4.0e-4, 8.0e-4),
        "dispersion": (0.14, 0.22),
        "gauge_length": (10.0, 14.0),
        "channel_spacing": (0.9, 1.1),
        "track_offset": (6.0, 11.0),
        "road_offset": (25.0, 45.0),
        "noise_correlation": (2.5, 5.0),
        "ambient_sources": (10, 20),
        "ambient_level": (0.15, 0.28),
        "burst_rate": (0.1, 0.6),
        "glitch_rate": (0.05, 0.3),
        "event_mix": {"train": 3.0, "wheel_flat": 1.0, "road_vehicle": 1.0,
                      "walker": 2.0, "digging": 1.0, "burst": 2.0},
    },
    # held out: neither the medium, the geometry nor the event mix is seen above
    "viaduct_hard": {
        "wave_speed": (400.0, 550.0),
        "attenuation": (0.5e-4, 1.2e-4),
        "dispersion": (0.01, 0.05),
        "gauge_length": (6.0, 9.0),
        "channel_spacing": (1.4, 1.8),
        "track_offset": (2.0, 5.0),
        "road_offset": (30.0, 60.0),
        "noise_correlation": (0.8, 2.0),
        "ambient_sources": (5, 12),
        "ambient_level": (0.06, 0.15),
        "burst_rate": (0.05, 0.5),
        "glitch_rate": (0.03, 0.25),
        "event_mix": {"train": 4.0, "wheel_flat": 2.0, "road_vehicle": 0.5,
                      "walker": 1.0, "digging": 1.0, "burst": 1.5},
    },
    "peat_quiet": {
        "wave_speed": (95.0, 135.0),
        "attenuation": (6.0e-4, 1.2e-3),
        "dispersion": (0.18, 0.28),
        "gauge_length": (12.0, 20.0),
        "channel_spacing": (2.5, 4.0),
        "track_offset": (10.0, 18.0),
        "road_offset": (40.0, 80.0),
        "noise_correlation": (3.0, 6.0),
        "ambient_sources": (3, 8),
        "ambient_level": (0.03, 0.09),
        "burst_rate": (0.01, 0.15),
        "glitch_rate": (0.01, 0.1),
        "event_mix": {"train": 3.0, "wheel_flat": 1.5, "road_vehicle": 1.0,
                      "walker": 2.5, "digging": 2.0, "burst": 1.0},
    },
}

TRAIN_FAMILIES = ("embankment_soft", "cutting_stiff", "urban_fill", "coastal_marsh")
TEST_FAMILIES = ("viaduct_hard", "peat_quiet")


def _uniform(rng, span):
    lo, hi = span
    return float(rng.uniform(lo, hi))


def _log_uniform(rng, span):
    lo, hi = span
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_site(rng, family: str, base: dict) -> dict:
    """Resolve one run's physics and background knobs from its family."""
    spec = SITE_FAMILIES[family]
    cfg = dict(base)
    cfg.update({
        "wave_speed": _uniform(rng, spec["wave_speed"]),
        "attenuation": _log_uniform(rng, spec["attenuation"]),
        "dispersion": _uniform(rng, spec["dispersion"]),
        "gauge_length": _uniform(rng, spec["gauge_length"]),
        "channel_spacing": _uniform(rng, spec["channel_spacing"]),
        "noise_correlation_m": _uniform(rng, spec["noise_correlation"]),
        "family": family,
        "background": {
            "ambient_sources": int(rng.integers(*spec["ambient_sources"])),
            "ambient_level": _uniform(rng, spec["ambient_level"]),
            "burst_rate": _log_uniform(rng, spec["burst_rate"]),
            "glitch_rate": _log_uniform(rng, spec["glitch_rate"]),
            "gust": _uniform(rng, (0.3, 1.4)),
            "microseism": _uniform(rng, (0.12, 0.40)),
            "dead_fraction": _uniform(rng, (0.0, 0.05)),
            "noisy_fraction": _uniform(rng, (0.0, 0.04)),
            "burst_gain": [_uniform(rng, (3.0, 8.0)), _uniform(rng, (10.0, 30.0))],
        },
    })
    return cfg


def _axle_offsets(rng):
    """Bogie geometry: ``n_cars`` cars, two axles per bogie."""
    n_cars = int(rng.integers(1, 7))
    car_length = _uniform(rng, (13.0, 26.0))
    axle_gap = _uniform(rng, (1.8, 3.0))
    offsets = []
    for car in range(n_cars):
        base = car * car_length
        offsets += [base, base + axle_gap, base + car_length - axle_gap - 2.0,
                    base + car_length - 2.0]
    return [round(o, 2) for o in offsets[: 4 * n_cars]]


def _moving_start(rng, ch_pos, speed, duration, margin):
    """Start position that keeps the source crossing the array during the run."""
    lo, hi = float(ch_pos[0]) - margin, float(ch_pos[-1]) + margin
    travel = speed * duration
    if speed >= 0:
        return float(rng.uniform(lo - travel, hi - 0.25 * travel))
    return float(rng.uniform(lo - 0.25 * travel, hi - travel))


def sample_event(rng, kind, cfg, ch_pos, index, snr_range):
    duration = float(cfg["duration"])
    spec = SITE_FAMILIES[cfg["family"]]
    snr = _uniform(rng, snr_range)
    common = {"id": f"{kind}{index}", "snr_db": round(snr, 2)}

    if kind in ("train", "wheel_flat"):
        speed = _uniform(rng, (12.0, 48.0)) * rng.choice([-1.0, 1.0])
        offsets = _axle_offsets(rng)
        ev = {
            "kind": kind, "class": kind, **common,
            "x0": round(_moving_start(rng, ch_pos, speed, duration, 80.0), 1),
            "speed": round(float(speed), 2),
            "y": round(_uniform(rng, spec["track_offset"]), 2),
            "z": round(_uniform(rng, (0.6, 2.0)), 2),
            "axle_offsets": offsets,
            "fmin": round(_uniform(rng, (8.0, 25.0)), 1),
            "fmax": round(_uniform(rng, (200.0, 420.0)), 1),
            "f0": round(_uniform(rng, (35.0, 90.0)), 1),
            "rumble": round(_uniform(rng, (0.1, 0.4)), 3),
            "qs_ratio": round(_uniform(rng, (0.2, 0.8)), 3),
            "speed_jitter": round(_uniform(rng, (0.0, 0.05)), 4),
            "coda": round(_uniform(rng, (0.05, 0.5)), 3),
        }
        if kind == "wheel_flat":
            ev["wheel_period"] = round(_uniform(rng, (0.12, 0.45)), 3)
            ev["flat_amp"] = round(_uniform(rng, (0.1, 0.6)), 3)
        return ev

    if kind == "road_vehicle":
        speed = _uniform(rng, (6.0, 33.0)) * rng.choice([-1.0, 1.0])
        return {
            "kind": kind, "class": kind, **common,
            "x0": round(_moving_start(rng, ch_pos, speed, duration, 40.0), 1),
            "speed": round(float(speed), 2),
            "y": round(_uniform(rng, spec["road_offset"]), 2),
            "z": round(_uniform(rng, (0.3, 1.2)), 2),
            "length": round(_uniform(rng, (4.0, 18.0)), 1),
            "fmin": round(_uniform(rng, (4.0, 12.0)), 1),
            "fmax": round(_uniform(rng, (70.0, 180.0)), 1),
            "f0": round(_uniform(rng, (25.0, 60.0)), 1),
            "engine_f0": round(_uniform(rng, (20.0, 75.0)), 1),
            "engine_amp": round(_uniform(rng, (0.05, 0.6)), 3),
            "qs_ratio": round(_uniform(rng, (0.0, 0.35)), 3),
            "speed_jitter": round(_uniform(rng, (0.0, 0.08)), 4),
            "coda": round(_uniform(rng, (0.05, 0.45)), 3),
        }

    if kind == "walker":
        speed = _uniform(rng, (0.7, 2.2)) * rng.choice([-1.0, 1.0])
        t0 = _uniform(rng, (0.0, max(duration - 4.0, 0.5)))
        t1 = min(duration, t0 + _uniform(rng, (3.0, min(30.0, duration))))
        return {
            "kind": kind, "class": "footsteps", **common,
            "x0": round(float(rng.uniform(ch_pos[0], ch_pos[-1])), 1),
            "speed": round(float(speed), 2),
            "t_start": round(t0, 2), "t_end": round(t1, 2),
            "y": round(_uniform(rng, (0.5, 6.0)), 2),
            "z": round(_uniform(rng, (0.02, 0.2)), 3),
            "rate": round(_uniform(rng, (1.2, 2.6)), 2),
            "f0": round(_uniform(rng, (20.0, 60.0)), 1),
            "speed_jitter": 0.0,
            "coda": round(_uniform(rng, (0.05, 0.3)), 3),
        }

    if kind == "digging":
        t0 = _uniform(rng, (0.0, max(duration - 3.0, 0.5)))
        rate = _uniform(rng, (1.0, 4.0))
        return {
            "kind": "static", "class": "digging", **common,
            "x": round(float(rng.uniform(ch_pos[0], ch_pos[-1])), 1),
            "y": round(_uniform(rng, (3.0, 30.0)), 2),
            "z": round(_uniform(rng, (0.1, 1.5)), 2),
            "t_start": round(t0, 2),
            "n_hits": int(rng.integers(6, 80)),
            "rate": round(rate, 2),
            "f0": round(_uniform(rng, (12.0, 45.0)), 1),
            "scrape_prob": round(_uniform(rng, (0.0, 0.7)), 3),
            "coda": round(_uniform(rng, (0.05, 0.5)), 3),
        }

    if kind == "burst":
        dur = _uniform(rng, (0.3, 5.0))
        t0 = _uniform(rng, (0.0, max(duration - dur, 0.1)))
        fmin = _uniform(rng, (20.0, 180.0))
        return {
            "kind": "burst", "class": "burst", **common,
            "x": round(float(rng.uniform(ch_pos[0], ch_pos[-1])), 1),
            "y": round(_uniform(rng, (1.0, 25.0)), 2),
            "z": round(_uniform(rng, (0.05, 1.0)), 2),
            "t_start": round(t0, 2), "duration": round(dur, 2),
            "fmin": round(fmin, 1),
            "fmax": round(min(0.45 * cfg["fs"], fmin * _uniform(rng, (1.8, 5.0))), 1),
            "coda": round(_uniform(rng, (0.05, 0.5)), 3),
        }

    raise ValueError(f"unknown event kind: {kind}")


def sample_scenario(rng, cfg, max_events, snr_range, empty_probability):
    """A fresh scene: count, kinds, placement and SNR all drawn per run."""
    if rng.random() < empty_probability:
        return []
    ch_pos = sd.channel_positions(cfg)
    mix = SITE_FAMILIES[cfg["family"]]["event_mix"]
    kinds = list(mix)
    weights = np.asarray([mix[k] for k in kinds], float)
    weights /= weights.sum()
    n_events = int(rng.integers(1, max_events + 1))
    scenario = []
    for index in range(n_events):
        kind = str(rng.choice(kinds, p=weights))
        scenario.append(sample_event(rng, kind, cfg, ch_pos, index, snr_range))
    return scenario


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_run(args, split: str, family: str, run_index: int, seed: int) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
    base = {
        "fs": args.fs,
        "duration": args.duration,
        "n_channels": args.n_channels,
        "x_offset": 0.0,
        "attenuation_power": 0.6,
        "attenuation_ref_freq": 50.0,
        "dispersion_ref_freq": 20.0,
        "near_field": 2.0,
        "propagation_bands": args.propagation_bands,
        "band_fmin": 0.5,
        "band_pad_seconds": 4.0,
        "oversample": args.oversample,
        "doppler_focus": 1.0,
        "geometry_step": 0.01,
        "envelope_rate": 40.0,
        "envelope_window": 0.2,
        "label_hi_db": args.label_hi_db,
        "label_lo_db": args.label_lo_db,
        "label_margin_db": args.label_margin_db,
        "label_transient_grow": -1,
        "noise_std": args.noise_std,
        "seed": seed,
        "split": split,
    }
    cfg = sample_site(rng, family, base)
    scenario = sample_scenario(
        rng, cfg, args.max_events, (args.min_snr_db, args.max_snr_db), args.empty_probability
    )

    out_dir = Path(args.out) / split / f"{family}_{run_index:03d}"
    started = time.time()
    result = sd.synthesize(cfg, scenario)
    h5_path = sd.write_outputs(out_dir, cfg, result, scenario,
                               emit_components=not args.no_components)
    if not args.no_qa:
        sd.qa_plot(h5_path, out_dir / "qa.png", result["ch_pos"], cfg)
    elapsed = time.time() - started

    mask = result["mask"]
    coverage = {sd.CLASS_NAMES[k]: int((mask == k).sum()) for k in sd.CLASS_NAMES}
    coverage["ignore"] = int((mask == sd.IGNORE_ID).sum())
    return {
        "split": split,
        "family": family,
        "seed": seed,
        "path": str(out_dir.relative_to(Path(args.out))),
        "sha256": _file_digest(h5_path),
        "seconds": round(elapsed, 2),
        "config": cfg,
        "scenario": scenario,
        "boxes": result["boxes"],
        "mask_coverage": coverage,
        "mask_fraction": {k: round(v / mask.size, 5) for k, v in coverage.items()},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-runs", type=int, default=24)
    ap.add_argument("--val-runs", type=int, default=6)
    ap.add_argument("--test-runs", type=int, default=12)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--fs", type=float, default=1000.0)
    ap.add_argument("--n-channels", type=int, default=300)
    ap.add_argument("--propagation-bands", type=int, default=12)
    ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--noise-std", type=float, default=1e-7)
    ap.add_argument("--max-events", type=int, default=6)
    ap.add_argument("--min-snr-db", type=float, default=2.0)
    ap.add_argument("--max-snr-db", type=float, default=26.0)
    ap.add_argument("--empty-probability", type=float, default=0.1,
                    help="fraction of runs with no events at all")
    ap.add_argument("--label-hi-db", type=float, default=6.0)
    ap.add_argument("--label-lo-db", type=float, default=3.0)
    ap.add_argument("--label-margin-db", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-components", action="store_true")
    ap.add_argument("--no-qa", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    sd.set_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[str, str, int, int]] = []
    # fixed split keys: str.__hash__ is salted per process and would make the
    # seeds -- and so the whole dataset -- irreproducible between runs
    split_keys = {"train": 0, "val": 1, "test": 2}
    for split, count, families in (
        ("train", args.train_runs, TRAIN_FAMILIES),
        ("val", args.val_runs, TRAIN_FAMILIES),
        ("test", args.test_runs, TEST_FAMILIES),
    ):
        for index in range(count):
            family = families[index % len(families)]
            seed = int(np.random.SeedSequence(
                [args.seed, split_keys[split], index]
            ).generate_state(1)[0])
            plan.append((split, family, index, seed))

    runs = []
    started = time.time()
    for position, (split, family, index, seed) in enumerate(plan, start=1):
        record = build_run(args, split, family, index, seed)
        runs.append(record)
        events = len(record["scenario"])
        print(f"[{position:3d}/{len(plan)}] {split:5s} {family:16s} seed={seed:<12d} "
              f"events={events} ignore={record['mask_fraction']['ignore']:.3f} "
              f"({record['seconds']:.1f}s)", flush=True)

    manifest = {
        "generator": "simulate_das.py",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git_revision": _git_revision(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": __import__("torch").__version__,
            "h5py": __import__("h5py").__version__,
            "scipy": __import__("scipy").__version__,
        },
        "device": str(sd.device()),
        "class_ids": sd.CLASS_IDS,
        "ignore_id": sd.IGNORE_ID,
        "families": SITE_FAMILIES,
        "train_families": list(TRAIN_FAMILIES),
        "test_families": list(TEST_FAMILIES),
        "split_policy": "by scenario family; test families never appear in train or val",
        "total_seconds": round(time.time() - started, 1),
        "runs": runs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out/'manifest.json'}  ({len(runs)} runs, "
          f"{manifest['total_seconds']:.0f}s)")
    for split in ("train", "val", "test"):
        rows = [r for r in runs if r["split"] == split]
        if not rows:
            continue
        events = sum(len(r["scenario"]) for r in rows)
        ignore = np.mean([r["mask_fraction"]["ignore"] for r in rows])
        background = np.mean([r["mask_fraction"]["background"] for r in rows])
        print(f"  {split:5s}: {len(rows):3d} runs, {events:3d} events, "
              f"background {background:.1%}, ignore {ignore:.1%}, "
              f"families {sorted({r['family'] for r in rows})}")


if __name__ == "__main__":
    main()
