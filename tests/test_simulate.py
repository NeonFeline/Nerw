"""Guards for the properties the generator is supposed to have.

Each test pins one of the defects the rewrite fixed, so a regression fails here
rather than silently producing a dataset that scores well for the wrong reason.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

import make_dataset as md
import simulate_das as sd

FS = 1000.0


def small_cfg(**overrides):
    cfg = {
        "fs": FS, "duration": 6.0, "n_channels": 60, "channel_spacing": 1.02,
        "gauge_length": 10.0, "x_offset": 0.0, "wave_speed": 250.0,
        "attenuation": 2e-4, "attenuation_power": 0.6, "attenuation_ref_freq": 50.0,
        "dispersion": 0.08, "dispersion_ref_freq": 20.0, "near_field": 2.0,
        "noise_correlation_m": 2.0, "propagation_bands": 12, "band_fmin": 0.5,
        "band_pad_seconds": 4.0, "oversample": 8, "doppler_focus": 1.0,
        "geometry_step": 0.01, "envelope_rate": 40.0, "envelope_window": 0.2,
        "label_hi_db": 6.0, "label_lo_db": 3.0, "label_margin_db": 6.0,
        "label_transient_grow": -1, "noise_std": 1e-7, "seed": 11,
    }
    cfg.update(overrides)
    return cfg


def small_scenario():
    return [
        {"kind": "train", "class": "train", "id": "t0", "x0": -120.0, "speed": 25.0,
         "y": 7.0, "z": 1.2, "snr_db": 18.0, "qs_ratio": 0.4, "fmin": 15.0,
         "fmax": 300.0, "f0": 55.0, "axle_offsets": [0.0, 2.5, 13.0, 15.5]},
        {"kind": "static", "class": "digging", "id": "d0", "x": 30.0, "y": 8.0,
         "z": 0.4, "t_start": 1.0, "n_hits": 10, "rate": 2.5, "snr_db": 14.0,
         "f0": 22.0, "scrape_prob": 0.3},
    ]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = small_cfg()
    scenario = small_scenario()
    result = sd.synthesize(cfg, scenario)
    out = tmp_path_factory.mktemp("run")
    sd.write_outputs(out, cfg, result, scenario)
    return out, cfg, result


# -- one propagation operator ---------------------------------------------


def test_band_weights_are_a_partition_of_unity():
    cfg = small_cfg()
    weights = sd.band_weights(np.fft.rfftfreq(4096, 1.0 / FS), cfg)
    assert np.allclose(weights.sum(axis=0), 1.0, atol=1e-12)
    assert (weights >= 0).all()


def test_moving_and_static_paths_are_the_same_operator():
    """A source that does not move must reproduce the frequency-domain result."""
    cfg = small_cfg(duration=4.0)
    nt = int(cfg["duration"] * FS)
    waveform = sd._band_noise(nt, FS, np.random.default_rng(0), 5.0, 350.0)
    taps, _ = sd.gauge_taps(sd.channel_positions(cfg), cfg)

    static = sd._propagate(waveform, cfg, taps - 20.0, 7.0, 1.2)
    moving = sd._np(sd.moving_field(
        cfg, taps, np.full(nt, 20.0), np.zeros(nt), waveform, 7.0, 1.2, oversample=8
    ))
    error = np.linalg.norm(static - moving) / np.linalg.norm(static)
    assert error < 0.03, f"operators disagree by {error:.1%}"


def test_fractional_delay_error_falls_with_oversampling():
    """The residual between the paths is interpolation, not a modelling gap."""
    cfg = small_cfg(duration=4.0)
    nt = int(cfg["duration"] * FS)
    waveform = sd._band_noise(nt, FS, np.random.default_rng(1), 5.0, 350.0)
    taps, _ = sd.gauge_taps(sd.channel_positions(cfg), cfg)
    static = sd._propagate(waveform, cfg, taps - 20.0, 7.0, 1.2)

    errors = []
    for oversample in (2, 8):
        moving = sd._np(sd.moving_field(
            cfg, taps, np.full(nt, 20.0), np.zeros(nt), waveform, 7.0, 1.2,
            oversample=oversample,
        ))
        errors.append(np.linalg.norm(static - moving) / np.linalg.norm(static))
    assert errors[1] < errors[0] / 3


def test_ridge_slope_recovers_the_source_speed():
    """A source that crosses the array must stamp its own speed on the record."""
    from scipy.ndimage import uniform_filter1d
    from scipy.stats import theilslopes

    cfg = small_cfg(duration=14.0, n_channels=150, channel_spacing=1.0)
    nt = int(cfg["duration"] * FS)
    ch_pos = sd.channel_positions(cfg)
    taps, _ = sd.gauge_taps(ch_pos, cfg)
    for speed in (30.0, -18.0):
        rng = np.random.default_rng(2)
        wave = sd._axle_wavelet(nt, FS, rng, 15.0, 300.0, f0=55.0, impulse_rate=8.0)
        start = float(ch_pos[0] - 60.0) if speed > 0 else float(ch_pos[-1] + 60.0)
        track = start + speed * np.arange(nt) / FS
        field = sd._gauge_combine(
            sd._np(sd.moving_field(cfg, taps, track, np.full(nt, speed), wave, 7.0, 1.2)), cfg
        )
        envelope = uniform_filter1d(field.astype(np.float64) ** 2, 200, axis=1)
        slope = theilslopes(np.argmax(envelope, axis=1) / FS, ch_pos)[0]
        assert abs(1.0 / slope - speed) < 0.05 * abs(speed)


def test_a_moving_source_is_doppler_shifted():
    """Approaching and receding tones must land either side of the emitted one."""
    cfg = small_cfg(duration=6.0, dispersion=0.0)
    nt = int(cfg["duration"] * FS)
    t = np.arange(nt) / FS
    f0 = 60.0
    tone = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    taps = np.array([0.0])
    speed = 60.0
    track = -150.0 + speed * t

    field = sd._np(sd.moving_field(cfg, taps, track, np.full(nt, speed), tone, 4.0, 0.0))[0]
    half = nt // 2
    freqs = np.fft.rfftfreq(half, 1.0 / FS)
    approaching = freqs[np.argmax(np.abs(np.fft.rfft(field[:half])))]
    receding = freqs[np.argmax(np.abs(np.fft.rfft(field[half:])))]
    assert approaching > f0 > receding


# -- outputs ---------------------------------------------------------------


def test_traces_are_exactly_signal_plus_background(run):
    out, _, _ = run
    with h5py.File(out / "das.h5", "r") as f:
        traces = f["traces"][:]
    with h5py.File(out / "components.h5", "r") as f:
        total = f["signal"][:] + f["background"][:]
    assert np.allclose(traces, total, rtol=1e-5, atol=1e-14)


def test_scale_factor_is_not_double_applied(run):
    """Data is written in rad/s, so a compliant reader must not rescale it."""
    out, cfg, _ = run
    with h5py.File(out / "das.h5", "r") as f:
        assert float(f.attrs["scale_factor"]) == 1.0
        assert f.attrs["units_after_scaling"] == "rad/s"
        traces = f["traces"][:]
    # the quiet channels already sit at the requested noise level
    quiet = np.median(np.std(traces, axis=0))
    assert 0.2 * cfg["noise_std"] < quiet < 5.0 * cfg["noise_std"]


def test_boxes_stay_inside_the_record(run):
    """A negative-speed event used to be written with t1 past the end."""
    out, cfg, result = run
    rows = (out / "labels.csv").read_text().strip().splitlines()[1:]
    assert rows
    for row in rows:
        _, _, t0, t1, x0, x1, _, _ = row.split(",")
        if not t0:
            continue
        assert 0.0 <= float(t0) <= float(t1) <= cfg["duration"] + 1e-9
        # box coordinates are rounded to 0.1 m for the csv
        assert result["ch_pos"][0] - 0.05 <= float(x0) <= float(x1) <= result["ch_pos"][-1] + 0.05


def test_negative_speed_boxes_stay_inside_the_record():
    cfg = small_cfg(seed=5)
    scenario = [dict(small_scenario()[0], x0=180.0, speed=-25.0, id="rev")]
    result = sd.synthesize(cfg, scenario)
    for box in result["boxes"]:
        if box["t0"] is None:
            continue
        assert 0.0 <= box["t0"] <= box["t1"] <= cfg["duration"] + 1e-9


# -- labels ----------------------------------------------------------------


def test_every_event_class_outranks_the_background(run):
    """digging and burst used to sit below the class-0 samples."""
    from scipy.signal import butter, filtfilt

    out, _, _ = run
    with h5py.File(out / "das.h5", "r") as f:
        traces, mask = f["traces"][:], f["labels/class_mask"][:]
    b, a = butter(4, (15 / (FS / 2), 150 / (FS / 2)), btype="band")
    energy = filtfilt(b, a, traces, axis=0) ** 2
    background = np.sqrt(energy[mask == 0].mean())
    for class_id in np.unique(mask):
        if class_id in (0, sd.IGNORE_ID):
            continue
        assert np.sqrt(energy[mask == class_id].mean()) > background


def test_labels_follow_snr_not_geometry(run):
    """Labelled cells clear label_hi_db; plain background stays under label_lo_db."""
    out, cfg, result = run
    with h5py.File(out / "components.h5", "r") as f:
        snr = f["event_snr_db"][:]
        decim = int(f.attrs["envelope_decimation"])
    mask = result["mask"][:, :: 1]
    coarse = mask[:, ::decim][:, : snr.shape[2]]
    best = snr.max(axis=0)
    labelled = (coarse != 0) & (coarse != sd.IGNORE_ID)
    assert labelled.any()
    assert best[labelled].min() >= cfg["label_hi_db"] - 1e-3
    assert best[coarse == 0].max() <= cfg["label_lo_db"] + 1e-3


def test_unlabelled_transients_are_marked_ignore():
    """A loud background burst must not be handed to the model as class 0."""
    cfg = small_cfg(seed=3)
    cfg["background"] = {"burst_rate": 4.0, "burst_gain": [12.0, 30.0], "glitch_rate": 1.0}
    result = sd.synthesize(cfg, [])
    mask = result["mask"]
    assert (mask == sd.IGNORE_ID).any(), "no transient was flagged"

    background = result["background"]
    level = np.median(np.abs(background))
    loud = np.abs(background) > 12.0 * level
    labelled_background = loud & (mask == 0)
    assert labelled_background.mean() < 0.002


def test_dead_channels_do_not_look_like_perfect_events():
    """Coupling scales the signal and the ground-borne background together."""
    cfg = small_cfg(seed=17)
    cfg["background"] = {"dead_fraction": 0.15, "noisy_fraction": 0.0}
    result = sd.synthesize(cfg, small_scenario())
    with np.errstate(divide="ignore"):
        quiet = result["coupling"] < 0.1 * np.median(result["coupling"])
    if not quiet.any():
        pytest.skip("no dead run drawn")
    labelled = (result["mask"] != 0) & (result["mask"] != sd.IGNORE_ID)
    assert labelled[quiet].mean() < labelled[~quiet].mean()


def test_overlapping_events_are_ignored_not_overwritten():
    """Two events of similar strength in one place must not pick a silent winner."""
    cfg = small_cfg(seed=23)
    shared = {"y": 6.0, "z": 0.4, "t_start": 1.5, "n_hits": 30, "rate": 6.0,
              "snr_db": 20.0, "f0": 25.0, "scrape_prob": 0.0}
    scenario = [
        {"kind": "static", "class": "digging", "id": "a", "x": 30.0, **shared},
        {"kind": "burst", "class": "burst", "id": "b", "x": 30.0, "y": 6.0, "z": 0.4,
         "t_start": 1.5, "duration": 3.0, "snr_db": 20.0, "fmin": 20.0, "fmax": 200.0},
    ]
    result = sd.synthesize(cfg, scenario)
    assert (result["mask"] == sd.IGNORE_ID).any()


# -- background ------------------------------------------------------------


def test_background_carries_the_gauge_spatial_signature():
    """Noise and signal must share a spatial transfer function.

    A gauge of L over a channel pitch dx leaves adjacent channels strongly
    correlated and puts a negative lobe near lag L/dx.  Channel-independent
    noise has neither, and a model trained on it learns a spatial prior that
    does not exist on a real fibre.
    """
    from scipy.signal import butter, filtfilt

    cfg = small_cfg(seed=31, n_channels=200, duration=8.0)
    result = sd.synthesize(cfg, [])
    b, a = butter(4, (15 / (FS / 2), 150 / (FS / 2)), btype="band")
    x = filtfilt(b, a, result["background"].T, axis=0)
    x = (x - x.mean(0)) / (x.std(0) + 1e-30)

    def lag(k):
        return float(np.mean(x[:, :-k] * x[:, k:]))

    gauge_lag = int(round(cfg["gauge_length"] / cfg["channel_spacing"]))
    assert lag(1) > 0.5
    assert lag(gauge_lag) < -0.05


def test_background_is_not_a_handful_of_common_modes():
    """In the band where events live, the noise must not be near-degenerate.

    Measured the way the pre-rewrite data was measured (15-150 Hz), which came
    out at an effective rank of 7 of 300 -- a subspace small enough that a model
    can explain "normal" without learning anything about the fibre.  Below the
    band, drift, common mode and the microseism plane wave are genuinely
    low-rank and are left alone.
    """
    from scipy.signal import butter, filtfilt

    cfg = small_cfg(seed=37, n_channels=200, duration=8.0)
    result = sd.synthesize(cfg, [])
    b, a = butter(4, (15 / (FS / 2), 150 / (FS / 2)), btype="band")
    x = filtfilt(b, a, result["background"].T, axis=0)
    x = x - x.mean(0)
    weights = np.linalg.eigvalsh(np.cov(x[::10].T))[::-1]
    weights = weights / weights.sum()
    effective_rank = 1.0 / np.sum(weights**2)
    assert effective_rank > 15.0, f"effective rank {effective_rank:.1f}"


def test_background_does_not_depend_on_the_scenario():
    """Independent streams: adding an event must not reshuffle the noise."""
    cfg_a, cfg_b = small_cfg(seed=41), small_cfg(seed=41)
    empty = sd.synthesize(cfg_a, [])
    full = sd.synthesize(cfg_b, small_scenario())
    assert np.array_equal(empty["background"] / cfg_a["realized_scale"],
                          full["background"] / cfg_b["realized_scale"])


def test_recorded_background_is_never_tiled(tmp_path):
    cfg = small_cfg(seed=43, duration=2.0, n_channels=20)
    result = sd.synthesize(cfg, [])
    sd.write_outputs(tmp_path, cfg, result, [], emit_components=False)
    big = small_cfg(seed=44, duration=6.0, n_channels=60)
    with pytest.raises(ValueError, match="tiling is not allowed"):
        sd.mix_background(np.zeros((60, 6000), np.float32), tmp_path / "das.h5",
                          np.random.default_rng(0), 0.5, big)


# -- dataset ---------------------------------------------------------------


def test_test_families_never_appear_in_training():
    assert not set(md.TRAIN_FAMILIES) & set(md.TEST_FAMILIES)
    assert set(md.TRAIN_FAMILIES) | set(md.TEST_FAMILIES) == set(md.SITE_FAMILIES)


def test_scenarios_differ_between_runs():
    cfg = small_cfg()
    scenarios = []
    for seed in range(8):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
        resolved = md.sample_site(rng, "embankment_soft", dict(cfg))
        scenarios.append(md.sample_scenario(rng, resolved, 6, (2.0, 26.0), 0.0))

    signatures = {json.dumps(s, sort_keys=True) for s in scenarios}
    assert len(signatures) == len(scenarios)
    positions = [e.get("x0", e.get("x")) for s in scenarios for e in s]
    assert len(set(positions)) > 0.8 * len(positions)


def test_scenario_sampling_is_reproducible():
    cfg = small_cfg()
    def draw():
        rng = np.random.default_rng(np.random.SeedSequence([7, 4242]))
        resolved = md.sample_site(rng, "urban_fill", dict(cfg))
        return resolved, md.sample_scenario(rng, resolved, 6, (2.0, 26.0), 0.0)
    first, second = draw(), draw()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
