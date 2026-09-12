"""Synthetic DAS generator for a fibre optic cable running along a railway.

Writes a miniDAS-style HDF5 file (``traces`` with shape [nt, nch], float32),
a per-sample class mask, event boxes, metadata and a QA waterfall plot, and
optionally a companion file holding the clean event field and the background
separately.

One propagation operator
------------------------
Every source -- moving or stationary -- is propagated with the same operator.
Surface waves are dispersive, ``c(f) = c0 (f / f_ref)^-beta``, and attenuate as
``a(f) = a0 (f / f_ref)^p``.  The operator is quantised into ``propagation_bands``
log-spaced bands that form a partition of unity in frequency; within a band the
phase velocity and attenuation are constant, so propagation to range ``r`` is

    H(f, r) = sum_b B_b(f) exp(-2i pi f r / c_b) exp(-a_b r) / sqrt(max(r, r0)).

Stationary sources apply ``H`` directly in the frequency domain; moving sources
apply the same ``(B_b, c_b, a_b)`` triple in the time domain at the retarded
time, solved from ``tau = t - r(tau)/c_b`` by fixed-point iteration, with the
exact ``dtau/dt`` Doppler amplitude factor.  Both paths therefore share one
operator by construction -- ``tests/test_simulate.py`` asserts they agree for a
stationary source -- so nothing about the propagation is correlated with the
class label.  Fractional delays are taken on an ``oversample``-times upsampled
copy of each band, so the interpolator is flat to well past the highest
modelled frequency.

A channel measures the axial strain rate averaged over the gauge length,
``eps_rate = (v(x + L/2) - v(x - L/2)) / L``.

Background
----------
The background mimics a raw field recording rather than white noise:

* instrument phase noise (1/f and white) generated as a spatially correlated
  field along the fibre and passed through the *same* gauge difference operator
  as the signals, so noise and signal share a spatial transfer function;
* laser common-mode, thermal drift and mains hum, each with a smooth spatial
  profile along the route rather than being identical on every channel;
* a propagating microseism wavefield and distributed ambient surface-wave
  sources with gust modulation;
* smooth coupling variation with weak/free-span sections, noisy and dead
  channel runs, sparse glitches and unlabelled transient bursts.

Labels
------
Labels are derived from signal-to-background ratio measured on the clean field,
not from a geometric corridor.  A cell is given an event class where that
event's envelope exceeds the background envelope by ``label_hi_db``; cells
between ``label_lo_db`` and ``label_hi_db``, cells where two events are within
``label_margin_db`` of each other, and cells touched by the unlabelled
background transients are set to ``IGNORE_ID`` (255) and must be excluded from
both training and scoring.  Event amplitudes are calibrated to a requested
``snr_db`` against the realised background rather than being fixed constants.

This remains a kinematic model: coupling, instrument response and medium
heterogeneity are empirical knobs, not inverted physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt

CLASS_IDS = {
    "background": 0,
    "train": 1,
    "road_vehicle": 2,
    "footsteps": 3,
    "digging": 4,
    "burst": 5,
    "wheel_flat": 6,
}
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}
IGNORE_ID = 255

MOVING_KINDS = ("train", "wheel_flat", "road_vehicle", "walker")
STATIC_KINDS = ("static", "burst")

_DEVICE: torch.device | None = None


def set_device(name: str | None = None) -> torch.device:
    global _DEVICE
    if name:
        _DEVICE = torch.device(name)
    elif _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


def device() -> torch.device:
    return set_device()


def _t(array, dtype=torch.float32) -> torch.Tensor:
    array = np.ascontiguousarray(array)
    if not array.flags.writeable:
        array = array.copy()
    return torch.as_tensor(array, dtype=dtype, device=device())


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to("cpu").numpy()


# --------------------------------------------------------------------------
# medium
# --------------------------------------------------------------------------


def _phase_velocity(freqs, cfg):
    beta = float(cfg.get("dispersion", 0.08))
    f_ref = float(cfg.get("dispersion_ref_freq", 20.0))
    f = np.maximum(np.asarray(freqs, np.float64), 1e-2)
    return float(cfg["wave_speed"]) * (f / f_ref) ** (-beta)


def _attenuation(freqs, cfg):
    power = float(cfg.get("attenuation_power", 0.6))
    f_ref = float(cfg.get("attenuation_ref_freq", 50.0))
    f = np.maximum(np.asarray(freqs, np.float64), 1.0)
    return float(cfg["attenuation"]) * (f / f_ref) ** power


def band_plan(cfg):
    """Band centre frequencies, slownesses and attenuations of the operator."""
    fs = float(cfg["fs"])
    n_bands = max(1, int(cfg.get("propagation_bands", 12)))
    f_lo = float(cfg.get("band_fmin", 0.5))
    f_hi = float(cfg.get("band_fmax", 0.5 * fs))
    if n_bands == 1:
        centres = np.array([np.sqrt(f_lo * f_hi)])
    else:
        centres = np.geomspace(f_lo, f_hi, n_bands)
    return centres, 1.0 / _phase_velocity(centres, cfg), _attenuation(centres, cfg)


def band_weights(freqs, cfg) -> np.ndarray:
    """``(n_bands, n_freqs)`` partition of unity: ``sum_b B_b(f) == 1``."""
    centres, _, _ = band_plan(cfg)
    freqs = np.asarray(freqs, np.float64)
    n = centres.size
    out = np.zeros((n, freqs.size), np.float64)
    if n == 1:
        out[0] = 1.0
        return out
    u = np.log(np.clip(freqs, centres[0], centres[-1]))
    uc = np.log(centres)
    k = np.clip(np.searchsorted(uc, u, side="right") - 1, 0, n - 2)
    span = uc[k + 1] - uc[k]
    t = np.where(span > 0, (u - uc[k]) / np.maximum(span, 1e-30), 0.0)
    upper = np.sin(0.5 * np.pi * t) ** 2
    idx = np.arange(freqs.size)
    out[k, idx] += 1.0 - upper
    out[k + 1, idx] += upper
    return out


def band_pad(cfg) -> int:
    """Samples of zero padding that keep the band filters from wrapping around.

    The lowest band is a raised cosine with a corner near ``band_fmin``, so its
    impulse response is metres of seconds long; both the frequency-domain and
    the time-domain path pad by the same amount so they stay comparable at the
    edges of the record.
    """
    return int(np.ceil(float(cfg.get("band_pad_seconds", 4.0)) * float(cfg["fs"])))


def _range_response(freqs_t, r_t, cfg, spreading=True, attenuation=True) -> torch.Tensor:
    """``(n_r, n_f)`` complex response of the shared propagation operator."""
    centres, slow, att = band_plan(cfg)
    weights = _t(band_weights(_np(freqs_t), cfg))
    out = torch.zeros((r_t.numel(), freqs_t.numel()), dtype=torch.complex64, device=device())
    r_col = r_t.reshape(-1, 1)
    for b in range(centres.size):
        angle = (-2.0 * np.pi * float(slow[b])) * (freqs_t.reshape(1, -1) * r_col)
        mag = weights[b].reshape(1, -1).expand(r_t.numel(), -1)
        if attenuation:
            mag = mag * torch.exp(-float(att[b]) * r_col)
        out += torch.polar(mag.contiguous(), angle)
    if spreading:
        out = out / torch.sqrt(torch.clamp(r_col, min=float(cfg["near_field"])))
    return out


def _propagate(waveform, cfg, dx, y=0.0, z=0.0, geometry=True) -> np.ndarray:
    """Propagate ``waveform`` to slant ranges ``sqrt(dx^2 + y^2 + z^2)``.

    Returns a ``(len(dx), len(waveform))`` array.  ``geometry=False`` drops
    attenuation and geometric spreading and keeps only the dispersive phase.
    """
    waveform = np.asarray(waveform, dtype=np.float32).ravel()
    nt = waveform.size
    fs = float(cfg["fs"])
    dx = np.atleast_1d(np.asarray(dx, dtype=np.float64))
    r = np.sqrt(dx**2 + float(y) ** 2 + float(z) ** 2)
    _, slow, _ = band_plan(cfg)
    n_pad = int(np.ceil(r.max() * slow.max() * fs)) + 2 if r.size else 0
    n2 = nt + n_pad + band_pad(cfg)
    freqs_t = _t(np.fft.rfftfreq(n2, 1.0 / fs))
    spec = torch.fft.rfft(_t(waveform), n=n2)
    out = np.empty((r.size, nt), np.float32)
    chunk = max(1, int(6e6 // max(freqs_t.numel(), 1)))
    for i0 in range(0, r.size, chunk):
        r_t = _t(r[i0 : i0 + chunk], dtype=torch.float32)
        resp = _range_response(freqs_t, r_t, cfg, spreading=geometry, attenuation=geometry)
        block = torch.fft.irfft(spec.reshape(1, -1) * resp, n=n2, dim=1)
        out[i0 : i0 + chunk] = _np(block[:, :nt])
    return out


def _fft_delay(waveform, fs, taus) -> np.ndarray:
    """Pure fractional delays of ``waveform`` by each of ``taus`` seconds."""
    waveform = np.asarray(waveform, np.float32).ravel()
    nt = waveform.size
    freqs_t = _t(np.fft.rfftfreq(nt, 1.0 / fs))
    spec = torch.fft.rfft(_t(waveform), n=nt)
    tau_t = _t(np.atleast_1d(taus)).reshape(-1, 1)
    angle = (-2.0 * np.pi) * freqs_t.reshape(1, -1) * tau_t
    phase = torch.polar(torch.ones_like(angle), angle)
    return _np(torch.fft.irfft(spec.reshape(1, -1) * phase, n=nt, dim=1))


def channel_positions(cfg) -> np.ndarray:
    return cfg["x_offset"] + (np.arange(cfg["n_channels"]) + 0.5) * cfg["channel_spacing"]


def gauge_taps(ch_pos, cfg) -> tuple[np.ndarray, bool]:
    """Tap positions the gauge operator differences, and whether it is active."""
    gauge = float(cfg["gauge_length"])
    if gauge <= 0:
        return np.asarray(ch_pos, np.float64), False
    return np.concatenate([ch_pos + 0.5 * gauge, ch_pos - 0.5 * gauge]), True


def _gauge_combine(field, cfg) -> np.ndarray:
    gauge = float(cfg["gauge_length"])
    if gauge <= 0:
        return field
    n = field.shape[0] // 2
    return (field[:n] - field[n:]) / gauge


def _propagate_gauge(waveform, cfg, x_src, ch_pos, y=0.0, z=0.0) -> np.ndarray:
    taps, _ = gauge_taps(ch_pos, cfg)
    return _gauge_combine(_propagate(waveform, cfg, taps - float(x_src), y, z), cfg)


# --------------------------------------------------------------------------
# moving sources: retarded time on the same band triple
# --------------------------------------------------------------------------


def _upsampled_bands(waveform, cfg, oversample: int) -> torch.Tensor:
    """``(n_bands, nt * oversample)`` band split, each upsampled for interpolation."""
    waveform = np.asarray(waveform, np.float32).ravel()
    nt = waveform.size
    fs = float(cfg["fs"])
    n2 = nt + band_pad(cfg)
    freqs = np.fft.rfftfreq(n2, 1.0 / fs)
    weights = _t(band_weights(freqs, cfg))
    spec = torch.fft.rfft(_t(waveform), n=n2).reshape(1, -1) * weights
    n_up = n2 * int(oversample)
    padded = torch.zeros((spec.shape[0], n_up // 2 + 1), dtype=torch.complex64, device=device())
    padded[:, : spec.shape[1]] = spec
    if n2 % 2 == 0 and spec.shape[1] > 1:
        padded[:, spec.shape[1] - 1] *= 0.5
    return torch.fft.irfft(padded, n=n_up, dim=1) * float(oversample)


def _gather_linear(buffer: torch.Tensor, index: torch.Tensor, clamp=False) -> torch.Tensor:
    """Linear interpolation of ``buffer`` (1-D) at fractional sample ``index``.

    Out-of-range indices read as zero, or hold the end value when ``clamp`` is
    set -- a source waveform stops outside the record, but the trajectory it
    follows has to keep a sensible position at negative retarded time.
    """
    n = buffer.numel()
    valid = True if clamp else ((index >= 0) & (index <= n - 1))
    clamped = torch.clamp(index, 0.0, float(n - 1))
    # clamp the integer part rather than the float: at large n, float32 rounds
    # `n - 1 - eps` back up to `n - 1` and the gather runs off the end.
    i0 = torch.clamp(clamped.floor().to(torch.long), 0, n - 2)
    frac = clamped - i0.to(clamped.dtype)
    flat = i0.reshape(-1)
    lo = buffer[flat].reshape(index.shape)
    hi = buffer[flat + 1].reshape(index.shape)
    out = lo + frac * (hi - lo)
    if clamp:
        return out
    return torch.where(valid, out, torch.zeros((), device=index.device))


def _retarded(track_t, vel_t, px, y2z2, slowness, att, near, focus, fs, nt, coarse):
    """Retarded time and amplitude of a moving source at tap positions ``px``.

    Solves ``tau = t - r(tau) * slowness`` by fixed-point iteration.  The
    geometry varies far more slowly than the waveform, so the solve runs on a
    decimated time grid and the result is interpolated back to full rate.
    """
    n_coarse = max(2, int(np.ceil((nt - 1) / coarse)) + 1)
    duration = (nt - 1) / fs
    t_c = torch.linspace(0.0, duration, n_coarse, device=px.device).reshape(1, -1)
    px_col = px.reshape(-1, 1)

    def track_at(tau):
        index = tau * fs
        return (
            _gather_linear(track_t, index, clamp=True),
            _gather_linear(vel_t, index, clamp=True),
        )

    s_tau, v_tau = track_at(t_c.expand(px.numel(), -1))
    for _ in range(3):
        r = torch.sqrt((s_tau - px_col) ** 2 + y2z2)
        s_tau, v_tau = track_at(t_c - r * slowness)
    r = torch.sqrt((s_tau - px_col) ** 2 + y2z2)
    tau = t_c - r * slowness

    # dt/dtau = 1 + dr/dtau * slowness; the reciprocal is the Doppler factor.
    drdtau = (s_tau - px_col) * v_tau / torch.clamp(r, min=1e-6)
    jac = 1.0 / torch.clamp(1.0 + drdtau * slowness, min=0.5, max=2.0)
    gain = torch.exp(-att * r) / torch.sqrt(torch.clamp(r, min=near))
    if focus:
        gain = gain * (jac**focus)

    def to_full(x):
        return torch.nn.functional.interpolate(
            x.unsqueeze(1), size=nt, mode="linear", align_corners=True
        ).squeeze(1)

    return to_full(tau), to_full(gain)


def moving_field(cfg, tap_pos, track, velocity, waveform, y, z, oversample=8, out=None):
    """Field of a source following ``track``, sampled at every tap position.

    Uses the same band triple as :func:`_propagate`, evaluated at the retarded
    time, so a stationary track reproduces the frequency-domain operator.
    Accumulates into ``out`` when given and returns the tensor it wrote to.
    """
    fs = float(cfg["fs"])
    nt = np.asarray(waveform).size
    centres, slow, att = band_plan(cfg)
    bands = _upsampled_bands(waveform, cfg, oversample)
    track_t = _t(np.broadcast_to(np.asarray(track, np.float64), (nt,)))
    vel_t = _t(np.broadcast_to(np.asarray(velocity, np.float64), (nt,)))
    y2z2 = float(y) ** 2 + float(z) ** 2
    coarse = max(1, int(round(fs * float(cfg.get("geometry_step", 0.01)))))
    focus = float(cfg.get("doppler_focus", 1.0))
    near = float(cfg["near_field"])

    if out is None:
        out = torch.zeros((len(tap_pos), nt), dtype=torch.float32, device=device())
    chunk = max(1, int(cfg.get("tap_chunk", 128)))
    for i0 in range(0, len(tap_pos), chunk):
        px = _t(tap_pos[i0 : i0 + chunk])
        block = out[i0 : i0 + px.numel()]
        for b in range(centres.size):
            tau, gain = _retarded(
                track_t, vel_t, px, y2z2, float(slow[b]), float(att[b]),
                near, focus, fs, nt, coarse,
            )
            block += _gather_linear(bands[b], tau * (fs * oversample)) * gain
    return out


# --------------------------------------------------------------------------
# source waveforms
# --------------------------------------------------------------------------


def _ricker(f0: float, fs: float, half_width: float | None = None) -> np.ndarray:
    half_width = half_width if half_width is not None else max(0.02, 3.0 / f0)
    n = max(9, int(2 * half_width * fs) | 1)
    t = (np.arange(n) - n // 2) / fs
    a = (np.pi * f0 * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return (w / np.abs(w).max()).astype(np.float32)


def _shaped_noise(nt, fs, rng, exponent=0.0, fmin=0.05, fmax=None):
    fmax = fmax if fmax is not None else fs / 2
    spec = np.fft.rfft(rng.standard_normal(nt))
    f = np.fft.rfftfreq(nt, 1.0 / fs)
    f[0] = f[1]
    shape = f ** (-exponent / 2.0)
    shape[(f < fmin) | (f > fmax)] = 0.0
    y = np.fft.irfft(spec * shape, n=nt)
    std = y.std()
    return (y / std if std > 1e-12 else y).astype(np.float32)


def _shaped_noise_multi(nch, nt, fs, rng, exponent=0.0, fmin=0.05, fmax=None):
    fmax = fmax if fmax is not None else fs / 2
    freqs = np.fft.rfftfreq(nt, 1.0 / fs)
    fc = np.maximum(freqs, fmin)
    shape = fc ** (-exponent / 2.0)
    shape[(freqs < fmin) | (freqs > fmax)] = 0.0
    white = _t(rng.standard_normal((nch, nt)))
    y = torch.fft.irfft(torch.fft.rfft(white, n=nt, dim=1) * _t(shape), n=nt, dim=1)
    y = y - y.mean(dim=1, keepdim=True)
    return _np(y / torch.clamp(y.std(dim=1, keepdim=True), min=1e-12))


def _band_noise(nt, fs, rng, fmin, fmax):
    return _shaped_noise(nt, fs, rng, exponent=0.0, fmin=fmin, fmax=fmax)


def _axle_wavelet(nt, fs, rng, fmin, fmax, f0=55.0, rumble=0.22, impulse_rate=None):
    w = rumble * _band_noise(nt, fs, rng, fmin, fmax)
    rate = impulse_rate if impulse_rate else rng.uniform(2.5, 5.0)
    e = np.zeros(nt, np.float32)
    t = rng.uniform(0.0, 1.0 / rate)
    while t < nt / fs:
        hit = rng.uniform(0.5, 1.5) * _ricker(f0 * rng.uniform(0.8, 1.2), fs)
        i = int(round(t * fs))
        j = min(hit.size, nt - i)
        if 0 < j:
            e[i : i + j] += hit[:j]
        t += (1.0 / rate) * max(0.5, 1.0 + 0.15 * rng.standard_normal())
    return (w + e).astype(np.float32)


def _flat_impulses(n, fs, rng, period, amp):
    e = np.zeros(n, np.float32)
    t = rng.uniform(0.0, period)
    while t < n / fs:
        w = _ricker(rng.uniform(120.0, 200.0), fs)
        i = int(round(t * fs))
        j = min(w.size, n - i)
        if 0 < j:
            e[i : i + j] += amp * w[:j]
        t += period * max(0.5, 1.0 + 0.03 * rng.standard_normal())
    return e


def _engine_wavelet(n, fs, rng, f0, amp):
    t = np.arange(n) / fs
    w = np.zeros(n, np.float32)
    f0 = f0 * rng.uniform(0.9, 1.1)
    for k, a in enumerate((1.0, 0.5, 0.3, 0.15), start=1):
        w += a * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0.0, 2 * np.pi)).astype(np.float32)
    return (amp * w / 2.0).astype(np.float32)


def _footstep_wavelet(nt, fs, rng, ev):
    w = np.zeros(nt, np.float32)
    t = float(ev["t_start"])
    while t < ev["t_end"]:
        for frac, weight in ((0.0, 1.0), (0.13, 0.5)):
            th = t + frac
            if th >= ev["t_end"]:
                break
            hit = _ricker(ev["f0"] * rng.uniform(0.75, 1.25), fs)
            i = int(round(th * fs))
            j = min(hit.size, nt - i)
            if 0 <= i < nt and j > 0:
                w[i : i + j] += weight * rng.uniform(0.7, 1.3) * hit[:j]
        t += (1.0 / ev["rate"]) * max(0.2, 1.0 + 0.08 * rng.standard_normal())
    return w


def _digging_wavelet(nt, fs, rng, ev):
    w = np.zeros(nt, np.float32)
    t = float(ev["t_start"])
    duration = nt / fs
    for _ in range(int(ev["n_hits"])):
        if t >= duration:
            break
        f0 = ev["f0"] * rng.uniform(0.7, 1.3)
        half_width = max(0.02, 3.0 / f0)  # one width, so the two rickers align
        hit = _ricker(f0, fs, half_width) + 0.35 * _ricker(3.0 * f0, fs, half_width)
        i = int(round(t * fs))
        if 0 <= i < nt:
            j = min(hit.size, nt - i)
            w[i : i + j] += rng.uniform(0.6, 1.4) * hit[:j]
        if rng.random() < ev.get("scrape_prob", 0.0):
            dur = rng.uniform(0.4, 1.2)
            scrape = _band_noise(nt, fs, rng, 150.0, 600.0)
            n0 = int(round(t * fs))
            m = min(int(dur * fs), nt - n0)
            if m > 0:
                w[n0 : n0 + m] += 0.35 * scrape[:m] * np.hanning(m).astype(np.float32)
            t += dur
        t += (1.0 / ev["rate"]) * max(0.2, 1.0 + 0.2 * rng.standard_normal())
    return w


def _burst_wavelet(nt, fs, rng, ev):
    n = min(int(ev["duration"] * fs), nt)
    w = _band_noise(nt, fs, rng, ev["fmin"], ev["fmax"])
    out = np.zeros(nt, np.float32)
    i0 = int(round(ev["t_start"] * fs))
    n = min(n, nt - i0)
    if n > 0:
        out[i0 : i0 + n] = w[:n] * np.exp(-np.linspace(0, 5, n)).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# site
# --------------------------------------------------------------------------


def cable_site(nch, rng, dead_fraction=0.02, noisy_fraction=0.015):
    """Per-channel coupling and instrument-noise gain of one installation.

    ``coupling`` scales everything that reaches the fibre through the ground --
    the events and the ground-borne background alike -- so a dead run suppresses
    both and cannot masquerade as a perfect-SNR event.  ``noise_gain`` scales
    only the fibre's own phase noise, which is what actually goes up on a bad
    splice or a low-return section.
    """
    smooth = np.exp(0.18 * gaussian_filter1d(rng.standard_normal(nch), 8.0))
    weak = np.ones(nch)
    n_weak = max(1, int(round(rng.uniform(0.03, 0.06) * nch)))
    for start in rng.choice(nch, size=n_weak, replace=False):
        width = int(rng.integers(2, 15))
        weak[int(start) : int(start) + width] *= rng.uniform(0.12, 0.7)

    noise_gain = np.ones(nch, np.float32)
    dead = []
    for start in rng.choice(nch, size=min(int(round(dead_fraction * nch)), nch), replace=False):
        end = min(int(start) + int(rng.integers(1, 4)), nch)
        weak[int(start) : end] *= 0.02
        noise_gain[int(start) : end] *= rng.uniform(1.5, 3.0)
        dead.extend(range(int(start), end))
    noisy = []
    for start in rng.choice(nch, size=min(int(round(noisy_fraction * nch)), nch), replace=False):
        end = min(int(start) + int(rng.integers(1, 4)), nch)
        noise_gain[int(start) : end] *= rng.uniform(2.5, 5.0)
        noisy.extend(range(int(start), end))

    coupling = (smooth * gaussian_filter1d(weak, 0.8)).astype(np.float32)
    return coupling, noise_gain, sorted(set(dead)), sorted(set(noisy))


def _position_track(nt, fs, rng, x0, speed, jitter):
    """Source position and velocity with a slowly drifting speed."""
    if not jitter:
        v = np.full(nt, float(speed), np.float64)
    else:
        w = _shaped_noise(nt, fs, rng, exponent=2.0, fmin=0.02, fmax=0.3)
        v = speed * (1.0 + float(jitter) * w)
    x = x0 + np.concatenate([[0.0], np.cumsum(v[:-1])]) / fs
    return x.astype(np.float64), v.astype(np.float64)


def _coda_spec(fs, rng, strength=0.3, taps=2):
    return [
        (int(rng.uniform(0.04, 0.5) * fs), strength * rng.uniform(0.3, 1.0), rng.uniform(0.5, 1.5))
        for _ in range(taps)
    ]


def _apply_coda(field, spec):
    """Delayed, attenuated and channel-diffused copies: scattering coda."""
    if not spec:
        return field
    nt = field.shape[1]
    out = field
    for delay, amp, sigma in spec:
        if 0 < delay < nt:
            blurred = gaussian_filter1d(field, sigma=sigma, axis=0)
            out = out.copy()
            out[:, delay:] += amp * blurred[:, :-delay]
    return out


def _gauge_spatial(field, cfg):
    """Apply the gauge difference operator along the channel axis."""
    gauge = float(cfg["gauge_length"])
    dx = float(cfg["channel_spacing"])
    if gauge <= 0 or dx <= 0:
        return field
    half = 0.5 * gauge / dx
    n = field.shape[0]
    idx = np.arange(n, dtype=np.float64)

    def shift(offset):
        pos = np.clip(idx + offset, 0.0, n - 1.0)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        frac = (pos - i0)[:, None].astype(np.float32)
        return field[i0] * (1.0 - frac) + field[i1] * frac

    return (shift(half) - shift(-half)) / gauge


def _instrument_noise(nch, nt, fs, rng, cfg, exponent):
    """Optical phase noise: spatially correlated along the fibre, then gauged."""
    field = _shaped_noise_multi(nch, nt, fs, rng, exponent=exponent, fmin=0.03, fmax=0.5 * fs)
    sigma = float(cfg.get("noise_correlation_m", 2.0)) / max(float(cfg["channel_spacing"]), 1e-6)
    if sigma > 0.05:
        field = gaussian_filter1d(field, sigma, axis=0, mode="nearest")
    field = _gauge_spatial(field, cfg)
    return field / max(float(field.std()), 1e-12)


def _spatial_profile(nch, rng, scale=40.0):
    """Smooth, strictly positive weighting along the route (mean 1)."""
    p = np.exp(0.5 * gaussian_filter1d(rng.standard_normal(nch), scale, mode="wrap"))
    return (p / p.mean()).astype(np.float32)


# --------------------------------------------------------------------------
# background
# --------------------------------------------------------------------------


BACKGROUND_DEFAULTS = {
    "noise_pink": 0.09,
    "noise_white": 0.03,
    "noise_common": 0.04,
    "drift": 0.03,
    "microseism": 0.20,
    "hum": 0.015,
    "ambient_sources": 8,
    "ambient_level": 0.10,
    "gust": 0.5,
    "dead_fraction": 0.02,
    "noisy_fraction": 0.015,
    "glitch_rate": 0.06,
    "burst_rate": 0.08,
    "burst_gain": (3.0, 9.0),
}


def make_background(cfg, rng, rng_transient, ch_pos, coupling, noise_gain, env_decim):
    """Background field and a decimated map of the unlabelled transients."""
    nch, nt = len(ch_pos), int(cfg["duration"] * cfg["fs"])
    fs = float(cfg["fs"])
    t = np.arange(nt, dtype=np.float32) / fs
    add = dict(BACKGROUND_DEFAULTS)
    add.update(cfg.get("background", {}))

    traces = np.zeros((nch, nt), np.float32)
    gust = 1.0 + add["gust"] * _shaped_noise(nt, fs, rng, exponent=2.0, fmin=0.01, fmax=0.3)
    gust = np.clip(gust, 0.15, 3.0).astype(np.float32)

    pink = _instrument_noise(nch, nt, fs, rng, cfg, 1.6)
    white = _instrument_noise(nch, nt, fs, rng, cfg, 0.0)
    common = _shaped_noise(nt, fs, rng, exponent=1.5, fmin=0.02, fmax=30.0)
    drift = _shaped_noise(nt, fs, rng, exponent=2.0, fmin=0.005, fmax=0.5)

    micro = np.zeros((nch, nt), np.float32)
    for fmin, fmax in ((0.08, 0.2), (0.2, 0.5)):
        w = _shaped_noise(nt, fs, rng, exponent=0.0, fmin=fmin, fmax=fmax)
        velocity = rng.uniform(400.0, 1200.0) * rng.choice([-1.0, 1.0])
        micro += _fft_delay(w, fs, np.asarray(ch_pos, np.float64) / velocity)
    micro /= np.maximum(micro.std(), 1e-12)

    ambient = np.zeros((nch, nt), np.float32)
    n_sources = int(add["ambient_sources"])
    for _ in range(n_sources):
        x_src = rng.uniform(ch_pos[0] - 150.0, ch_pos[-1] + 150.0)
        y_src = float(rng.uniform(15.0, 250.0))
        w = _band_noise(nt, fs, rng, float(rng.uniform(0.5, 4.0)), float(rng.uniform(30.0, 120.0)))
        field = _propagate_gauge(w, cfg, x_src, ch_pos, y_src, 0.0)
        field /= max(float(field.std()), 1e-12)
        active = 0.03 + 2.5 * np.clip(
            _shaped_noise(nt, fs, rng, exponent=2.0, fmin=0.03, fmax=1.0), 0.0, None
        )
        ambient += field * active[None, :] * (rng.uniform(0.3, 1.0) / max(n_sources, 1))
    ambient *= add["ambient_level"]

    hum = np.zeros(nt, np.float32)
    for k, a in enumerate((1.0, 0.35, 0.15, 0.07)):
        f = 50.0 * (k + 1)
        am = 1.0 + 0.25 * np.sin(2 * np.pi * 0.03 * t + rng.uniform(0.0, 2 * np.pi))
        hum += a * (am * np.sin(2 * np.pi * f * t + rng.uniform(0.0, 2 * np.pi))).astype(np.float32)
    hum /= np.maximum(hum.std(), 1e-12)

    if coupling is not None:
        micro = micro * coupling[:, None]
        ambient = ambient * coupling[:, None]

    traces += add["noise_white"] * noise_gain[:, None] * white
    traces += add["noise_pink"] * noise_gain[:, None] * gust * pink
    traces += add["noise_common"] * _spatial_profile(nch, rng)[:, None] * common[None, :]
    traces += add["drift"] * _spatial_profile(nch, rng, 80.0)[:, None] * drift[None, :]
    traces += add["microseism"] * gust[None, :] * micro
    traces += gust[None, :] * ambient
    traces += add["hum"] * _spatial_profile(nch, rng, 25.0)[:, None] * hum[None, :]

    nte = int(np.ceil(nt / env_decim))
    transient = np.zeros((nch, nte), bool)

    def mark(ch0, ch1, i0, i1):
        transient[ch0:ch1, i0 // env_decim : (i1 - 1) // env_decim + 1] = True

    n_burst = int(rng_transient.poisson(add["burst_rate"] * nt / fs))
    for _ in range(n_burst):
        ch = int(rng_transient.integers(0, nch))
        dur = float(rng_transient.uniform(0.2, 2.0))
        m = max(int(dur * fs), 8)
        i0 = int(rng_transient.integers(0, max(nt - m, 1)))
        fmax = float(rng_transient.uniform(60.0, min(400.0, 0.9 * fs / 2)))
        w = _band_noise(m, fs, rng_transient, float(rng_transient.uniform(3.0, 0.8 * fmax)), fmax)
        w *= np.hanning(m).astype(np.float32)
        g0, g1 = add["burst_gain"]
        gain = float(rng_transient.uniform(g0, g1)) * max(float(np.median(np.abs(traces[ch]))), 1e-9)
        width = int(rng_transient.integers(1, 4))
        for k in range(width):
            if ch + k < nch:
                traces[ch + k, i0 : i0 + m] += (gain * (0.7**k)) * w
        mark(ch, min(ch + width, nch), i0, min(i0 + m, nt))

    chan_std = traces.std(axis=1)
    n_glitch = int(rng_transient.poisson(add["glitch_rate"] * nt / fs))
    for _ in range(n_glitch):
        ch = int(rng_transient.integers(0, nch))
        i = int(rng_transient.integers(0, max(nt - 1, 1)))
        amp = float(rng_transient.uniform(4.0, 12.0)) * max(float(chan_std[ch]), 1e-9)
        sign = float(rng_transient.choice([-1.0, 1.0]))
        traces[ch, i] += sign * amp
        traces[ch, i + 1] -= 0.6 * sign * amp
        end = i + 2
        if rng_transient.random() < 0.6:
            i0 = i + 1
            i1 = min(nt, i0 + int(rng_transient.integers(10, 120)))
            n = max(i1 - i0, 1)
            ring = np.exp(-np.linspace(0.0, 6.0, n)) * np.sin(
                2 * np.pi * rng_transient.uniform(200.0, 450.0) * np.arange(n) / fs
            )
            traces[ch, i0:i1] += (0.4 * amp * ring).astype(np.float32)
            end = i1
        mark(ch, ch + 1, i, min(end, nt))

    info = {
        "n_transient_bursts": int(n_burst),
        "n_glitches": int(n_glitch),
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in add.items()},
    }
    return traces, transient, info


def mix_background(traces, path, rng, level, cfg):
    """Mix a recorded background file in, recording exactly what was used.

    The file is never tiled: a recording that does not cover the synthetic grid
    is an error rather than a silently repeated pattern.  Its content is
    unlabelled, so the whole record is flagged and the caller marks it ignore.
    """
    with h5py.File(path, "r") as fb:
        real = np.asarray(fb["traces"][:], np.float32).T
    nch, nt = traces.shape
    if real.shape[0] < nch or real.shape[1] < nt:
        raise ValueError(
            f"background {path} is {real.shape} (channels, samples); "
            f"needs at least ({nch}, {nt}) -- tiling is not allowed"
        )
    ch0 = int(rng.integers(0, real.shape[0] - nch + 1))
    t0 = int(rng.integers(0, real.shape[1] - nt + 1))
    crop = real[ch0 : ch0 + nch, t0 : t0 + nt]
    scale = level * np.median(np.abs(traces)) / (np.median(np.abs(crop)) + 1e-12)
    cfg["background_mix"] = {
        "path": str(path),
        "level": float(level),
        "channel_offset": ch0,
        "sample_offset": t0,
        "scale": float(scale),
        "unlabelled": True,
    }
    return traces + scale * crop


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def signal_envelope(field, cfg, env_decim):
    """Decimated RMS envelope of ``field`` above the drift band.

    Labels, SNR calibration and the validator all measure through this one
    recipe, so "6 dB above background" means the same thing everywhere.
    """
    fs = float(cfg["fs"])
    b, a = butter(2, max(1.0, 0.002 * fs) / (fs / 2), btype="high")
    x = filtfilt(b, a, np.asarray(field, np.float64), axis=1)
    win = max(1, int(round(float(cfg.get("envelope_window", 0.2)) * fs)))
    # the running sum can land a few ulp below zero wherever the field is
    # identically zero, which sqrt turns into NaN
    energy = np.maximum(uniform_filter1d(x * x, size=win, axis=1, mode="nearest"), 0.0)
    return np.sqrt(energy[:, ::env_decim]).astype(np.float32)


def build_labels(event_env, event_classes, background_env, transient, cfg, nt, env_decim):  # noqa: PLR0913
    """SNR-based class mask with ignore bands, plus the per-event SNR maps."""
    hi = 10.0 ** (float(cfg["label_hi_db"]) / 20.0)
    lo = 10.0 ** (float(cfg["label_lo_db"]) / 20.0)
    margin = 10.0 ** (float(cfg["label_margin_db"]) / 20.0)
    floor = np.maximum(background_env, 1e-30)[None, :, :]
    snr = np.asarray(event_env, np.float32) / floor

    if snr.shape[0] == 0:
        coarse = np.zeros(background_env.shape, np.uint8)
        winner = np.zeros(background_env.shape, int)
    else:
        order = np.argsort(snr, axis=0)
        winner = order[-1]
        top1 = np.take_along_axis(snr, order[-1:], axis=0)[0]
        top2 = (
            np.take_along_axis(snr, order[-2:-1], axis=0)[0]
            if snr.shape[0] > 1
            else np.zeros_like(top1)
        )
        classes = np.asarray(event_classes, np.uint8)
        coarse = np.zeros(top1.shape, np.uint8)
        is_event = top1 >= hi
        coarse[is_event] = classes[winner][is_event]
        ambiguous = is_event & (top2 >= hi) & (top1 < top2 * margin)
        coarse[ambiguous] = IGNORE_ID
        coarse[(~is_event) & (top1 >= lo)] = IGNORE_ID

    # Only the transients need growing: the envelope window smears their energy
    # about half a window either side of the samples they actually occupy.  The
    # hi/lo band is already the guard band everywhere else, and dilating it too
    # costs ~10% of the record for no extra safety.
    grow = int(cfg.get("label_transient_grow", -1))
    if grow < 0:
        window = float(cfg.get("envelope_window", 0.2))
        grow = int(np.ceil(0.5 * window * float(cfg["fs"]) / env_decim))
    if grow > 0 and transient.any():
        transient = maximum_filter1d(transient.astype(np.uint8), 2 * grow + 1, axis=1).astype(bool)
    coarse[transient] = IGNORE_ID

    mask = np.repeat(coarse, env_decim, axis=1)[:, :nt]
    if mask.shape[1] < nt:
        mask = np.pad(mask, ((0, 0), (0, nt - mask.shape[1])), mode="edge")
    return mask, coarse, winner


def _boxes_from_mask(coarse, winner, event_classes, event_names, ch_pos, cfg, env_decim):
    fs = float(cfg["fs"])
    boxes = []
    for index, (cls_id, name) in enumerate(zip(event_classes, event_names)):
        sel = (coarse == cls_id) & (winner == index)
        if not sel.any():
            boxes.append(
                {"class": name, "t0": None, "t1": None, "x0": None, "x1": None, "labelled": 0}
            )
            continue
        ch_idx = np.flatnonzero(sel.any(axis=1))
        t_idx = np.flatnonzero(sel.any(axis=0))
        boxes.append({
            "class": name,
            "t0": round(float(t_idx[0] * env_decim / fs), 3),
            "t1": round(float(min((t_idx[-1] + 1) * env_decim, cfg["duration"] * fs) / fs), 3),
            "x0": round(float(ch_pos[ch_idx[0]]), 1),
            "x1": round(float(ch_pos[ch_idx[-1]]), 1),
            "labelled": int(sel.sum()),
        })
    return boxes


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------


def default_scenario(cfg=None):
    """A single fixed scene.  Use ``make_dataset.py`` for randomised scenarios."""
    return [
        {
            "kind": "train", "class": "train", "id": "train0",
            "x0": -225.0, "speed": 30.0, "y": 7.0, "z": 1.2, "snr_db": 12.0,
            "qs_ratio": 0.5, "fmin": 15.0, "fmax": 350.0, "f0": 55.0,
            "axle_offsets": [0.0, 2.5, 13.0, 15.5, 26.0, 28.5, 39.0, 41.5],
        },
        {
            "kind": "wheel_flat", "class": "wheel_flat", "id": "flat0",
            "x0": 720.0, "speed": -15.0, "y": 7.0, "z": 1.2, "snr_db": 10.0,
            "qs_ratio": 0.5, "fmin": 15.0, "fmax": 350.0, "f0": 55.0,
            "axle_offsets": [0.0, 2.5, 13.0, 15.5, 26.0, 28.5],
            "wheel_period": 0.2, "flat_amp": 0.25,
        },
        {
            "kind": "road_vehicle", "class": "road_vehicle", "id": "road0",
            "x0": 560.0, "speed": -16.0, "y": 14.0, "z": 0.8, "snr_db": 8.0,
            "qs_ratio": 0.18, "fmin": 8.0, "fmax": 120.0, "length": 6.0,
            "engine_f0": 45.0, "engine_amp": 0.27,
        },
        {
            "kind": "walker", "class": "footsteps", "id": "walk0",
            "x0": 80.0, "speed": 1.3, "t_start": 21.0, "t_end": 31.0,
            "y": 1.5, "z": 0.05, "rate": 1.8, "snr_db": 6.0, "f0": 35.0,
        },
        {
            "kind": "static", "class": "digging", "id": "dig0",
            "x": 250.0, "y": 12.0, "z": 0.4, "t_start": 29.0, "n_hits": 24,
            "rate": 2.4, "snr_db": 9.0, "f0": 22.0, "scrape_prob": 0.4,
        },
        {
            "kind": "burst", "class": "burst", "id": "burst0",
            "x": 60.0, "y": 8.0, "z": 0.2, "t_start": 12.0, "duration": 2.5,
            "snr_db": 7.0, "fmin": 100.0, "fmax": 450.0,
        },
    ]


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------


def _event_rng(cfg, event, index):
    key = event.get("id", index)
    entropy = [int(cfg["seed"]), 9001, index]
    if isinstance(key, str):
        entropy.append(int.from_bytes(key.encode()[:8].ljust(8, b"\0"), "little"))
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _render_event(cfg, ev, ch_pos, rng, coupling):
    """Clean, unit-amplitude field of one event: ``(n_channels, nt)``."""
    fs = float(cfg["fs"])
    nt = int(cfg["duration"] * fs)
    kind = ev["kind"]
    taps, gauged = gauge_taps(ch_pos, cfg)
    oversample = int(cfg.get("oversample", 8))

    if kind in MOVING_KINDS:
        if kind == "walker":
            sources = [(0.0, _footstep_wavelet(nt, fs, rng, ev))]
        else:
            offsets = ev.get("axle_offsets")
            if offsets is None:
                n = max(1, int(round(ev["length"] / 2.5)))
                offsets = [2.5 * i for i in range(n)]
            sources = []
            for k, offset in enumerate(offsets):
                w = _axle_wavelet(
                    nt, fs, rng, ev["fmin"], ev["fmax"],
                    f0=ev.get("f0", 55.0), rumble=ev.get("rumble", 0.22),
                )
                if kind == "road_vehicle":
                    w = w + _engine_wavelet(nt, fs, rng, ev.get("engine_f0", 45.0),
                                            ev.get("engine_amp", 0.27))
                if kind == "wheel_flat" and k == 0:
                    w = w + _flat_impulses(nt, fs, rng, ev.get("wheel_period", 0.2),
                                           ev.get("flat_amp", 0.25))
                sources.append((float(offset), w.astype(np.float32)))
            if ev.get("qs_ratio"):
                sources.append((
                    0.5 * (max(offsets) + min(offsets)),
                    (float(ev["qs_ratio"]) * _band_noise(nt, fs, rng, 0.3, 1.5)).astype(np.float32),
                ))
        track, velocity = _position_track(
            nt, fs, rng, ev["x0"], ev["speed"], ev.get("speed_jitter", 0.02)
        )
        direction = 1.0 if ev["speed"] >= 0 else -1.0
        acc = torch.zeros((len(taps), nt), dtype=torch.float32, device=device())
        for offset, waveform in sources:
            moving_field(
                cfg, taps, track + direction * offset, velocity, waveform,
                ev["y"], ev["z"], oversample=oversample, out=acc,
            )
        field = _np(acc)
        del acc
        if kind == "walker":
            window = np.zeros(nt, np.float32)
            i0 = int(max(0, ev["t_start"] - 1.0) * fs)
            i1 = int(min(cfg["duration"], ev["t_end"] + 3.0) * fs)
            window[i0:i1] = 1.0
            field = field * window[None, :]
    elif kind in STATIC_KINDS:
        waveform = _digging_wavelet(nt, fs, rng, ev) if kind == "static" else _burst_wavelet(nt, fs, rng, ev)
        field = _propagate(waveform, cfg, taps - float(ev["x"]), ev["y"], ev["z"])
    else:
        raise ValueError(f"unknown event kind: {kind}")

    field = _gauge_combine(field, cfg) if gauged else field
    coda = ev.get("coda", cfg.get("coda", 0.3))
    if coda:
        field = _apply_coda(field, _coda_spec(fs, rng, coda))
    if coupling is not None:
        field = field * coupling[:, None]
    return np.ascontiguousarray(field, dtype=np.float32)


def _calibrate(field, background_env, cfg, env_decim, snr_db):
    """Scale ``field`` so its envelope sits ``snr_db`` above the background."""
    env = signal_envelope(field, cfg, env_decim)
    peak = float(np.percentile(env, 99.9))
    if peak <= 0:
        return 0.0, env
    active = env > 0.1 * peak
    if not active.any():
        active = env > 0
    channels = active.any(axis=1)
    floor = float(np.median(background_env[channels])) if channels.any() else float(np.median(background_env))
    target = (10.0 ** (float(snr_db) / 20.0)) * max(floor, 1e-30)
    scale = target / peak
    return scale, env * scale


def synthesize(cfg, scenario):
    fs = float(cfg["fs"])
    nt = int(cfg["duration"] * fs)
    ch_pos = channel_positions(cfg)
    env_decim = max(1, int(round(fs / float(cfg.get("envelope_rate", 40.0)))))
    cfg["envelope_decimation"] = env_decim

    root = np.random.SeedSequence(int(cfg["seed"]))
    rng_site, rng_bg, rng_transient = (np.random.default_rng(s) for s in root.spawn(3))

    settings = dict(BACKGROUND_DEFAULTS)
    settings.update(cfg.get("background", {}))
    coupling, noise_gain, dead, noisy = cable_site(
        cfg["n_channels"], rng_site,
        dead_fraction=settings["dead_fraction"],
        noisy_fraction=settings["noisy_fraction"],
    )
    background, transient, bg_info = make_background(
        cfg, rng_bg, rng_transient, ch_pos, coupling, noise_gain, env_decim
    )
    bg_info["dead_channels"] = dead
    bg_info["noisy_channels"] = noisy
    background_env = signal_envelope(background, cfg, env_decim)

    signal = np.zeros_like(background)
    event_env, event_classes, event_names, realised = [], [], [], []
    for index, ev in enumerate(scenario):
        rng = _event_rng(cfg, ev, index)
        field = _render_event(cfg, ev, ch_pos, rng, coupling)
        scale, env = _calibrate(field, background_env, cfg, env_decim, ev["snr_db"])
        signal += (scale * field).astype(np.float32)
        event_env.append(env)
        event_classes.append(CLASS_IDS[ev["class"]])
        event_names.append(ev["class"])
        realised.append({"id": ev.get("id", index), "class": ev["class"],
                         "snr_db": float(ev["snr_db"]), "amplitude": float(scale)})
        del field

    event_env = np.asarray(event_env, np.float32) if event_env else np.zeros((0,) + background_env.shape, np.float32)
    mask, coarse, winner = build_labels(
        event_env, event_classes, background_env, transient, cfg, nt, env_decim
    )
    boxes = _boxes_from_mask(coarse, winner, event_classes, event_names, ch_pos, cfg, env_decim)
    for box, info in zip(boxes, realised):
        box.update({"id": info["id"], "snr_db": info["snr_db"]})

    scale = 1.0
    if cfg.get("noise_std"):
        stds = background.std(axis=1)
        well_coupled = coupling >= np.median(coupling)
        quiet = float(np.median(stds[well_coupled]))
        scale = float(cfg["noise_std"]) / max(quiet, 1e-30)
        signal = signal * scale
        background = background * scale
    # sum the scaled parts, so traces == signal + background holds exactly in
    # float32 rather than to within a rescaling round-off
    traces = signal + background
    cfg["realized_scale"] = scale
    cfg["background_info"] = bg_info
    cfg["events_realized"] = realised

    return {
        "traces": traces,
        "signal": signal,
        "background": background,
        "mask": mask,
        "ch_pos": ch_pos,
        "boxes": boxes,
        "coupling": coupling,
        "event_snr_db": (20.0 * np.log10(np.maximum(event_env / np.maximum(background_env[None], 1e-30), 1e-6))).astype(np.float32),
        "event_ids": [str(e.get("id", i)) for i, e in enumerate(scenario)],
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def write_outputs(out_dir: Path, cfg, result, scenario, emit_components=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "das.h5"
    traces = np.ascontiguousarray(result["traces"].T)
    mask = np.ascontiguousarray(result["mask"].T)
    nch = cfg["n_channels"]
    t0_ns = 1_700_000_000 * 10**9
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("traces", data=traces, compression="gzip", compression_opts=1)
        f.attrs["format"] = "miniDAS"
        f.attrs["version"] = "0.1.0-synthetic"
        f.attrs["data_units"] = "rad/s"
        # traces are already in rad/s; a reader must not rescale them
        f.attrs["scale_factor"] = np.float32(1.0)
        f.attrs["units_after_scaling"] = "rad/s"
        f.attrs["start_time"] = np.uint64(t0_ns)
        f.attrs["sampling_rate"] = np.float32(cfg["fs"])
        f.attrs["gauge_length"] = np.float32(cfg["gauge_length"])
        f.attrs["latitudes"] = np.full(nch, 63.39, np.float32)
        f.attrs["longitudes"] = np.linspace(
            10.40, 10.40 + nch * cfg["channel_spacing"] / 111_000, nch
        ).astype(np.float32)
        f.attrs["elevations"] = np.zeros(nch, np.float32)
        f.attrs["meta_json"] = json.dumps({"config": cfg, "scenario": scenario})
        f.create_dataset("labels/class_mask", data=mask, compression="gzip", compression_opts=1)
        f.attrs["class_ids"] = json.dumps(CLASS_IDS)
        f.attrs["ignore_id"] = np.uint8(IGNORE_ID)

    if emit_components:
        with h5py.File(out_dir / "components.h5", "w") as f:
            f.create_dataset("signal", data=np.ascontiguousarray(result["signal"].T),
                             compression="gzip", compression_opts=1)
            f.create_dataset("background", data=np.ascontiguousarray(result["background"].T),
                             compression="gzip", compression_opts=1)
            f.create_dataset("coupling", data=result["coupling"])
            f.create_dataset("event_snr_db", data=result["event_snr_db"],
                             compression="gzip", compression_opts=1)
            f.attrs["event_ids"] = json.dumps(result["event_ids"])
            f.attrs["envelope_decimation"] = np.int32(cfg["envelope_decimation"])
            f.attrs["note"] = "traces == signal + background; both in rad/s"

    with open(out_dir / "labels.csv", "w") as fh:
        fh.write("class,id,t0,t1,x0,x1,snr_db,labelled_cells\n")
        for b in result["boxes"]:
            if b["t0"] is None:
                fh.write(f"{b['class']},{b['id']},,,,,{b['snr_db']},0\n")
            else:
                fh.write(f"{b['class']},{b['id']},{b['t0']},{b['t1']},{b['x0']},{b['x1']},"
                         f"{b['snr_db']},{b['labelled']}\n")
    with open(out_dir / "scenario.json", "w") as fh:
        json.dump({"config": cfg, "scenario": scenario}, fh, indent=2)
    return h5_path


def qa_plot(h5_path: Path, out_png: Path, ch_pos, cfg):
    with h5py.File(h5_path, "r") as f:
        traces = f["traces"][:]
        mask = f["labels/class_mask"][:]
    fs = float(cfg["fs"])
    t = np.arange(traces.shape[0]) / fs
    b, a = butter(4, (1.0 / (fs / 2), min(200.0, 0.9 * fs / 2) / (fs / 2)), btype="band")
    bp = filtfilt(b, a, traces, axis=0)

    from scipy.signal import welch

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    ext = [t[0], t[-1], ch_pos[0], ch_pos[-1]]

    gain = np.percentile(np.abs(traces), 99)
    db = 20 * np.log10(np.abs(traces) / (gain + 1e-12) + 1e-6)
    im = axes[0].imshow(db.T, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=-40, vmax=20)
    axes[0].set_ylabel("distance along fibre (m)")
    axes[0].set_title("raw strain-rate (dB rel. p99)")
    fig.colorbar(im, ax=axes[0], label="dB")

    gain_bp = np.percentile(np.abs(bp), 99)
    db_bp = 20 * np.log10(np.abs(bp) / (gain_bp + 1e-12) + 1e-6)
    im = axes[1].imshow(db_bp.T, aspect="auto", origin="lower", extent=ext, cmap="magma", vmin=-40, vmax=20)
    axes[1].set_ylabel("distance (m)")
    axes[1].set_title("1-200 Hz band (events, ambient ridges, axle impulses)")
    fig.colorbar(im, ax=axes[1], label="dB")

    shown = np.where(mask == IGNORE_ID, 7, mask).astype(np.uint8)
    axes[2].imshow(shown.T, aspect="auto", origin="lower", extent=ext, cmap="tab10", vmin=0, vmax=9)
    axes[2].set_ylabel("distance (m)")
    axes[2].set_title("class mask (1=train 2=vehicle 3=footsteps 4=digging 5=burst "
                      "6=wheel_flat 7=ignore)")

    quiet = int(0.02 * traces.shape[1])
    loud = int(np.argmax(np.abs(traces).max(axis=0)))
    f, pxx_q = welch(traces[:, quiet], fs=fs, nperseg=min(2048, traces.shape[0]))
    f, pxx_l = welch(traces[:, loud], fs=fs, nperseg=min(2048, traces.shape[0]))
    axes[3].semilogx(f, 10 * np.log10(pxx_q + 1e-30), label=f"quiet ch {quiet} ({ch_pos[quiet]:.0f} m)")
    axes[3].semilogx(f, 10 * np.log10(pxx_l + 1e-30), label=f"active ch {loud} ({ch_pos[loud]:.0f} m)")
    axes[3].set_title("PSD")
    axes[3].set_xlim(0.01, 500)
    axes[3].set_ylabel("dB/Hz")
    axes[3].set_xlabel("frequency (Hz)")
    axes[3].legend(loc="lower left", fontsize=8)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def build_config(args) -> dict:
    cfg = {
        "fs": args.fs,
        "duration": args.duration,
        "n_channels": args.n_channels,
        "channel_spacing": args.channel_spacing,
        "gauge_length": args.gauge_length,
        "x_offset": 0.0,
        "wave_speed": args.wave_speed,
        "attenuation": args.attenuation,
        "attenuation_power": args.attenuation_power,
        "attenuation_ref_freq": 50.0,
        "dispersion": args.dispersion,
        "dispersion_ref_freq": 20.0,
        "near_field": 2.0,
        "noise_correlation_m": args.noise_correlation,
        "propagation_bands": args.propagation_bands,
        "band_fmin": 0.5,
        "oversample": args.oversample,
        "doppler_focus": args.doppler_focus,
        "geometry_step": 0.01,
        "envelope_rate": 40.0,
        "envelope_window": 0.2,
        "label_hi_db": args.label_hi_db,
        "label_lo_db": args.label_lo_db,
        "label_margin_db": args.label_margin_db,
        "label_transient_grow": args.label_transient_grow,
        "noise_std": args.noise_std,
        "seed": args.seed,
    }
    return cfg


def add_common_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--fs", type=float, default=1000.0)
    ap.add_argument("--n-channels", type=int, default=300)
    ap.add_argument("--channel-spacing", type=float, default=1.02)
    ap.add_argument("--gauge-length", type=float, default=10.0)
    ap.add_argument("--wave-speed", type=float, default=250.0)
    ap.add_argument("--attenuation", type=float, default=2e-4)
    ap.add_argument("--dispersion", type=float, default=0.08,
                    help="exponent of c(f) = c0 (f/20)^-beta; 0 disables dispersion")
    ap.add_argument("--attenuation-power", type=float, default=0.6)
    ap.add_argument("--noise-correlation", type=float, default=2.0,
                    help="spatial correlation length (m) of the instrument noise field")
    ap.add_argument("--propagation-bands", type=int, default=12,
                    help="frequency bands the shared propagation operator is quantised into")
    ap.add_argument("--oversample", type=int, default=8,
                    help="upsampling factor used for fractional delays of moving sources")
    ap.add_argument("--doppler-focus", type=float, default=1.0,
                    help="exponent on the exact dtau/dt Doppler amplitude factor")
    ap.add_argument("--label-hi-db", type=float, default=6.0,
                    help="signal-over-background ratio at which a cell gets its event class")
    ap.add_argument("--label-lo-db", type=float, default=3.0,
                    help="ratio below which a cell counts as background; between the two: ignore")
    ap.add_argument("--label-margin-db", type=float, default=6.0,
                    help="two overlapping events closer than this are labelled ignore")
    ap.add_argument("--label-transient-grow", type=int, default=-1,
                    help="envelope cells of ignore padding around unlabelled transients "
                         "(-1 derives it from the envelope window)")
    ap.add_argument("--noise-std", type=float, default=1e-7,
                    help="rescale traces to this quiet-channel std in rad/s (0 to keep raw units)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=str, default=None)


def main():
    ap = argparse.ArgumentParser(description="Generate one synthetic DAS recording.")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/run01"))
    add_common_arguments(ap)
    ap.add_argument("--scenario", type=Path, default=None)
    ap.add_argument("--background", type=Path, default=None,
                    help="miniDAS/h5 file mixed in as recorded background (never tiled)")
    ap.add_argument("--background-level", type=float, default=0.5)
    ap.add_argument("--background-json", type=Path, default=None,
                    help="JSON object overriding cfg['background'] knobs (ambient/transients)")
    ap.add_argument("--no-components", action="store_true",
                    help="skip the companion signal/background file")
    args = ap.parse_args()
    set_device(args.device)

    cfg = build_config(args)
    if args.background_json:
        cfg["background"] = json.loads(args.background_json.read_text())
    scenario = json.loads(args.scenario.read_text()) if args.scenario else default_scenario(cfg)
    result = synthesize(cfg, scenario)
    if args.background:
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 777]))
        result["traces"] = mix_background(
            result["traces"], args.background, rng, args.background_level, cfg
        )
        # the synthetic events are still where they were; it is the background
        # that now holds unknown content, so only class 0 becomes ignore
        result["mask"][result["mask"] == 0] = IGNORE_ID
        print("[warn] --background mixes unlabelled content: every background cell "
              "is marked ignore, event labels are kept")

    h5_path = write_outputs(args.out, cfg, result, scenario, emit_components=not args.no_components)
    png = qa_plot(h5_path, args.out / "qa.png", result["ch_pos"], cfg)
    traces = result["traces"]
    print(f"wrote {h5_path}  traces={traces.T.shape} ({traces.nbytes/1e6:.1f} MB)  "
          f"events={len(result['boxes'])}")
    print(f"wrote {png}")
    if args.noise_std:
        print(f"scaled to {args.noise_std:g} rad/s quiet-channel std "
              f"(factor {cfg.get('realized_scale', 1.0):.3g})")
    mask = result["mask"]
    counts = ", ".join(
        f"{CLASS_NAMES[k]}={int((mask == k).sum())}" for k in sorted(CLASS_NAMES) if (mask == k).any()
    )
    print(f"class mask coverage: {counts}, ignore={int((mask == IGNORE_ID).sum())}")


if __name__ == "__main__":
    main()
