"""DAS window datasets, miniDAS HDF5 loading, and synthetic data.

Data layout is ``(num_channels, num_samples)`` per recording: one row per
fiber channel, one column per time sample.

Supported sources:

* ``.npy``  -- memory-mapped 2-D array, ``(channels, time)``.
* ``.npz``  -- eagerly loaded array (pass ``key`` to select it).
* ``.h5``/``.hdf5`` -- miniDAS files with a ``traces`` dataset of shape
  ``(time, channels)`` (transposed on the fly) and an optional
  ``labels/class_mask`` dataset used for background filtering and anomaly
  evaluation.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


IGNORE_CLASS = 255
"""Label value for cells that must be excluded from training and scoring.

The generator marks the boundary band between event and background, cells where
two events overlap ambiguously, and anything touched by an unlabelled background
transient.  Treating these as class 0 is what made the background calibration
set contain the loudest transients in the file.
"""


class H5Transposed:
    """Channel-major view over a miniDAS HDF5 ``traces`` dataset."""

    ndim = 2

    def __init__(self, path: Path, key: str = "traces") -> None:
        import h5py

        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        if key not in self._file:
            self._file.close()
            raise ValueError(f"{self.path} has no dataset named {key!r}")
        self._dataset = self._file[key]
        self.shape = (self._dataset.shape[1], self._dataset.shape[0])
        self.dtype = self._dataset.dtype

    def __getitem__(self, item):
        channels, times = item
        return np.ascontiguousarray(self._dataset[times, channels].T)

    def __array__(self, dtype=None):
        data = np.ascontiguousarray(self._dataset[...].T)
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def close(self) -> None:
        self._file.close()

    def __del__(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def read_h5_metadata(path: str | Path) -> dict:
    """miniDAS attributes and scenario config from an HDF5 recording."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        meta = dict(handle.attrs)
        meta_json = meta.get("meta_json")
        if meta_json:
            meta["meta"] = json.loads(meta_json)
            meta["config"] = meta["meta"].get("config", {})
        class_ids = meta.get("class_ids")
        if class_ids:
            meta["class_ids"] = json.loads(class_ids)
        return meta


def load_h5_labels(
    path: str | Path,
    key: str = "labels/class_mask",
) -> np.ndarray | None:
    """Class mask ``(channels, time)`` from a miniDAS file, or ``None``."""
    import h5py

    with h5py.File(Path(path), "r") as handle:
        if key not in handle:
            return None
        return np.ascontiguousarray(handle[key][:].T)


def _load_source(
    path: Path,
    key: str | None = None,
    with_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix in (".h5", ".hdf5"):
        trace_key = key or "traces"
        array = H5Transposed(path, key=trace_key)
        labels = load_h5_labels(path) if with_labels else None
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            selected = key or next(iter(archive.files))
            array = np.asarray(archive[selected])
        labels = None
    else:
        array = np.asarray(np.load(path, mmap_mode="r"))
        labels = None
    if array.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D (channels, time) array")
    return array, labels


def _is_recording(path: Path, key: str = "traces") -> bool:
    if path.suffix not in (".h5", ".hdf5"):
        return True
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            return key in handle
    except OSError:
        return False


def discover_recordings(source: str | Path) -> list[Path]:
    """Sorted recording paths from a file or a (recursively searched) directory."""
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"{source} is neither a file nor a directory")
    paths: list[Path] = []
    for pattern in ("*.npy", "*.npz", "*.h5", "*.hdf5"):
        paths.extend(source.rglob(pattern))
    # HDF5 files without a `traces` dataset are companions, not recordings --
    # the generator writes components.h5 (clean signal and background) next to
    # every das.h5, and loading it as a recording would be a label-free duplicate.
    paths = sorted({p for p in set(paths) if _is_recording(p)})
    if not paths:
        raise FileNotFoundError(f"no .npy/.npz/.h5 recordings found under {source}")
    return paths


class WindowDataset(Dataset):
    """Sliding windows over one or more ``(C, T)`` DAS recordings."""

    def __init__(
        self,
        arrays: Sequence[np.ndarray],
        window_size: int,
        stride: int | None = None,
        channel_mean: np.ndarray | torch.Tensor | None = None,
        channel_std: np.ndarray | torch.Tensor | None = None,
        normalize: bool = True,
        labels: Sequence[np.ndarray | None] | None = None,
        metadata: list[dict] | None = None,
        return_labels: bool = False,
        highpass_hz: float | None = None,
        sample_rate: float | None = None,
    ) -> None:
        if not arrays:
            raise ValueError("at least one array is required")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.arrays = list(arrays)
        self.window_size = int(window_size)
        self.stride = int(stride or window_size)
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        self.normalize = bool(normalize)
        self.metadata = metadata or [{} for _ in self.arrays]
        self.return_labels = bool(return_labels)
        self.highpass_hz = float(highpass_hz) if highpass_hz else None
        self.sample_rate = float(sample_rate) if sample_rate else None
        if self.highpass_hz:
            if not self.sample_rate:
                rate = next(
                    (m.get("sampling_rate") for m in self.metadata if m.get("sampling_rate")),
                    None,
                )
                if rate is None:
                    raise ValueError("highpass_hz needs sample_rate (or miniDAS metadata)")
                self.sample_rate = float(rate)
            self.arrays = [self._highpass(a) for a in self.arrays]

        num_channels = self.arrays[0].shape[0]
        for array in self.arrays:
            if array.ndim != 2:
                raise ValueError("all arrays must be 2-D (channels, time)")
            if array.shape[0] != num_channels:
                raise ValueError("all arrays must have the same number of channels")
            if array.shape[1] < self.window_size:
                raise ValueError("array shorter than window_size")

        if labels is None:
            self.labels: list[np.ndarray | None] = [None] * len(self.arrays)
        else:
            if len(labels) != len(self.arrays):
                raise ValueError("labels must align with arrays")
            self.labels = list(labels)
            for array, label in zip(self.arrays, self.labels):
                if label is not None and label.shape != array.shape:
                    raise ValueError("label masks must match array shape (channels, time)")

        self.index: list[tuple[int, int]] = []
        for file_index, array in enumerate(self.arrays):
            length = array.shape[1]
            for start in range(0, length - self.window_size + 1, self.stride):
                self.index.append((file_index, start))

        self.channel_mean: torch.Tensor | None = None
        self.channel_std: torch.Tensor | None = None
        if channel_mean is not None and channel_std is not None:
            self.set_channel_stats(channel_mean, channel_std)

    def _highpass(self, array: np.ndarray) -> np.ndarray:
        """Strip the sub-Hz drift the events are buried under.

        Raw DAS is dominated by drift, common mode and the microseism; a
        broadband window is mostly energy no detector can use, which is why an
        unfiltered RMS detector scores chance on data where a 15-150 Hz one
        scores 0.78.  Every real pipeline filters first, so the model should
        see what the pipeline would hand it.
        """
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, self.highpass_hz / (self.sample_rate / 2), btype="high",
                     output="sos")
        return sosfiltfilt(sos, np.asarray(array, dtype=np.float32), axis=1).astype(np.float32)

    @classmethod
    def from_paths(
        cls,
        paths: str | Path | Sequence[str | Path],
        window_size: int,
        stride: int | None = None,
        key: str | None = None,
        labels_key: str = "labels/class_mask",
        **kwargs,
    ) -> "WindowDataset":
        if isinstance(paths, (str, Path)):
            paths = [paths]
        arrays, labels, metadata = [], [], []
        for path in paths:
            path = Path(path)
            array, label = _load_source(path, key=key)
            arrays.append(array)
            labels.append(label)
            if path.suffix in (".h5", ".hdf5"):
                metadata.append(read_h5_metadata(path))
            else:
                metadata.append({})
        return cls(
            arrays,
            window_size=window_size,
            stride=stride,
            labels=labels,
            metadata=metadata,
            **kwargs,
        )

    def set_channel_stats(
        self,
        channel_mean: np.ndarray | torch.Tensor,
        channel_std: np.ndarray | torch.Tensor,
    ) -> None:
        mean = torch.as_tensor(channel_mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(channel_std, dtype=torch.float32).flatten()
        if mean.numel() != self.num_channels or std.numel() != self.num_channels:
            raise ValueError("channel stats must have one value per channel")
        self.channel_mean = mean
        self.channel_std = std.clamp_min(1e-8)

    @property
    def num_channels(self) -> int:
        return int(self.arrays[0].shape[0])

    def split(self, val_fraction: float = 0.1, gap: int = 0) -> tuple["WindowDataset", "WindowDataset"]:
        """Contiguous time split (no shuffling) with an optional window gap."""
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        num_val = max(1, int(round(len(self.index) * val_fraction)))
        gap = max(0, int(gap))
        train_end = max(0, len(self.index) - num_val - gap)
        val_start = len(self.index) - num_val

        train = copy.copy(self)
        val = copy.copy(self)
        train.index = self.index[:train_end]
        val.index = self.index[val_start:]
        if not train.index or not val.index:
            raise ValueError("split produced an empty dataset; use more data")
        return train, val

    def filter_background(
        self,
        max_event_fraction: float = 0.0,
        background_class: int = 0,
    ) -> "WindowDataset":
        """Keep only windows whose label mask is (almost) pure background.

        ``IGNORE_CLASS`` counts against a window exactly like an event class, so
        windows holding unlabelled transients never reach the calibration set.
        """
        if all(label is None for label in self.labels):
            raise RuntimeError("no label masks available for background filtering")
        if not 0.0 <= max_event_fraction <= 1.0:
            raise ValueError("max_event_fraction must be in [0, 1]")

        kept: list[tuple[int, int]] = []
        for file_index, start in self.index:
            label = self.labels[file_index]
            if label is None:
                kept.append((file_index, start))
                continue
            window = label[:, start : start + self.window_size]
            event_fraction = float((window != background_class).mean())
            if event_fraction <= max_event_fraction:
                kept.append((file_index, start))
        if not kept:
            raise ValueError("background filtering removed every window")

        filtered = copy.copy(self)
        filtered.index = kept
        return filtered

    def window_label_fractions(self, item: int) -> tuple[float, float]:
        """``(event, ignore)`` pixel fractions of window ``item``.

        Kept apart because they mean opposite things to a scorer: event pixels
        make a window a positive, ignore pixels make it unusable as either.
        """
        file_index, start = self.index[item]
        label = self.labels[file_index]
        if label is None:
            return 0.0, 0.0
        window = label[:, start : start + self.window_size]
        ignore = window == IGNORE_CLASS
        event = (window != 0) & ~ignore
        return float(event.mean()), float(ignore.mean())

    def window_event_fraction(self, item: int) -> float:
        """Fraction of labelled-event pixels in window ``item`` (ignore excluded)."""
        return self.window_label_fractions(item)[0]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        file_index, start = self.index[item]
        window = self.arrays[file_index][:, start : start + self.window_size]
        x = torch.as_tensor(np.ascontiguousarray(window), dtype=torch.float32)
        if self.normalize and self.channel_mean is not None:
            x = (x - self.channel_mean[:, None]) / self.channel_std[:, None]
        if not self.return_labels:
            return x
        label = self.labels[file_index]
        if label is None:
            y = torch.full(x.shape, IGNORE_CLASS, dtype=torch.uint8)
        else:
            y = torch.as_tensor(
                np.ascontiguousarray(label[:, start : start + self.window_size])
            )
        return x, y.long()


def estimate_channel_stats(
    dataset: WindowDataset,
    max_windows: int = 512,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std estimated from a random subset of raw windows."""
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    count = min(max_windows, len(dataset))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), size=count, replace=False)

    total = np.zeros(dataset.num_channels, dtype=np.float64)  # raw windows, not __getitem__
    total_sq = np.zeros(dataset.num_channels, dtype=np.float64)
    samples = 0
    for pick in picks:
        file_index, start = dataset.index[int(pick)]
        window = np.asarray(
            dataset.arrays[file_index][:, start : start + dataset.window_size],
            dtype=np.float64,
        )
        total += window.sum(axis=1)
        total_sq += (window**2).sum(axis=1)
        samples += window.shape[1]

    mean = total / samples
    variance = np.maximum(total_sq / samples - mean**2, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _smooth(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = kernel / kernel.sum()
    return np.convolve(signal, kernel, mode="same")


def make_synthetic_das(
    num_channels: int = 32,
    num_samples: int = 30000,
    sample_rate: float = 1000.0,
    seed: int = 0,
    noise: float = 0.15,
    correlated_fraction: float = 0.35,
    tone_fraction: float = 0.15,
) -> np.ndarray:
    """Synthetic ambient DAS recording: ``(num_channels, num_samples)``."""
    rng = np.random.default_rng(seed)
    time = np.arange(num_samples) / sample_rate
    smooth_kernel = np.hanning(65)
    drift_kernel = np.hanning(1025)

    common = _smooth(rng.standard_normal(num_samples), smooth_kernel)
    common /= common.std() + 1e-12

    data = np.empty((num_channels, num_samples), dtype=np.float32)
    for channel in range(num_channels):
        ambient = _smooth(rng.standard_normal(num_samples), smooth_kernel)
        ambient /= ambient.std() + 1e-12

        drift = _smooth(rng.standard_normal(num_samples), drift_kernel)
        drift /= drift.std() + 1e-12

        phase = rng.uniform(0.0, 2.0 * np.pi)
        tone = np.sin(2.0 * np.pi * 47.0 * time + phase)

        signal = (
            ambient
            + correlated_fraction * common
            + 0.3 * drift
            + tone_fraction * tone
        )
        data[channel] = signal / (signal.std() + 1e-12) * noise

    return data


def inject_anomalies(
    data: np.ndarray,
    num_events: int = 4,
    seed: int = 1,
    duration: int = 64,
    min_channels: int = 2,
    max_channels: int = 8,
    amplitude: float = 12.0,
) -> np.ndarray:
    """Add localized high-energy bursts (in place) and return the array."""
    rng = np.random.default_rng(seed)
    num_channels, num_samples = data.shape
    if duration >= num_samples:
        raise ValueError("duration must be smaller than num_samples")
    max_channels = min(max_channels, num_channels)
    window = np.hanning(duration)

    for _ in range(num_events):
        start_channel = int(rng.integers(0, num_channels))
        width = int(rng.integers(min_channels, max_channels + 1))
        end_channel = min(num_channels, start_channel + width)
        start_time = int(rng.integers(0, num_samples - duration + 1))
        burst = rng.standard_normal((end_channel - start_channel, duration))
        data[start_channel:end_channel, start_time : start_time + duration] += (
            amplitude * burst * window
        )
    return data
