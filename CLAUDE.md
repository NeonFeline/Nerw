# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Synthetic distributed acoustic sensing (DAS) recordings for a fibre optic cable along a railway, and a self-supervised transformer that scores DAS windows for anomalies.

- `simulate_das.py`: kinematic generator for one recording (miniDAS-style HDF5). It defines the physics, background, labels and output format.
- `make_dataset.py`: draws randomised scenarios over site families and builds `data/synthetic/<version>/`.
- `validate_sim.py`: checks one generated run using only its files (format, `traces == signal + background`, class contrast, gauge spatial signature, apparent event speed).
- `model/`: LeJEPA-style backbone (masked reconstruction + SIGReg), training, anomaly calibration, inference and evaluation. Stage 2 (`segmenter.py`, `finetune.py`, `evaluate_classes.py`) puts a per-pixel classification head on the pretrained backbone.
- `make_showcase.py`: renders one overview figure of a dataset into `data/showcase/`.
- `specfem2d/surface_blast/`: standalone SPECFEM2D elastic simulation. Nothing else imports it (see the end of this file).

## Commands

uv manages the environment. torch comes from the PyTorch cu130 index configured in `pyproject.toml`. `model/requirements.txt` is a leftover and is not what the project installs from.

```bash
uv sync
uv run pytest                                   # tests/ (generator + dataset) and model/tests/ (model smoke)
uv run pytest tests/test_simulate.py::test_moving_and_static_paths_are_the_same_operator
uv run pytest model/tests/test_smoke.py -k sigreg
uv run python model/tests/test_smoke.py         # smoke suite also runs without pytest

# one recording: fixed default scene, or --scenario scene.json; --background mixes in a real recording
uv run python simulate_das.py --out data/synthetic/run01
# full dataset. --workers parallelises independent runs; 8 is the sweet spot on one
# GPU (beyond that the propagation kernels serialise), giving ~2.8 s/run.
uv run python make_dataset.py --out data/synthetic/v4 \
    --train-runs 300 --val-runs 60 --test-runs 120 --max-events 12 --workers 8
uv run python validate_sim.py data/synthetic/v4/train/embankment_soft_000/das.h5
uv run python make_showcase.py --data data/synthetic/v4 --out data/showcase

# split a build across a Slurm job array, then combine the shard manifests
uv run python make_dataset.py --out data/synthetic/v4 --shard-index $SLURM_ARRAY_TASK_ID --shard-count 4
uv run python make_dataset.py --out data/synthetic/v4 --merge

# model entry points: run as modules from the repo root
uv run python -m model.train --data data/synthetic/v2/train --background-only \
    --window-size 1024 --stride 256 --epochs 30 --out runs/das_jepa
uv run python -m model.evaluate --checkpoint runs/das_jepa/checkpoint.pt \
    --anomaly-stats runs/das_jepa/anomaly_stats.pt --input data/synthetic/v2/test --stride 256
uv run python -m model.infer --checkpoint ... --anomaly-stats ... --input <file-or-dir> --out scores.npz
uv run python -m model.train --synthetic --epochs 3 --window-size 256 --d-model 64 --n-layers 2 --out /tmp/run

# stage 2: classify what the activity is, not just that something happened
uv run python -m model.finetune --pretrained runs/das_jepa/checkpoint.pt \
    --train data/synthetic/v4/train --val data/synthetic/v4/val --epochs 40 --out runs/segmenter
uv run python -m model.evaluate_classes --segmenter runs/segmenter/segmenter.pt \
    --input data/synthetic/v4/test
uv run python -m model.finetune --scratch --train ... --out runs/segmenter_scratch  # LeJEPA ablation
```

`data/` is gitignored except `manifest.json` and `data/showcase/*.png`, and the recordings are not in the repo. Rebuild them from the argv recorded in the manifest, which also stores library versions, and each run's seed, resolved config, scenario and sha256.

## Array orientation

This is the most common source of bugs.

- Inside `simulate_das.py`, fields are `(n_channels, nt)`.
- On disk they are time-major, `(nt, n_channels)`: `das.h5` `traces` and `labels/class_mask`, and `components.h5` `signal`/`background`. The exception is `components.h5` `event_snr_db`, which is `(n_events, n_channels, nt_envelope)` and is not transposed.
- `model/data.py` works in `(channels, time)`. `H5Transposed` flips HDF5 traces lazily.

Traces are in rad/s. The `scale_factor` attribute must stay `1.0` so a reader never rescales them. Float arrays are written uncompressed: gzip buys ~6% on noise-like float32 for 23x the write time. The uint8 class mask is the opposite and stays compressed.

## Generator invariants

`tests/test_simulate.py` pins each of these. Every test names the defect it guards against, so read the relevant one before changing behaviour.

- **One propagation operator.** `band_plan`/`band_weights` quantise a dispersive, attenuating surface-wave operator into log-spaced bands that form a partition of unity. Stationary sources apply it in the frequency domain (`_propagate`). Moving sources apply the same band triple at the retarded time (`moving_field` → `_retarded`). The two paths must agree for a source that doesn't move. Otherwise the synthesis method correlates with the class label. Change both paths together. `band_weights` is cached on the rfft grid; it depends only on the band edges, never on the medium.
- **Attenuation is physical.** With `damping_ratio` set (the normal case), `alpha(f) = pi f D / c(f)` -- hysteretic soil damping, linear in frequency. A hand-set coefficient is the fallback. This is what confines an event to its own stretch of fibre: at D = 0.03 a 50 Hz source is 20 dB down by ~66 m. An earlier constant 2e-4/m was ~80x too weak and smeared every event over the whole array.
- **Multiple propagation paths.** An event's `paths` list gives `{weight, medium}` entries, each a cfg override rendered through the same operator and summed. `track_tamper` uses it for the guided path along the rail: kilometres per second, `spreading_power` near 0, so it arrives on a near-vertical moveout far beyond the ground arrival. `spreading_power` is the exponent on `r` (0.5 is a surface wave on a half-space).
- **Gauge length.** Signals are evaluated at taps `x ± L/2` and differenced (`gauge_taps`, `_gauge_combine`). Instrument noise goes through the same operator on the channel grid (`_gauge_spatial`), so noise and signal share a spatial transfer function.
- **The background must leave room for the events.** Instrument noise is white above `noise_knee_hz` and rises below it (`_knee_noise_multi`), not a 1/f ramp to DC. Shaped as a ramp, 91% of background variance sat below 1 Hz and an unfiltered window-RMS detector scored exactly chance on the same data a 15-150 Hz one scored 0.78 on. A test caps the sub-1 Hz share.
- **Labels come from SNR, not geometry.** `signal_envelope` is the single measurement recipe used by `build_labels`, by `_calibrate` (which scales each event to its `snr_db` against the realised background) and by `validate_sim.py`. A cell gets its event class when it is ≥ `label_hi_db` above background. The following cells become `255`, which must be excluded from both training and scoring:
  - cells between `label_lo_db` and `label_hi_db`
  - ambiguous overlaps (two events within `label_margin_db`)
  - cells touched by unlabelled background bursts or glitches

  The ignore value is defined separately in three places, so keep them in sync: `IGNORE_ID` in `simulate_das.py`, `IGNORE_CLASS` in `model/data.py`, and a local `IGNORE_ID` in `validate_sim.py`.
- **Site effects.** `coupling` scales events and the ground-borne background together. `noise_gain` scales only fibre phase noise. This split keeps dead channels from reading as perfect-SNR events.
- **RNG streams.** `synthesize` spawns independent site, background and transient streams from `cfg["seed"]`, and each event gets its own generator from `_event_rng`. As a result, the background is bit-identical with or without events. Don't reorder draws across streams. Derive seeds with `SeedSequence`, never `hash()` of a string (it is salted per process).
- **Exact decomposition.** The `noise_std` rescale is applied to signal and background before summing, so `traces == signal + background` holds exactly.
- **`synthesize` mutates `cfg`.** It adds `envelope_decimation`, `realized_scale`, `background_info` and `events_realized`, and the whole cfg is serialised into `meta_json`/`scenario.json`. Build a fresh cfg per call.
- Heavy array work runs in torch on a module-global device (`set_device`, `--device`; CUDA when available). `signal_envelope` and the coda's channel blur are torch, not scipy -- they were 10 s of a 25 s run.
- Scenario events have a `kind` that selects the rendering path and a separate `class` that maps into `CLASS_IDS`. Moving kinds: `train`, `wheel_flat`, `road_vehicle`, `walker`, `vehicle_stop`, `fence_cut`. Stationary: `static`, `burst`, `cable_cut`, `track_tamper`. Where they differ, `walker` → `footsteps` and `static` → `digging`.

## Classes

`CLASS_IDS` runs 0-10 with `IGNORE_ID = 255`. Ordinary traffic and nuisance: `train`, `road_vehicle`, `footsteps`, `burst`, `wheel_flat`. `SECURITY_CLASSES` -- what a trackside system is asked to flag -- are `digging`, `cable_cut` (grinder tonal plus grit, gated into cutting bursts), `track_tamper` (hammer blows on the rail, ground plus rail path), `fence_cut` (sparse bolt-cropper snips creeping along a fence line) and `vehicle_stop` (brakes to a halt, then idles).

These are **activity** classes. DAS resolves what is being done and where; it does not resolve intent or identity, and no amount of training changes that. A head that separates digging from walking from a grinder is doing real work. One advertised as separating a threat from a passer-by is not.

`SNR_RANGES` in `make_dataset.py` gives each class its own peak-envelope SNR band. A train on trackside fibre is unmissable and the job is to classify it, not find it (26-48 dB); a person walking or croppers on a fence sit just above the noise (3-16 dB). `--mix-temperature` flattens the per-family event mix so no class is left with a handful of examples to memorise; it is a training mix, not a claim about base rates.

## Dataset split

The split is by **site family**, not by seed. `SITE_FAMILIES` holds each family's medium, geometry, background and event-mix ranges. `TRAIN_FAMILIES` feed `train/` and `val/`, and `TEST_FAMILIES` appear only in `test/`. A new family must go into exactly one of the two tuples (a test enforces this). Each run writes `<out>/<split>/<family>_<NNN>/` containing `das.h5`, `components.h5` (clean signal, background, coupling, per-event SNR maps), `labels.csv`, `scenario.json` and `qa.png`.

`discover_recordings` recurses and skips HDF5 files without a `traces` dataset, such as `components.h5`. Point `--data` at `train/`, never at the dataset root, or the held-out test families leak into training. `model.train` does not read `val/`. It takes a contiguous validation split from the tail of the window index, with a gap.

## Model pipeline

- **Tokens.** Each token is one time patch spanning *all* channels (`patch_dim = num_channels * patch_size`). A checkpoint is therefore tied to one channel count and window size, and every recording in a `WindowDataset` must have the same channel count.
- **Objective** (`losses.py`): `(1 - λ)·MSE` on masked tokens + `λ·SIGReg` on the projection of the mean-pooled encoder output. Masks are contiguous time spans (`masking.py`).
- **`model.train` outputs.** It computes per-channel mean/std on train windows and saves them in `checkpoint.pt`. `infer` and `evaluate` reuse them. It keeps the best checkpoint by val MSE, then fits `AnomalyDetector` on background-only train windows and writes `anomaly_stats.pt`.
- **`AnomalyDetector`** combines two z-scored signals:
  - a masked-pixel reconstruction residual, averaged over `num_eval_masks` deterministic masks
  - the Mahalanobis distance of the projection under a shrunk Gaussian

  The threshold is a quantile of the combined score on the calibration set.
- **Ignore pixels.** `filter_background` counts `255` against a window just like an event pixel. `model.evaluate` counts a window as positive if it has any event pixel, and drops event-free windows that contain ignore pixels (`--max-ignore-fraction`). It also prints a band-passed RMS energy baseline AUC to compare against. `segmentation_loss` passes `255` as `ignore_index`, and `ConfusionMatrix` drops those pixels from every count.
- **High-pass.** Both stages take `--highpass-hz` (default 2 Hz) and `WindowDataset` applies it once at load. The value is recorded in the checkpoint so the two stages cannot silently disagree about what the model was fed. Compare against the `baseline_rms` AUC before believing any model number: on raw windows the model reproduced a raw-energy detector to three decimals.
- **Stage 2 (`model/segmenter.py`, `model/finetune.py`).** The head maps each token to a per-channel class distribution for its patch, mirroring the reconstruction head, then expands back to full time resolution -- so predictions are constant within a time patch. `--freeze-epochs` trains the head alone first so its early gradients cannot wreck the pretrained features, then unfreezes the backbone at `--backbone-lr-scale` of the head's learning rate. Background outnumbers the rare classes by four or five orders of magnitude, so `inverse_frequency_weights` is on by default. `--scratch` is the ablation that says whether the LeJEPA stage bought anything.
- `model/tests/test_smoke.py` uses `model.data.make_synthetic_das`, a simple ambient-noise generator unrelated to `simulate_das.py`.

## specfem2d/surface_blast

This is a SPECFEM2D run of an explosive moment-tensor source (50 Hz Ricker) at the free surface of a 50 m deep mesh, recorded by a horizontal fibre 1 m below the surface. `SPECFEM2D_BIN=/path/to/bin PYTHON="uv run python" ./run.sh [das_strain.py options]` does three things:
1. writes `DATA/STATIONS` (`make_stations.py`)
2. meshes and solves
3. converts the UX seismograms into a gauge-length strain gather plus optical phase (`das_strain.py` → `OUTPUT_FILES/das_strain.npz` and a PNG)

The gauge difference is exact only when `L/2` is a multiple of the channel spacing. `OUTPUT_FILES/` is gitignored.
