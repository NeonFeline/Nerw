"""Smoke tests for the DAS vibration JEPA model.

Run standalone (``python tests/test_smoke.py``) or with pytest.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.anomaly import AnomalyConfig, AnomalyDetector
from model.backbone import ModelConfig, VibrationTransformer
from model.data import (
    WindowDataset,
    estimate_channel_stats,
    inject_anomalies,
    make_synthetic_das,
    read_h5_metadata,
)
from model.losses import LeJEPALoss
from model.masking import MaskConfig, generate_span_mask
from model.sigreg import SIGReg


def tiny_model(num_channels: int = 8, window_size: int = 256, patch_size: int = 16) -> ModelConfig:
    return ModelConfig(
        num_channels=num_channels,
        window_size=window_size,
        patch_size=patch_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        n_decoder_layers=1,
        dim_feedforward=128,
        dropout=0.0,
        proj_dim=32,
        proj_hidden=64,
    )


def test_sigreg_prefers_isotropic_gaussian() -> None:
    torch.manual_seed(0)
    sigreg = SIGReg(num_slices=256)
    gaussian = torch.randn(512, 32)
    uniform = (torch.rand(512, 32) - 0.5) * 4.0
    collapsed = 0.1 * torch.randn(512, 32) + 0.05 * torch.ones(512, 32)
    shifted = torch.randn(512, 32) + 5.0
    scaled = torch.randn(512, 32) * 3.0

    gaussian_loss = float(sigreg(gaussian))
    for name, sample in [
        ("uniform", uniform),
        ("collapsed", collapsed),
        ("shifted", shifted),
        ("scaled", scaled),
    ]:
        assert gaussian_loss < float(sigreg(sample)), name


def test_sigreg_gradients() -> None:
    torch.manual_seed(1)
    sigreg = SIGReg(num_slices=64)
    embeddings = torch.randn(64, 16, requires_grad=True)
    loss = sigreg(embeddings)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_span_mask_properties() -> None:
    generator = torch.Generator().manual_seed(0)
    mask = generate_span_mask(
        16, 64, MaskConfig(mask_ratio=0.5), generator=generator
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (16, 64)
    ratios = mask.float().mean(dim=1)
    assert (ratios > 0).all() and (ratios < 0.95).all()

    random_mask = generate_span_mask(
        8,
        64,
        MaskConfig(num_spans=3, min_span=2, max_span=6),
        generator=generator,
    )
    assert random_mask.any(dim=1).all()
    assert (~random_mask.all(dim=1)).all()


def test_forward_backward_and_patch_roundtrip() -> None:
    torch.manual_seed(2)
    config = tiny_model()
    model = VibrationTransformer(config)
    x = torch.randn(4, config.num_channels, config.window_size)
    mask = generate_span_mask(
        4,
        model.num_patches,
        MaskConfig(num_spans=2, min_span=2, max_span=4),
        generator=torch.Generator().manual_seed(0),
    )

    output = model(x, mask)
    assert output.pred.shape == (4, model.num_patches, config.num_channels * config.patch_size)
    assert output.proj.shape == (4, config.proj_dim)
    assert torch.allclose(model.unpatchify(model.patchify(x)), x)

    error, pixel_mask, _ = model.per_pixel_error(x, mask)
    assert error.shape == x.shape
    assert int(pixel_mask.sum()) == int(mask.sum()) * config.patch_size * config.num_channels

    loss_fn = LeJEPALoss(sigreg_weight=0.1, num_slices=64)
    losses = loss_fn(output, mask)
    losses.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_overfit_single_batch() -> None:
    torch.manual_seed(3)
    config = tiny_model()
    model = VibrationTransformer(config)
    loss_fn = LeJEPALoss(sigreg_weight=0.05, num_slices=32)
    optimizer = AdamW(model.parameters(), lr=3e-3)

    array = make_synthetic_das(
        num_channels=config.num_channels, num_samples=1200, seed=0, noise=0.05
    )
    dataset = WindowDataset([array], config.window_size, stride=128, normalize=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    x = next(iter(loader))
    mask = generate_span_mask(
        8,
        model.num_patches,
        MaskConfig(mask_ratio=0.4),
        generator=torch.Generator().manual_seed(0),
    )

    first_mse = final_mse = None
    for _ in range(80):
        output = model(x, mask)
        losses = loss_fn(output, mask)
        optimizer.zero_grad()
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if first_mse is None:
            first_mse = float(losses.mse.detach())
        final_mse = float(losses.mse.detach())

    assert final_mse < 0.8 * first_mse, (first_mse, final_mse)
    assert torch.isfinite(losses.sigreg)


def test_anomaly_detector_end_to_end() -> None:
    torch.manual_seed(4)
    config = tiny_model()
    model = VibrationTransformer(config)
    loss_fn = LeJEPALoss(sigreg_weight=0.05, num_slices=32)
    optimizer = AdamW(model.parameters(), lr=3e-3)
    mask_config = MaskConfig(num_spans=2, min_span=2, max_span=5)

    normal = make_synthetic_das(
        num_channels=config.num_channels, num_samples=12000, seed=1, noise=0.1
    )
    normal_dataset = WindowDataset([normal], config.window_size, stride=128, normalize=False)
    train_loader = DataLoader(normal_dataset, batch_size=16, shuffle=True)

    model.train()
    for _ in range(5):
        for batch in train_loader:
            mask = generate_span_mask(
                batch.shape[0],
                model.num_patches,
                mask_config,
                generator=torch.Generator().manual_seed(0),
            )
            output = model(batch, mask)
            losses = loss_fn(output, mask)
            optimizer.zero_grad()
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    calibration_loader = DataLoader(normal_dataset, batch_size=16, shuffle=False)
    detector = AnomalyDetector(
        model,
        AnomalyConfig(num_eval_masks=2, threshold_quantile=0.99),
        mask_config,
    )
    detector.fit(calibration_loader, max_windows=128)

    anomalous = make_synthetic_das(
        num_channels=config.num_channels, num_samples=4000, seed=2, noise=0.1
    )
    inject_anomalies(anomalous, num_events=6, seed=3, duration=64, amplitude=15.0)
    anomalous_dataset = WindowDataset([anomalous], config.window_size, stride=128, normalize=False)
    anomalous_loader = DataLoader(anomalous_dataset, batch_size=16, shuffle=False)

    normal_scores = detector.score_loader(calibration_loader)["score"]
    anomalous_scores = detector.score_loader(anomalous_loader)["score"]
    assert float(anomalous_scores.mean()) > float(normal_scores.mean())
    assert float(anomalous_scores.max()) > float(normal_scores.max())
    detected = float(detector.score_loader(anomalous_loader)["is_anomaly"].float().mean())
    assert detected > 0.1

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "anomaly_stats.pt"
        detector.save(path)
        reloaded = AnomalyDetector.load(path, model)
        reloaded_scores = reloaded.score_loader(anomalous_loader)["score"]
        assert torch.allclose(anomalous_scores, reloaded_scores)


def test_dataset_split_and_stats() -> None:
    array = make_synthetic_das(num_channels=8, num_samples=4000, seed=5)
    dataset = WindowDataset([array], window_size=256, stride=128)
    train, val = dataset.split(0.25, gap=2)
    assert len(train) > 0 and len(val) > 0

    mean, std = estimate_channel_stats(train, max_windows=32)
    assert mean.shape == (8,) and std.shape == (8,)
    assert (std > 0).all()

    train.set_channel_stats(mean, std)
    window = train[0]
    assert window.shape == (8, 256)
    assert torch.isfinite(window).all()


def test_h5_dataset_labels_and_background_filter() -> None:
    import h5py

    rng = np.random.default_rng(0)
    traces = rng.standard_normal((2000, 4)).astype(np.float32)
    labels = np.zeros((2000, 4), dtype=np.uint8)
    labels[1000:1100, 1:3] = 4

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "das.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("traces", data=traces)
            handle.create_dataset("labels/class_mask", data=labels)
            handle.attrs["sampling_rate"] = 1000.0
            handle.attrs["class_ids"] = '{"background": 0, "digging": 4}'

        dataset = WindowDataset.from_paths(path, window_size=256, stride=128)
        assert dataset.num_channels == 4
        window = dataset[0]
        assert window.shape == (4, 256)
        assert torch.allclose(window, torch.from_numpy(traces[:256, :].T.copy()))

        background = dataset.filter_background()
        assert len(background) < len(dataset)
        for file_index, start in background.index:
            assert (labels[start : start + 256, :] == 0).all()

        metadata = read_h5_metadata(path)
        assert float(metadata["sampling_rate"]) == 1000.0
        assert metadata["class_ids"]["digging"] == 4


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
