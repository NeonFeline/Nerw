"""Guards for the classification stage.

The point of these is that the head is wired to the right axes, that ignore
pixels really are excluded from every number, and that the metrics say what
they claim to say.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model.backbone import ModelConfig, VibrationTransformer
from model.data import IGNORE_CLASS, WindowDataset
from model.segmenter import (
    ConfusionMatrix,
    DASSegmenter,
    SegmenterConfig,
    class_frequencies,
    inverse_frequency_weights,
    load_segmenter,
    save_segmenter,
    segmentation_loss,
)

CHANNELS, WINDOW, PATCH, CLASSES = 8, 64, 16, 5


def make_model():
    backbone = VibrationTransformer(ModelConfig(
        num_channels=CHANNELS, window_size=WINDOW, patch_size=PATCH,
        d_model=32, n_heads=4, n_layers=2, dim_feedforward=64, proj_dim=16,
        proj_hidden=16,
    ))
    return DASSegmenter(backbone, SegmenterConfig(num_classes=CLASSES, hidden=32))


def test_logits_cover_the_full_channel_time_grid():
    model = make_model()
    logits = model(torch.randn(3, CHANNELS, WINDOW))
    assert logits.shape == (3, CLASSES, CHANNELS, WINDOW)


def test_predictions_are_constant_within_a_time_patch():
    """The backbone tokenizes time, so a patch is the finest time resolution."""
    model = make_model().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, CHANNELS, WINDOW))
    for start in range(0, WINDOW, PATCH):
        block = logits[0, :, :, start : start + PATCH]
        assert torch.allclose(block, block[:, :, :1].expand_as(block))


def test_ignore_pixels_do_not_contribute_to_the_loss():
    model = make_model()
    x = torch.randn(2, CHANNELS, WINDOW)
    logits = model(x)
    target = torch.zeros(2, CHANNELS, WINDOW, dtype=torch.long)
    reference = segmentation_loss(logits, target)

    masked = target.clone()
    masked[:, CHANNELS // 2 :, :] = IGNORE_CLASS
    half = segmentation_loss(logits, masked)
    # the ignored half is gone, not averaged in as zero
    assert not torch.isclose(reference, half)
    kept = segmentation_loss(logits[:, :, : CHANNELS // 2], target[:, : CHANNELS // 2])
    assert torch.allclose(half, kept, atol=1e-6)


def test_a_fully_ignored_target_yields_no_gradient():
    model = make_model()
    logits = model(torch.randn(1, CHANNELS, WINDOW))
    target = torch.full((1, CHANNELS, WINDOW), IGNORE_CLASS, dtype=torch.long)
    loss = segmentation_loss(logits, target)
    assert torch.isnan(loss) or float(loss) == 0.0


def test_confusion_matrix_ignores_unknown_cells():
    confusion = ConfusionMatrix(3)
    predicted = torch.tensor([0, 1, 2, 1])
    target = torch.tensor([0, 1, IGNORE_CLASS, 2])
    confusion.update(predicted, target)
    assert confusion.matrix.sum() == 3
    assert confusion.matrix[0, 0] == 1 and confusion.matrix[1, 1] == 1
    assert confusion.matrix[2, 1] == 1


def test_metrics_match_a_hand_worked_example():
    confusion = ConfusionMatrix(2)
    confusion.matrix = np.array([[8, 2], [1, 9]], dtype=np.int64)
    m = confusion.metrics()
    assert m["recall"][1] == pytest.approx(9 / 10)
    assert m["precision"][1] == pytest.approx(9 / 11)
    assert m["iou"][1] == pytest.approx(9 / 12)
    assert m["accuracy"] == pytest.approx(17 / 20)


def test_class_weights_lift_the_rare_classes():
    counts = np.array([1_000_000, 1000, 100])
    weights = inverse_frequency_weights(counts).numpy()
    assert weights[2] > weights[1] > weights[0]
    assert weights.mean() == pytest.approx(1.0, abs=1e-5)


def test_class_frequencies_skip_ignore():
    mask = np.array([[0, 1, IGNORE_CLASS, 1]], dtype=np.uint8)
    counts = class_frequencies([mask], 3)
    assert counts.tolist() == [1, 2, 0]


def test_the_head_can_learn_a_trivial_pattern():
    """Sanity: loud channels are class 1, quiet ones class 0."""
    torch.manual_seed(0)
    model = make_model()
    x = torch.randn(16, CHANNELS, WINDOW) * 0.1
    target = torch.zeros(16, CHANNELS, WINDOW, dtype=torch.long)
    x[:, :3] *= 25.0
    target[:, :3] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(120):
        optimizer.zero_grad()
        loss = segmentation_loss(model(x), target)
        loss.backward()
        optimizer.step()
    accuracy = (model(x).argmax(1) == target).float().mean()
    assert float(accuracy) > 0.95


def test_segmenter_round_trips_through_disk(tmp_path):
    model = make_model()
    save_segmenter(tmp_path / "s.pt", model, {"class_ids": {"background": 0}})
    restored, checkpoint = load_segmenter(tmp_path / "s.pt")
    assert checkpoint["class_ids"] == {"background": 0}
    x = torch.randn(1, CHANNELS, WINDOW)
    model.eval(); restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x))


def test_dataset_returns_aligned_labels_and_highpasses():
    rng = np.random.default_rng(0)
    drift = np.linspace(0, 30, 4000)[None, :] * np.ones((CHANNELS, 1))
    array = (drift + 0.01 * rng.standard_normal((CHANNELS, 4000))).astype(np.float32)
    labels = np.zeros((CHANNELS, 4000), dtype=np.uint8)
    labels[:, 2000:] = 3

    plain = WindowDataset([array], window_size=WINDOW, stride=WINDOW, labels=[labels],
                          return_labels=True, normalize=False)
    x, y = plain[0]
    assert x.shape == y.shape == (CHANNELS, WINDOW)
    assert int(y.max()) == 0

    filtered = WindowDataset([array], window_size=WINDOW, stride=WINDOW, labels=[labels],
                             return_labels=True, normalize=False,
                             highpass_hz=2.0, sample_rate=1000.0)
    assert float(filtered[10][0].abs().mean()) < float(plain[10][0].abs().mean()) / 10
    # a window past the boundary carries the event label
    index = next(i for i, (_, s) in enumerate(filtered.index) if s >= 2000)
    assert int(filtered[index][1].max()) == 3
