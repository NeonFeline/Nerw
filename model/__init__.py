"""Optical-cable vibration modeling: LeJEPA-style transformer + anomaly detection."""

from model.anomaly import AnomalyConfig, AnomalyDetector
from model.backbone import ModelConfig, ModelOutput, VibrationTransformer
from model.data import (
    H5Transposed,
    WindowDataset,
    discover_recordings,
    estimate_channel_stats,
    inject_anomalies,
    load_h5_labels,
    make_synthetic_das,
    read_h5_metadata,
)
from model.losses import LeJEPALoss, LossOutput
from model.masking import MaskConfig, expand_token_mask, generate_span_mask
from model.sigreg import SIGReg

__all__ = [
    "AnomalyConfig",
    "AnomalyDetector",
    "H5Transposed",
    "LeJEPALoss",
    "LossOutput",
    "MaskConfig",
    "ModelConfig",
    "ModelOutput",
    "SIGReg",
    "VibrationTransformer",
    "WindowDataset",
    "discover_recordings",
    "estimate_channel_stats",
    "expand_token_mask",
    "generate_span_mask",
    "inject_anomalies",
    "load_h5_labels",
    "make_synthetic_das",
    "read_h5_metadata",
]
