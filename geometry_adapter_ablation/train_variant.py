from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from paired_token_reliability import train_unified_geometry_contribution as trainer
from paired_token_reliability.unified_loss import UnifiedGeometryContributionLoss

from .variants import make_variant_model


class DiagnosticLoss(UnifiedGeometryContributionLoss):
    """Add diagnostics without changing any optimized objective."""

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        output = args[0]
        final_depth = torch.stack([item["depth"] for item in output.ress], dim=1).squeeze(-1)
        coarse_depth = torch.stack(
            [item["depth_coarse"] for item in output.ress], dim=1
        ).squeeze(-1)
        zero_event_difference = (final_depth.float() - coarse_depth.float()).abs().mean()
        update_norm = torch.stack(
            [item["adapter_depth_update_magnitudes"].float().mean() for item in output.ress]
        ).mean()
        result.details["update_norm"] = update_norm
        # A zero event produces the exact RGB-only path, so this is the output
        # difference to the zero-event counterfactual without a second forward.
        result.details["zero_event_difference"] = zero_event_difference
        return result


def _argument_value(argv, name, default=None):
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def _write_tensorboard(output: Path) -> None:
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"TensorBoard writer unavailable: {exc}", flush=True)
        return
    history = json.loads(metrics_path.read_text(encoding="utf-8"))
    writer = SummaryWriter(str(output / "tensorboard"))
    for record in history:
        step = int(record.get("global_epoch", 0))
        phase = record.get("phase", "unknown")
        for split in ("train", "validation"):
            for key, value in record.get(split, {}).items():
                writer.add_scalar(f"{phase}/{split}/{key}", float(value), step)
    writer.close()


def main(config_path: str) -> None:
    config_file = Path(config_path).resolve()
    ablation = OmegaConf.load(config_file)
    variant = str(ablation.variant)
    base_config = (config_file.parent / str(ablation.base_config)).resolve()

    model_type = make_variant_model(variant)
    trainer.UnifiedGeometryContributionModel = model_type
    trainer.UnifiedGeometryContributionLoss = DiagnosticLoss

    argv = list(sys.argv[1:])
    if _argument_value(argv, "--config") is None:
        argv = ["--config", str(base_config), *argv]
    print(f"GeometryAdapter ablation={variant} config={config_file}", flush=True)
    trainer.main(argv)

    if int(os.environ.get("RANK", "0")) == 0:
        output = Path(_argument_value(argv, "--output", "exp/geometry_adapter_ablation"))
        _write_tensorboard(output)
        metrics = output / "metrics.json"
        if metrics.is_file():
            last = json.loads(metrics.read_text(encoding="utf-8"))[-1]
            values = last.get("validation", {})
            print(
                "diagnostics "
                f"Cmean={values.get('contribution_mean', float('nan')):.6f} "
                f"Cstd={values.get('contribution_std', float('nan')):.6f} "
                f"update_norm={values.get('update_norm', float('nan')):.6f} "
                f"zero_event_difference={values.get('zero_event_difference', float('nan')):.6f}",
                flush=True,
            )


__all__ = ["main"]
