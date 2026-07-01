"""Three-branch event-token supervision plus image-guided reliability loss."""

from __future__ import annotations

import torch

import finetune_event as fe
from mul_loss_fine.image_guided_event_reliability_loss import (
    make_configured_image_guided_event_reliability_loss,
)


BRANCH_FIELDS = {
    "geometry": ("pred_event_geometry_token", "event_geometry_voxel"),
    "material": ("pred_event_material_token", "event_material_voxel"),
    "noise": ("pred_event_noise_token", "event_noise_voxel"),
}


def make_additive_token_loss(cfg):
    reliability_loss = make_configured_image_guided_event_reliability_loss(cfg)

    class AdditiveTokenReliabilityLoss(reliability_loss):
        def _normalize_token(self, token: torch.Tensor) -> torch.Tensor:
            cmax = float(getattr(cfg.model, "event_count_cmax", 3.0))
            return torch.log1p(token.clamp_min(0.0).clamp_max(cmax)) / torch.log1p(
                token.new_tensor(cmax)
            )

        @staticmethod
        def _stack_prediction(model_output, key: str) -> torch.Tensor:
            return torch.stack([result[key] for result in model_output.ress], dim=1)

        def forward(self, model_output, views):
            # Base detail/reliability losses must see the predicted geometry
            # token, not the unfiltered full event input.
            geometry_views = model_output.views if model_output.views is not None else views
            total, details, aux = super().forward(model_output, geometry_views)
            # Detached aliases share storage with model outputs and let the
            # visualizer inspect every predicted temporal/polarity bin.
            if model_output.ress and all(
                all(prediction_key in result for result in model_output.ress)
                for prediction_key, _ in BRANCH_FIELDS.values()
            ):
                for name, (prediction_key, _) in BRANCH_FIELDS.items():
                    aux[f"pred_event_{name}_token"] = self._stack_prediction(
                        model_output, prediction_key
                    ).detach()
            if "event_geometry_voxel" not in views[0]:
                details.update(
                    {
                        "branch_token_loss": 0.0,
                        "branch_geometry_token_loss": 0.0,
                        "branch_material_token_loss": 0.0,
                        "branch_noise_token_loss": 0.0,
                        "branch_additive_consistency_loss": 0.0,
                    }
                )
                return total, details, aux

            branch_losses = {}
            predictions = []
            targets = []
            for name, (prediction_key, target_key) in BRANCH_FIELDS.items():
                prediction = self._stack_prediction(model_output, prediction_key)
                target = fe.stack_view_field(views, target_key).to(
                    device=prediction.device, dtype=prediction.dtype
                )
                predictions.append(prediction)
                targets.append(target)

            full_target = torch.stack(targets, dim=0).sum(dim=0)
            presence = (full_target.detach() > 0).to(dtype=predictions[0].dtype)
            weight = 0.02 + 0.98 * presence
            for name, prediction, target in zip(BRANCH_FIELDS, predictions, targets):
                error = (self._normalize_token(prediction) - self._normalize_token(target)).abs()
                branch_losses[name] = (error * weight).sum() / weight.sum().clamp_min(1.0)

            predicted_sum = torch.stack(predictions, dim=0).sum(dim=0)
            consistency = (
                self._normalize_token(predicted_sum) - self._normalize_token(full_target)
            ).abs().mean()
            weighted_branch = (
                float(getattr(cfg.loss, "branch_geometry_weight", 1.0)) * branch_losses["geometry"]
                + float(getattr(cfg.loss, "branch_material_weight", 0.75)) * branch_losses["material"]
                + float(getattr(cfg.loss, "branch_noise_weight", 0.50)) * branch_losses["noise"]
                + float(getattr(cfg.loss, "branch_consistency_weight", 0.10)) * consistency
            )
            token_loss = float(getattr(cfg.loss, "branch_token_weight", 0.50)) * weighted_branch
            total = total + token_loss
            previous_extra = float(details.get("extra_loss_total", 0.0))
            details.update(
                {
                    "branch_token_loss": float(token_loss.detach()),
                    "branch_geometry_token_loss": float(branch_losses["geometry"].detach()),
                    "branch_material_token_loss": float(branch_losses["material"].detach()),
                    "branch_noise_token_loss": float(branch_losses["noise"].detach()),
                    "branch_additive_consistency_loss": float(consistency.detach()),
                    "extra_loss_total": previous_extra + float(token_loss.detach()),
                    "total_loss_with_extra": float(total.detach()),
                }
            )
            return total, details, aux

    return AdditiveTokenReliabilityLoss
