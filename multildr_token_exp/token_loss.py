"""Asymmetric same-window token consistency for paired LDR observations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import finetune_event as fe
from mul_loss_fine.finetune_mul_ldr_event import _batch_group_keys


def _exposure_quality(views, *, device, dtype):
    images = fe.stack_view_field(views, "img").to(device=device, dtype=dtype)
    if float(images.detach().amin()) < -0.05:
        images = (images + 1.0) * 0.5
    images = images.clamp(0.0, 1.0)
    saturated = ((images < 0.03) | (images > 0.97)).to(dtype).mean(dim=(1, 2, 3, 4))
    contrast = images.std(dim=(1, 2, 3, 4))
    return (1.0 - saturated) + 0.05 * contrast


class PairedTokenConsistencyMixin:
    def _init_token_consistency(self, *, weight: float) -> None:
        self.exposure_token_weight = float(weight)

    def forward(self, model_output, views):
        total_loss, details, aux = super().forward(model_output, views)
        tokens = None
        if model_output.ress and all("exposure_token" in result for result in model_output.ress):
            tokens = torch.stack(
                [result["exposure_token"] for result in model_output.ress], dim=1
            )
        if tokens is None or self.exposure_token_weight <= 0.0:
            details.update(
                {
                    "ldr_token_loss": 0.0,
                    "ldr_token_cosine": 0.0,
                    "ldr_token_pair_count": 0.0,
                }
            )
            return total_loss, details, aux

        batch = tokens.shape[0]
        quality = _exposure_quality(views, device=tokens.device, dtype=tokens.dtype)
        groups = {}
        for batch_index, key in enumerate(_batch_group_keys(views, batch)):
            groups.setdefault(key, []).append(batch_index)

        losses = []
        cosines = []
        for indices in groups.values():
            if len(indices) < 2:
                continue
            anchor = max(indices, key=lambda index: float(quality[index].detach()))
            anchor_raw = tokens[anchor].detach().float()
            anchor_tokens = F.layer_norm(anchor_raw, (anchor_raw.shape[-1],))
            anchor_direction = F.normalize(anchor_raw, dim=-1, eps=1.0e-6)
            for other in indices:
                if other == anchor:
                    continue
                student_raw = tokens[other].float()
                student = F.layer_norm(student_raw, (student_raw.shape[-1],))
                student_direction = F.normalize(student_raw, dim=-1, eps=1.0e-6)
                cosine = (student_direction * anchor_direction).sum(dim=-1)
                losses.append(F.smooth_l1_loss(student, anchor_tokens))
                cosines.append(cosine.mean())

        zero = tokens.new_tensor(0.0)
        token_loss = torch.stack(losses).mean() if losses else zero
        token_cosine = torch.stack(cosines).mean() if cosines else zero
        extra = self.exposure_token_weight * token_loss
        total_loss = total_loss + extra
        details.update(
            {
                "ldr_token_loss": float(token_loss.detach()),
                "ldr_token_cosine": float(token_cosine.detach()),
                "ldr_token_pair_count": float(len(losses)),
                "extra_loss_total": float(details.get("extra_loss_total", 0.0))
                + float(extra.detach()),
                "total_loss_with_extra": float(total_loss.detach()),
            }
        )
        aux["exposure_tokens"] = tokens.detach()
        return total_loss, details, aux


def wrap_token_consistency(base_loss, cfg):
    class ConfiguredTokenConsistencyLoss(PairedTokenConsistencyMixin, base_loss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._init_token_consistency(weight=float(cfg.loss.ldr_token_weight))

    return ConfiguredTokenConsistencyLoss


__all__ = ["wrap_token_consistency"]
