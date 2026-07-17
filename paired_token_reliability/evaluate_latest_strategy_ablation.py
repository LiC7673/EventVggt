"""Four-scene/all-EV evaluation with variant-correct ablation inference."""
from __future__ import annotations

import argparse
import sys

from paired_token_reliability import evaluate_alternating_detail_first_fixed_four_scenes as evaluator
from paired_token_reliability.evaluate_cur_event_hf_residual_four_scenes import (
    build_model as build_v2_model,
)


VARIANTS = {"noisy_event_only", "multi_ldr_only", "without_refiner_normal"}
_VARIANT = None


def build_model(checkpoint, device, depth_scale):
    model, cfg = build_v2_model(checkpoint, device, depth_scale)
    if _VARIANT == "noisy_event_only":
        # This model never learned reliability or Multi-LDR. Preserve its
        # training-time oracle gates and noisy-event pixel refinement.
        model.set_confidence_stage("geo")
        model.disable_pixel_refiner = False
    elif _VARIANT == "multi_ldr_only":
        # Test the HDR-like base itself; a random/untrained refiner must never
        # enter the reported final output.
        model.set_confidence_stage("geo")
        model.disable_pixel_refiner = True
    elif _VARIANT == "without_refiner_normal":
        model.set_confidence_stage("full")
        model.disable_pixel_refiner = False
    else:
        raise RuntimeError(_VARIANT)
    print(
        f"[ablation eval] variant={_VARIANT} "
        f"confidence_stage={'full' if _VARIANT == 'without_refiner_normal' else 'geo/C=1'} "
        f"pixel_refiner={'OFF' if model.disable_pixel_refiner else 'ON'}",
        flush=True,
    )
    return model, cfg


def extract_variant(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    known, remaining = parser.parse_known_args(argv)
    return known.variant, remaining


def main():
    global _VARIANT
    _VARIANT, remaining = extract_variant(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    evaluator.build_model = build_model
    if "--event-source-mode" not in sys.argv:
        sys.argv += ["--event-source-mode", "cur_event"]
    evaluator.main()


if __name__ == "__main__": main()
