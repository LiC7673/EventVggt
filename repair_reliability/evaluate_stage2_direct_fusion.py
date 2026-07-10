"""Held-out counterfactual evaluation for direct input fusion."""

from eventvggt.models.streamvggt_reliability_direct_fusion import StreamVGGT
import real_reliability_stage.evaluate_stage2_heldout as evaluator


if __name__ == "__main__":
    evaluator.StreamVGGT = StreamVGGT
    evaluator.main()
