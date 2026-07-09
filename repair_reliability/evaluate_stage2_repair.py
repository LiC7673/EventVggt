"""Held-out counterfactual evaluation for the repair reliability model."""

from eventvggt.models.streamvggt_paired_token_reliability_repair import StreamVGGT
import real_reliability_stage.evaluate_stage2_heldout as evaluator


if __name__ == "__main__":
    evaluator.StreamVGGT = StreamVGGT
    evaluator.main()
