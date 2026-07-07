"""Use the established held-out counterfactual evaluator with the new model."""

from eventvggt.models.streamvggt_paired_token_reliability_detail import StreamVGGT
import real_reliability_stage.evaluate_stage2_heldout as evaluator


if __name__ == "__main__":
    evaluator.StreamVGGT = StreamVGGT
    evaluator.main()
