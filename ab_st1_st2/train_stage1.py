"""Train scene-disjoint Stage 1 for ``ours`` or ``no_multildr``.

This entry reuses the isolated proxy/contribution A/B schedule. The
``no_multildr`` condition loads one target exposure only and supervises the
fixed proxy through geometry loss on event support, without a bridge mask.
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader

from ab_st1_st2.dataset import SingleExposureGeometryDataset, single_exposure_collate, stage1_dataset
import paired_token_reliability.train_contribution_stage1 as upstream
from paired_token_reliability.contribution_dataset import contribution_pair_collate
from paired_token_reliability.contribution_stage1 import BridgeMasks


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=("ours", "no_multildr"), required=True)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def main(argv=None) -> None:
    wrapper, remaining = parse_wrapper_args(sys.argv[1:] if argv is None else argv)

    def make_dataset(cfg, split, pairs):
        return stage1_dataset(cfg, split, pairs, wrapper.method)

    def make_loader(dataset, *, batch_size: int, num_workers: int, train: bool):
        collate = (
            single_exposure_collate
            if isinstance(dataset, SingleExposureGeometryDataset)
            else contribution_pair_collate
        )
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=bool(train),
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
        )

    upstream.make_dataset = make_dataset
    upstream.make_loader = make_loader
    original_payload = upstream.checkpoint_payload

    def checkpoint_payload(*args, **kwargs):
        payload = original_payload(*args, **kwargs)
        payload["ablation_method"] = wrapper.method
        payload["reference_exposure_used"] = wrapper.method == "ours"
        payload["scene_split"] = {"train_start": 0, "train_count": 20, "test_start": 20, "test_count": 5}
        return payload

    upstream.checkpoint_payload = checkpoint_payload
    if wrapper.method == "no_multildr":
        # No reference exposure and no bridge construction. Only event support
        # defines where the fixed proxy receives the geometry objective.
        def event_support_masks(_rgb_reference, rgb_target, event_voxel, **_kwargs):
            support = event_voxel.float().abs().sum(dim=2) > 0.0
            empty = torch.zeros_like(support)
            area = support.flatten(1).float().mean(dim=1)
            return BridgeMasks(empty, empty, support, empty, area)

        upstream.build_bridge_masks = event_support_masks
        remaining.extend(["--supervision-region", "event_support"])
    print(
        f"[ablation Stage1] method={wrapper.method}; train scenes=20, test scenes=5; "
        f"reference exposure={'enabled' if wrapper.method == 'ours' else 'disabled'}",
        flush=True,
    )
    upstream.main(remaining)


if __name__ == "__main__":
    main()
