from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate, get_combined_dataset


class RandomLdrBatchSampler:
    """Yield batches whose samples share one randomly selected LDR exposure."""

    def __init__(
        self,
        dataset,
        batch_size,
        ldr_event_ids,
        *,
        num_views,
        shuffle=True,
        drop_last=True,
        seed=0,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not ldr_event_ids:
            raise ValueError("RandomLdrBatchSampler needs at least one LDR exposure")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.ldr_event_ids = list(ldr_event_ids)
        self.num_views = int(num_views)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        sample_indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng.shuffle(sample_indices)

        batch_count = len(self)
        for batch_idx in range(batch_count):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_indices = sample_indices[start:end]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue

            ldr_event_id = str(rng.choice(self.ldr_event_ids))
            yield [
                (int(sample_idx), 0, self.num_views, ldr_event_id)
                for sample_idx in batch_indices
            ]

        self.epoch += 1


def _build_dataset(cfg, split, ldr_event_id):
    return get_combined_dataset(
        root=cfg.data.root,
        num_views=cfg.data.num_views,
        resolution=tuple(cfg.data.resolution),
        fps=cfg.data.fps,
        seed=cfg.seed,
        scene_names=cfg.data.scene_names if cfg.data.scene_names else None,
        initial_scene_idx=cfg.data.initial_scene_idx,
        active_scene_count=cfg.data.active_scene_count,
        split=split,
        test_frame_count=getattr(cfg.data, "test_frame_count", 10),
        ldr_event_id=ldr_event_id,
    )


def build_random_ldr_event_loader(cfg, split="train"):
    random_train_ldr = bool(getattr(cfg.data, "random_train_ldr", True))
    if split == "train" and random_train_ldr:
        ldr_event_id = getattr(cfg.data, "ldr_event_id", "random")
    else:
        ldr_event_id = getattr(cfg.data, "eval_ldr_event_id", "auto")

    dataset = _build_dataset(cfg, split, ldr_event_id)
    if len(dataset) <= 0:
        raise ValueError(
            f"Dataset has no valid samples under {cfg.data.root}. "
            f"num_views={cfg.data.num_views}, active_scenes={dataset.get_active_scenes()}"
        )

    if split == "train" and random_train_ldr:
        ldr_event_ids = dataset.get_active_ldr_events(common=True)
        if not ldr_event_ids:
            raise ValueError(
                "No common LDR exposure was found across the active scenes. "
                "Use scenes with overlapping LDR/ev_xx folders or reduce active_scene_count."
            )
        batch_sampler = RandomLdrBatchSampler(
            dataset,
            batch_size=cfg.batch_size,
            ldr_event_ids=ldr_event_ids,
            num_views=cfg.data.num_views,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=event_multiview_collate,
        )
        fe.printer.info(
            "Random-LDR train dataset loaded from %s with %d active scenes, %d samples, exposures=%s",
            cfg.data.root,
            len(dataset.get_active_scenes()),
            len(dataset),
            ldr_event_ids,
        )
        return data_loader

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )
    fe.printer.info(
        "Fixed-LDR %s dataset loaded from %s with %d active scenes, %d samples, exposures=%s",
        split,
        cfg.data.root,
        len(dataset.get_active_scenes()),
        len(dataset),
        dataset.get_active_ldr_events(common=False),
    )
    return data_loader


def launch(cfg) -> None:
    OmegaConf.resolve(cfg)
    fe.build_event_loader = build_random_ldr_event_loader
    fe.train(cfg)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event_random_ldr.yaml",
)
def run(cfg: OmegaConf):
    launch(cfg)


if __name__ == "__main__":
    run()
