"""Train geometry-contributing event reliability with a high-exposure teacher."""

from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import finetune_event as fe
from eventvggt.datasets.my_event_dataset import event_multiview_collate
from finetune_mul_ldr_event import _build_dataset, _format_ldr, _to_list
from geo_contribution_event_loss import make_configured_geo_contribution_loss
from launcher import configure_mul_loss_cfg


class GeoTeacherLdrBatchSampler:
    """Each scene is repeated as one high-exposure teacher plus LDR students."""

    def __init__(
        self,
        dataset,
        *,
        scenes_per_batch,
        teacher_ldr_id,
        student_ldr_ids,
        num_views,
        exposures_per_sample=2,
        shuffle=True,
        drop_last=True,
        seed=0,
    ):
        self.dataset = dataset
        self.scenes_per_batch = max(int(scenes_per_batch), 1)
        self.teacher_ldr_id = _format_ldr(teacher_ldr_id)
        self.student_ldr_ids = [_format_ldr(x) for x in student_ldr_ids]
        self.num_views = int(num_views)
        self.exposures_per_sample = max(int(exposures_per_sample), 1)
        self.batch_size = self.scenes_per_batch * self.exposures_per_sample
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.scenes_per_batch
        return (len(self.dataset) + self.scenes_per_batch - 1) // self.scenes_per_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        sample_indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng.shuffle(sample_indices)

        students_per_sample = max(self.exposures_per_sample - 1, 0)
        for batch_idx in range(len(self)):
            start = batch_idx * self.scenes_per_batch
            end = start + self.scenes_per_batch
            base_indices = sample_indices[start:end]
            if len(base_indices) < self.scenes_per_batch and self.drop_last:
                continue

            batch = []
            for sample_idx in base_indices:
                batch.append((int(sample_idx), 0, self.num_views, self.teacher_ldr_id))
                if students_per_sample <= 0:
                    continue
                replace = students_per_sample > len(self.student_ldr_ids)
                chosen = rng.choice(self.student_ldr_ids, size=students_per_sample, replace=replace)
                for ldr_event_id in chosen:
                    batch.append((int(sample_idx), 0, self.num_views, str(ldr_event_id)))
            yield batch

        self.epoch += 1


def _numeric_ldr(value):
    value = str(value)
    if value.startswith("ev_"):
        value = value[3:]
    try:
        return int(value)
    except ValueError:
        return -10**9


def _resolve_teacher_students(cfg, dataset):
    available = dataset.get_active_ldr_events(common=True)
    if not available:
        raise ValueError("No common LDR event ids found in active scenes.")

    requested_teacher = str(getattr(cfg.data, "geo_teacher_ldr_id", "ev_10"))
    if requested_teacher.lower() in {"auto", "hdr", "highest"}:
        teacher = sorted(available, key=_numeric_ldr)[-1]
    else:
        teacher = _format_ldr(requested_teacher)
        if teacher not in available:
            fallback = sorted(available, key=_numeric_ldr)[-1]
            print(f"Requested geo teacher {teacher} is missing; using highest available {fallback}.")
            teacher = fallback

    requested_students = [_format_ldr(x) for x in _to_list(getattr(cfg.data, "geo_student_ldr_ids", []))]
    students = [ldr for ldr in requested_students if ldr in available and ldr != teacher]
    if not students:
        students = [ldr for ldr in available if ldr != teacher]
    if not students:
        students = [teacher]
    return teacher, students, available


def build_geo_teacher_loader(cfg, split="train"):
    if split == "train":
        dataset = _build_dataset(cfg, split, "random")
    else:
        dataset = _build_dataset(cfg, split, getattr(cfg.data, "eval_ldr_event_id", "auto"))

    if len(dataset) <= 0:
        raise ValueError(
            f"Dataset has no valid samples under {cfg.data.root}. "
            f"num_views={cfg.data.num_views}, active_scenes={dataset.get_active_scenes()}"
        )

    if split == "train":
        teacher, students, available = _resolve_teacher_students(cfg, dataset)
        cfg.data.geo_teacher_ldr_id = teacher
        if hasattr(cfg, "loss"):
            cfg.loss.geo_teacher_ldr_id = teacher
        sampler = GeoTeacherLdrBatchSampler(
            dataset,
            scenes_per_batch=getattr(cfg.data, "geo_scenes_per_batch", 1),
            teacher_ldr_id=teacher,
            student_ldr_ids=students,
            num_views=cfg.data.num_views,
            exposures_per_sample=getattr(cfg.data, "geo_exposures_per_sample", 2),
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
        print(
            f"Geo-event teacher loader: teacher={teacher}, students={students}, "
            f"available={available}, scenes_per_batch={sampler.scenes_per_batch}, "
            f"actual_batch={sampler.batch_size}"
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            collate_fn=event_multiview_collate,
        )

    print(
        f"Geo-event {split} loader: eval_ldr={getattr(cfg.data, 'eval_ldr_event_id', 'auto')}, "
        f"active_scenes={dataset.get_active_scenes()}, samples={len(dataset)}"
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=False,
        collate_fn=event_multiview_collate,
    )


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "config"),
    config_name="finetune_event.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    for branch in (cfg.data, cfg.model, cfg.train, cfg.loss, cfg.vis):
        OmegaConf.set_struct(branch, False)

    eval_ldr = getattr(cfg.data, "eval_ldr_event_id", getattr(cfg.data, "ldr_event_id", "ev_5"))
    cfg.data.ldr_event_id = "random"
    cfg.data.eval_ldr_event_id = eval_ldr
    cfg.data.geo_teacher_ldr_id = getattr(cfg.data, "geo_teacher_ldr_id", "ev_10")
    cfg.data.geo_student_ldr_ids = getattr(cfg.data, "geo_student_ldr_ids", ["ev_2", "ev_5"])
    cfg.data.geo_exposures_per_sample = int(getattr(cfg.data, "geo_exposures_per_sample", 2))
    cfg.data.geo_scenes_per_batch = int(getattr(cfg.data, "geo_scenes_per_batch", 1))
    cfg.data.num_views = int(getattr(cfg.data, "geo_num_views", 4))
    cfg.vis.test_max_batches = int(getattr(cfg.vis, "test_max_batches", 2))
    cfg.vis.test_num_views = int(getattr(cfg.vis, "test_num_views", cfg.data.num_views))

    cfg.model.variant = "temporal_reliability_v2"
    cfg.model.event_num_bins = int(getattr(cfg.data, "event_resize_bins", 10))
    cfg.model.event_hidden_dim = 16
    cfg.model.refiner_residual_scale = 0.03
    cfg.model.event_gate_downsample = 2
    cfg.model.event_gate_smooth_kernel = int(getattr(cfg.model, "event_gate_smooth_kernel", 5))
    cfg.model.event_reliability_floor = float(getattr(cfg.model, "event_reliability_floor", 0.25))
    cfg.model.event_reliability_init_bias = float(getattr(cfg.model, "event_reliability_init_bias", 0.25))
    cfg.model.proposal_depth_lowpass = bool(getattr(cfg.model, "proposal_depth_lowpass", True))
    # Events should decide reliability, not directly write depth residuals.
    # Direct event residual prediction tends to learn highlight/noise texture.
    cfg.model.event_proposal_weight = float(getattr(cfg.model, "event_proposal_weight", 0.0))
    cfg.model.exposure_forward_batch_chunk = int(getattr(cfg.model, "exposure_forward_batch_chunk", 1))

    cfg.train.unfreeze_heads = False
    cfg.train.unfreeze_aggregator_blocks = False
    if str(getattr(cfg, "pretrained", "")) in {"", "./ckpt/model.pt"}:
        preferred = Path("./checkpoints/mul_loss_detail_gt_temporal_reliability_v2/checkpoint-last.pth")
        fallback = Path("./checkpoints/mul_loss_detail_gt_temporal_gated/checkpoint-last.pth")
        cfg.pretrained = str(preferred if preferred.exists() else fallback)

    weights = {
        "mv_normal_weight": 0.0,
        "mv_presence_weight": 0.0,
        "mv_hf_weight": 0.0,
        "mv_orient_weight": 0.0,
        "detail_gt_normal_weight": 0.20,
        "detail_gt_hf_weight": 0.30,
        "detail_gt_grad_weight": 0.30,
        "detail_gt_event_boost": 0.0,
        "detail_gt_threshold": 0.02,
        "detail_gt_normal_source": "depth",
        "detail_gt_salient_hf_weight": 0.0,
        "detail_gt_salient_mag_weight": 0.0,
        "detail_gt_salient_presence_weight": 0.0,
        "mv_event_support_mode": "temporal_polarity",
        "mv_event_threshold": 0.20,
        "mv_event_dilate_kernel": 1,
        "mv_event_blur_kernel": 1,
        "mv_event_power": 2.0,
        "mv_event_top_fraction": 0.20,
        "residual_smooth_weight": 0.0,
        "residual_second_order_weight": 0.0,
        "residual_abs_weight": 0.0,
        "final_grid_weight": 0.04,
        "final_phase_weight": 0.02,
        "final_grid_patch_size": 14,
        "final_grid_band": 1,
        "final_grid_detail_threshold": 0.02,
        "v2_residual_target_weight": 0.70,
        "v2_gate_reliability_weight": 0.20,
        "v2_gate_need_floor": 0.10,
        "v2_gate_positive_boost": 2.0,
        "v2_temporal_quality_floor": 0.25,
        "v2_counterfactual_weight": 0.50,
        "v2_counterfactual_margin": 0.15,
        "v2_ldr_final_depth_weight": 0.10,
        "v2_ldr_final_normal_weight": 0.10,
        "v2_ldr_correction_weight": 0.20,
        "v2_ldr_base_weight": 0.10,
        "v2_non_detail_smooth_weight": 0.03,
        "v2_non_detail_second_order_weight": 0.03,
        "v2_target_detail_threshold": 0.02,
        "geo_teacher_ldr_id": cfg.data.geo_teacher_ldr_id,
        "geo_event_target_weight": 0.45,
        "geo_event_reject_weight": 0.08,
        "geo_teacher_consistency_weight": 0.12,
        "geo_event_delta_weight": 0.0,
        "geo_teacher_boost": 0.5,
        "geo_detail_threshold": 0.02,
        "geo_positive_floor": 0.20,
        "geo_negative_margin": 0.25,
    }
    cfg = configure_mul_loss_cfg(
        cfg,
        weights=weights,
        exp_name="mul_loss_detail_gt_geo_event_filter",
    )
    fe.build_event_loader = build_geo_teacher_loader
    fe.EventSupervisedLoss = make_configured_geo_contribution_loss(cfg)
    print(
        "Geo-event teacher training: "
        f"teacher={cfg.data.geo_teacher_ldr_id}, students={_to_list(cfg.data.geo_student_ldr_ids)}, "
        f"eval_ldr={cfg.data.eval_ldr_event_id}, num_views={cfg.data.num_views}, "
        f"pretrained={cfg.pretrained}"
    )
    print(
        "Goal: learn geometry-contributing events from the high-exposure teacher, "
        "then use the learned reliability gate to filter LDR event streams at test time."
    )
    fe.train(cfg)


if __name__ == "__main__":
    run()
