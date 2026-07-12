# Event-VGGT Geometry Adapter Ablation Experiments

Goal:

Generate four independent experimental variants to diagnose current event refinement failure.

Do NOT modify the original implementation directly.

Create isolated experiment folders and configs.

Each experiment should be runnable independently.

A runner script should launch all experiments on different GPUs.

============================

Experiment A:
Remove RGB shortcut

Hypothesis:
Current adapter learns RGB-conditioned refinement instead of event-conditioned geometry refinement.

Modification:

Before:

update = Adapter(concat(rgb_feature,event_feature))

After:

update = Adapter(event_feature)

Keep:

- contribution
- losses
- dataset
- training schedule


Expected validation:

1.
zero event should produce almost zero update.

2.
RGB feature shuffle should not change event update significantly.

Save:

- update magnitude
- depth difference map
- normal error


============================

Experiment B:
Soft contribution gating

Hypothesis:

Hard threshold contribution creates patch boundary artifact.

Modification:

Remove:

contribution > threshold

Use:

update = contribution * raw_update


Do not change:

event encoder.


Validation:

visualize:

- contribution
- update
- normal map


============================

Experiment C:
Event support constrained update

Hypothesis:

Sparse event becomes dense after encoder.

Modification:

Compute original event support:

event_mask = abs(event_voxel).sum(channel)>0


Build support pyramid using bilinear interpolation.


Apply:

update =
event_support *
contribution *
raw_update


Validation:

Compare:

full event
zero event

Depth difference outside event region should be near zero.


============================

Experiment D:
High resolution event injection

Hypothesis:

All-scale injection from shared 28x37 grid causes periodic artifact.

Modification:

Only inject event update into high resolution DPT layers.

Default:

levels=[0,1]

Disable:

levels=[2,3]


Validation:

Compare normal stripe artifact.


============================

Generate:

1.
four experiment folders

2.
four config files

3.
four train scripts

4.
one run_all_ablation.sh


Each experiment must:

- save checkpoint
- save tensorboard log
- save qualitative visualization
- print:
  Cmean
  Cstd
  update norm
  zero-event difference


Do not change:

- dataset
- ContributionNet supervision
- Multi-LDR consistency

unless required.

Before editing:
list affected files and expected tensor shape changes.