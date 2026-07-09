# 3. Method

## 3.1 Overview

We address multi-view 3D geometry estimation from low-dynamic-range (LDR)
images and asynchronous events under challenging exposure and reflective
appearance. Given a sequence of LDR observations
$\mathcal{I}=\{I_i^{e_i}\}_{i=1}^{N}$, their camera-aligned event streams
$\mathcal{E}=\{E_i\}_{i=1}^{N}$, and camera intrinsics, our goal is to estimate
per-view depth $D_i$, 3D points $P_i$, and camera poses $T_i$. The superscript
$e_i$ denotes the exposure level. At test time, only one LDR exposure and its
corresponding event stream are required.

The key difficulty is that events are exposure-robust but not necessarily
geometry-reliable. Object motion and occlusion boundaries provide useful
geometric events, whereas reflections, saturation transitions, and sensor
noise can produce events that do not correspond to surface discontinuities.
Directly fusing all events therefore transfers appearance-dependent noise into
the predicted depth. Conversely, aggressively filtering events removes the
sparse structures that are most useful for recovering fine geometry.

Our method resolves this tension through a progressive evidence-distillation
pipeline. First, **paired-exposure token alignment** identifies image features
that remain stable across LDR levels. Second, **geometry-detail supervision**
uses GT depth to distinguish geometric structures from appearance variation.
Third, a lightweight **event reliability network** distills both cues into a
dense reliability map. During geometry fine-tuning, the complete event voxel is
still processed by the event branch; reliability controls only how much of the
predicted event residual is written into the final depth. This preserves event
details while suppressing unreliable corrections.

## 3.2 Event Representation and Coarse Geometry

For each image interval, we divide events into $B$ temporal bins and separate
positive and negative polarities. This produces a voxel grid

$$
V_i \in \mathbb{R}^{2B\times H\times W}.
$$

Spatial resizing is performed on the voxel grid with antialiasing rather than
by directly dividing event coordinates. The temporal order and polarity are
therefore retained. Before event encoding, event counts are compressed as

$$
\hat V_i =
\frac{\log(1+\operatorname{clip}(V_i,0,c_{\max}))}
{\log(1+c_{\max})}.
$$

We use VGGT as the RGB geometry backbone. Its image aggregator extracts
multi-view patch tokens, and the pretrained depth, point, and camera heads
predict coarse geometry

$$
(D_i^c,P_i^c,T_i)=\mathcal{G}_{\mathrm{RGB}}(\mathcal I).
$$

The coarse branch provides globally consistent shape and pose, while events
are reserved for spatially localized geometric refinement.

## 3.3 Paired-Exposure Token Alignment

During training, we sample two LDR renderings $I_i^{a}$ and $I_i^{b}$ of the
same scene, frame interval, pose, and event stream. Thus, the image appearance
changes while the underlying geometry and event observation remain fixed. Let
$Z_i^e$ denote the patch tokens of exposure $e$. We insert a bounded residual
adapter at selected aggregator layers:

$$
\widetilde Z_i^e = Z_i^e + \alpha\tanh\left[
W_2\,\sigma\left(W_1\operatorname{LN}(Z_i^e)\right)\right],
$$

where $\alpha$ limits the correction magnitude. The final projection $W_2$ is
zero-initialized, so training starts from the pretrained VGGT representation
instead of perturbing it abruptly.

For each exposure pair, we select the less saturated observation as the anchor
using

$$
q(I)=1-\operatorname{Sat}(I)+\eta\operatorname{Contrast}(I).
$$

The other exposure is aligned to a stop-gradient anchor with

$$
\mathcal L_{\mathrm{tok}} =
\operatorname{SmoothL1}\left(
\operatorname{LN}(\widetilde Z_i^{b}),
\operatorname{sg}[\operatorname{LN}(\widetilde Z_i^{a})]
\right).
$$

Unlike output-level consistency, this constraint does not force two depth maps
to converge to an over-smoothed average. It removes exposure variation before
the dense geometry decoder while preserving its ability to reconstruct sharp
depth changes. The cosine agreement between aligned tokens is retained as a
spatial confidence cue:

$$
C_i^{\mathrm{exp}}(u)=
\operatorname{Up}\left[
\frac{\cos(\widetilde Z_i^a,\widetilde Z_i^b)-\tau_c}
{1-\tau_c}
\right]_{[0,1]}(u).
$$

Here $\operatorname{Up}$ bilinearly maps the patch confidence to image
resolution. This confidence measures exposure stability, not geometric
reliability by itself.

## 3.4 Geometry-Detail Supervision

Global depth losses are dominated by low-frequency surfaces and can obtain good
average metrics while missing fingers, thin boundaries, or small concavities.
We therefore construct a geometry-detail weighting map from GT depth. First,
we unproject GT depth using the camera intrinsics and estimate surface normals
$N_i^{\mathrm{gt}}$. The detail magnitude is

$$
G_i(u)=\mathcal N\left(
\left\|\nabla N_i^{\mathrm{gt}}(u)\right\|_2;\tau_g
\right),
$$

where $\mathcal N(\cdot;\tau_g)$ performs thresholded robust normalization.
Let $S_i^{\mathrm{evt}}$ be event support computed from polarity-separated
temporal bins. We use events as an emphasis cue rather than as the GT label:

$$
W_i^{\mathrm{detail}} = M_iG_i
\left(1+\beta S_i^{\mathrm{evt}}\right),
$$

where $M_i$ is the valid-depth mask. This distinction is important: an event
does not imply a normal discontinuity, but an event located at a GT geometric
detail deserves stronger supervision.

The detail objective combines normal orientation, normal-gradient magnitude,
and high-frequency normal residuals:

$$
\begin{aligned}
\mathcal L_{\mathrm{detail}} ={}&
\lambda_n\,\mathbb E_{W^{\mathrm{detail}}}
[1-\langle \hat N,N^{\mathrm{gt}}\rangle] \\
&+\lambda_g\,\mathbb E_{W^{\mathrm{detail}}}
[|\nabla\hat N-\nabla N^{\mathrm{gt}}|] \\
&+\lambda_h\,\mathbb E_{W^{\mathrm{detail}}}
[|\operatorname{HF}(\hat N)-\operatorname{HF}(N^{\mathrm{gt}})|].
\end{aligned}
$$

This objective explicitly allocates gradient budget to geometric details that
are weakly represented by the standard depth and point losses.

## 3.5 Exposure-Consistent Geometry Reliability

### Reliability target

Exposure-stable tokens alone may correspond to texture, while GT normal
gradients alone do not indicate whether an event was observed. We combine the
three complementary cues into a soft target:

$$
R_i^{\mathrm{gt}}(u)=
S_i^{\mathrm{evt}}(u)\,
G_i(u)\,
C_i^{\mathrm{exp}}(u).
$$

A $3\times3$ max-pooling operation slightly dilates the target to tolerate
small spatial misalignment between event edges and depth-derived normal
gradients. We deliberately avoid Gaussian smoothing, which was found to erase
thin structures and turn reliability into a low-frequency foreground mask.

### Reliability network

We use a compact U-Net $\mathcal R_\phi$ with the full polarity-separated
event voxel and one LDR image as input:

$$
R_i^e=\sigma\left(\mathcal R_\phi([\hat V_i,I_i^e])\right),
\qquad R_i^e\in[0,1]^{H\times W}.
$$

The network is trained on both exposures with weighted binary cross-entropy,
cross-exposure consistency, and a ranking loss:

$$
\mathcal L_R =
\frac{1}{2}\sum_{e\in\{a,b\}}
\operatorname{WBCE}(R_i^e,R_i^{\mathrm{gt}})
+\lambda_c\|R_i^a-R_i^b\|_1
+\lambda_r\mathcal L_{\mathrm{rank}}.
$$

The ranking term requires predictions on high-target pixels to exceed those on
non-geometric event pixels by a margin. Consequently, the network cannot
minimize the loss by predicting a spatially uniform confidence. Since the two
training inputs share events but differ in exposure, the consistency term
encourages reliability to depend on geometry-compatible evidence rather than
on a specific brightness level.

## 3.6 Reliability-Gated Event Refinement

The temporal detail branch predicts a bounded log-depth correction from the
complete event voxel, RGB context, and coarse depth:

$$
\delta_i=\mathcal F_{\mathrm{evt}}(V_i,I_i,D_i^c).
$$

A crucial design choice is that we do **not** use $R_i$ as an input mask. In
particular, we avoid replacing $V_i$ by $R_i\odot V_i$, because an imperfect
reliability map would irreversibly remove sparse event details before temporal
encoding. Instead, we dilate the predicted reliability and construct a soft
output gate

$$
\bar R_i=\operatorname{MaxPool}_{3\times3}(R_i),\qquad
A_i=r_0+(1-r_0)\bar R_i,
$$

where $r_0>0$ preserves a weak correction path for uncertain events. The final
depth and points are

$$
D_i=D_i^c\exp(A_i\odot\delta_i),\qquad
P_i=P_i^c\frac{D_i}{D_i^c}.
$$

Thus, all events can contribute to feature extraction, but only reliable event
evidence can produce a strong geometric correction. The residual is additionally
bounded and constrained to be high-frequency and locally zero-mean, reducing
low-frequency depth drift and patch-grid artifacts.

## 3.7 Training Objective and Schedule

The base geometry loss contains depth, point, and optional pose supervision:

$$
\mathcal L_{\mathrm{base}}=
\lambda_d\mathcal L_d+
\lambda_p\mathcal L_p+
\lambda_T\mathcal L_T.
$$

The geometry stage optimizes

$$
\mathcal L_{\mathrm{geo}}=
\mathcal L_{\mathrm{base}}+
\lambda_{\mathrm{detail}}\mathcal L_{\mathrm{detail}}+
\lambda_{\mathrm{res}}\mathcal L_{\mathrm{res}}+
\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}},
$$

where $\mathcal L_{\mathrm{res}}$ supervises the event-conditioned high-pass
log-depth residual and $\mathcal L_{\mathrm{reg}}$ bounds residual magnitude
and suppresses grid-periodic responses.

Training proceeds in three steps. (1) We train the bounded token adapter using
paired LDR observations and $\mathcal L_{\mathrm{tok}}$. (2) We freeze the
paired-token teacher, generate $R^{\mathrm{gt}}$, and train the standalone
ReliabilityUNet with $\mathcal L_R$. (3) We freeze ReliabilityUNet and fine-tune
the VGGT depth/point heads and temporal event refiner with
$\mathcal L_{\mathrm{geo}}$. At inference, the paired exposure and GT depth are
removed; the model consumes only a single LDR sequence and its events.

