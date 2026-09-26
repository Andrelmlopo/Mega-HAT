# Method and conventions

This package implements the shared causal Mega-HAT recipe (`f01`) from the
[HAT paper](https://arxiv.org/pdf/2609.21597):
anchor gap 4, K=10, five MegaPose RGB refinement iterations, unary scale 3,
rotation transition weight 0.3, fusion weight 0.002, disagreement threshold
75 degrees, uniform confidence, fusion warm-up 4 and Huber scale 0.3.

MegaPose classifies its 576 coarse rotations, refines the top ten hypotheses,
and scores those refined poses. The selector consumes the resulting logits,
which may be negative. Coarse images are cached from one canonical camera
crop at startup. Each query still computes its crop and depth initialization
from the current detection and calibration.

Forward dynamic programming keeps one score per current hypothesis. With
camera-to-world motion poses, the increment is
`R_wc(current).T @ R_wc(previous)`. Rotation penalties are geodesic degrees.
The first two anchors use the best appearance logit. Starting with the third,
the shared configuration uses the forward-score maximum. The historical
running tightness/override gate is disabled in this configuration. There is
no size-dependent unary weighting or dense refiner/confidence branch.

Sim(3) alignment uses the observed prefix. SE(3) fusion optimizes only the
current pose. Alignment is refreshed every ten fusion steps after sufficient
joint anchor/motion observations. A 75-degree disagreement downweights an
anchor by 0.05. Four coherent disagreeing anchors reset alignment. Missing
output holds the latest available pose. No prior emitted pose is revised.

## Inputs and coordinates

MegaPose sees RGB and a bounding box, with 20% padding in the generic runner.
Masks become their enclosing boxes. Fractional supplied box coordinates are
retained. DROID defaults to masked RGB with pixel masks, and to full RGB with
boxes. `slam_input` can set either policy explicitly. Missing masked detections
skip the DROID update. Unmasked background motion may differ from target motion.

DROID uses frontend/local optimization, with frozen per-frame readout. It never
calls offline `terminate()`. Non-keyframes use the two most recent keyframes
and three motion-only iterations. Intrinsics are scaled for resizing before
bottom/right cropping to multiples of eight. Default resize area is 196608
pixels. The paper used native input policies, including 393216 on SwissCube.

`prepare` applies stored scene-node transforms and converts mesh units to metres.
It bakes transforms into a self-contained GLB with identity nodes. The supplied
MegaPose renderer patch disables glTF's implicit axis conversion, so geometry
and rendering use the same coordinates. There is no recentering or object
rotation. Output satisfies `point_camera = R @ point_model_metres + t`.

Prepared meshes, generic padding and canonical render crops may differ from
historical benchmark inputs. Fresh generic inference is not claimed to reproduce
all paper frontend predictions, accuracy or sustained FPS. Exact temporal
replay is checked separately with saved shared-configuration observations.

## Sparse output grids

The generic CLI emits at every RGB frame. The paper's native YCB-V protocol runs
DROID on every input but advances fusion only on declared targets.
`Tracker.step(..., emit=False)` processes a raw-frame motion/candidate opportunity
without advancing fusion or emitting a pose. Anchor gaps still count raw input
frames. Replay accepts an optional sorted integer `output_frames` array for
this fusion grid. Other archives emit at every frame. Replay NPZ files record
both `poses` and `frames`.

## Scope and failure behavior

This is the shared causal method. SPARK-tuned ablations, offline smoothing and
the later thesis localization router use distinct configurations. Detection
generation and ground-truth inputs are outside this runner.

Before the first anchor, no absolute pose exists. Invalid poses or nonfinite
logits raise errors, matching the shared reference frontend's validity contract.
Zero-scale initialization uses a finite unit-scale fallback. DROID buffer
exhaustion raises an error. Indefinite bounded-memory operation is not claimed.
