# Mega-HAT

Hypothesis-Anchored Tracking for monocular RGB sequences, using MegaPose
for absolute pose hypotheses and DROID-SLAM for relative motion.

This repository implements Mega-HAT from
[HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation](https://arxiv.org/pdf/2609.21597)
by André Lopo, Atabak Dehban and Rodrigo Ventura.

The default is the **shared causal Mega-HAT configuration from the paper**.
Each output uses only observations available at that frame. Previously emitted
poses stay fixed. The pipeline includes sparse multi-hypothesis MegaPose anchors,
forward hypothesis selection, Sim(3) alignment, SE(3) fusion, disagreement
handling, reanchoring and hold filling.

## Inputs

- An ordered RGB image sequence of one rigid target.
- Calibrated camera intrinsics for the supplied image resolution.
- A target mask or bounding box for each available detection.
- A triangle CAD mesh with explicit units. Textured OBJ/MTL and GLB/glTF,
  colored PLY and geometry-only STL are supported through trimesh.

Images must be undistorted. Keep OBJ materials and texture files at their
referenced relative paths. Convert STEP/IGES to a triangle mesh before use.
Pose quality depends on the appearance and geometry matching the images.

## Installation

See [INSTALL.md](docs/INSTALL.md) for pinned sources, GPU environments,
checkpoints and the included MegaPose compatibility patch. CPU tracking and
replay can be installed independently:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m pip install .
```

## Run a sequence

Prepare the object once, preserving its coordinate origin and axes:

```bash
.venv-mega/bin/mega-hat prepare \
  --mesh data/object/model.obj --units mm --out objects/target
```

Create a sequence manifest using [the mask example](configs/sequence_masks.json)
or [the box example](configs/sequence_boxes.json). Replace the example camera
calibration with your own. Paths are relative to the manifest. Image filenames
are naturally sorted, so `2.png` precedes `10.png`.

```text
data/sequence/
  sequence.json
  rgb/000001.png
  rgb/000002.png
  masks/000001.png
  masks/000002.png
```

Masks must match RGB resolution and filename stems. Zero is background,
nonzero is the target. MegaPose uses the mask's bounding box and the original RGB.
For boxes, provide a JSON object keyed by complete RGB filenames:

```json
{
  "000001.png": [120, 80, 420, 360],
  "000002.png": [122, 81, 422, 361],
  "000003.png": null
}
```

Boxes are `[x1, y1, x2, y2]` pixels with exclusive upper edges. A missing box,
missing/empty mask or `null` means no detection. The RGB frame is still processed.
An eligible detection can produce an anchor once four frames have elapsed since
the previous anchor call. The generic runner applies 20% box padding.

Set interpreter, source and weight paths in [configs/runtime.json](configs/runtime.json):

```bash
.venv/bin/mega-hat validate data/sequence/sequence.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/mega-hat run \
  --sequence data/sequence/sequence.json \
  --runtime configs/runtime.json \
  --object objects/target --out outputs/sequence
```

The runner keeps both models resident. MegaPose's coarse views are rendered
once during startup. It writes `poses.jsonl` incrementally and `poses.npz` on
completion, together with raw observations, configuration, input paths and logs.
Existing output directories are never overwritten.

Poses transform model coordinates into the OpenCV camera frame, **x right,
y down, z forward**, with translations in **metres**. Frames before the first
valid anchor contain `null` in JSON and NaNs in NumPy. Later unavailable outputs
hold the previous pose and are marked `held`.

## Shared configuration

| Parameter | Value |
|---|---:|
| Minimum anchor gap | 4 frames |
| Hypotheses per anchor | 10 |
| MegaPose refinement iterations | 5 |
| Unary / motion transition weights | 3 / 0.3 |
| Fusion motion weight | 0.002 |
| Disagreement gate | 75° |
| Selection / fusion warm-up | 3 / 4 observations |
| Fusion confidence | Uniform |
| Huber scale | 0.3 |
| Alignment update interval | 10 fusion steps |
| Coherent disagreements before reanchoring | 4 anchors |

The complete settings are in the [shared configuration](src/mega_hat/shared.json), also printed
by `mega-hat config`. The shared `f01` recipe accepts temporal selection after
warm-up. It does not apply the running confidence rejection used by Pico-HAT
and historical SPARK-tuned Mega-HAT. [METHOD.md](docs/METHOD.md) explains the
input policies and conventions. Offline studies are separate configurations.

## Replay and Python API

```bash
.venv/bin/mega-hat replay outputs/sequence/observations.npz \
  --out outputs/replayed.npz
```

```python
from mega_hat import Tracker

tracker = Tracker()
# frozen T_WC: current camera-to-world DROID pose, shape (4, 4).
# candidates: (T_CO hypotheses [10, 4, 4], MegaPose logits [10]), or None.
pose, status = tracker.step(slam_T_wc, candidates)
tracker.close()
```

Call `step` once per RGB frame. `tracker.anchor_due` determines when an anchor
is eligible. The Python tracker does not load neural networks or generate
detections. See the method notes for sparse declared-output grids.

## Validation and attribution

[VALIDATION.md](docs/VALIDATION.md) records the release checks and their scope.
The standalone temporal implementation reproduces saved shared-configuration
outputs from all four paper datasets exactly. This is distinct from claiming
new accuracy or throughput for arbitrary inputs or prepared CAD assets.

This implementation builds on [MegaPose](https://github.com/megapose6d/megapose6d)
and [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM). Their source code
and checkpoints are obtained separately. Cite the upstream methods when using
their models. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

If you use this implementation, please cite the [HAT paper](https://arxiv.org/pdf/2609.21597)
and the upstream methods listed above.

```bibtex
@article{lopo2026hat,
  title={HAT: Hypothesis-Anchored Tracking for Video Monocular Spacecraft Pose Estimation},
  author={Lopo, Andr{\'e} and Dehban, Atabak and Ventura, Rodrigo},
  journal={arXiv preprint arXiv:2609.21597},
  year={2026},
  doi={10.48550/arXiv.2609.21597},
  url={https://arxiv.org/pdf/2609.21597}
}
```
