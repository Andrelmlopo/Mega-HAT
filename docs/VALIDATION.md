# Release validation

Version 0.1.0, 22 September 2026. Machine-readable receipts are in
[validation.json](validation.json).

## Temporal equivalence

The standalone shared configuration reproduces the saved native `f01` outputs
**bit for bit** on four representative full streams:

| Dataset | Input frames | Output frames |
|---|---:|---:|
| SPARK | 300 | 300 |
| YCB-V | 2243 | 75 declared targets |
| SwissCube | 100 | 100 |
| SHIRT | 2371 | 2371 |

All 5014 input frames were processed, with 2846 compared outputs. Saved
hypotheses, logits and frozen motion poses were replayed. Input-file hashes
are recorded in the receipt. YCB-V retains its native sparse fusion grid.
These checks cover temporal extraction, not a new full-benchmark accuracy run.

## Tests and packaging

- 23 CPU tests pass with Python 3.10, NumPy 2.2.6 and SciPy 1.15.3.
- Tests cover pose direction/scale, causal prefixes, future-observation changes,
  buffer growth, missing observations, negative logits, the shared warm-up rule,
  sparse output grids, replay, localization, immutable output paths, interpreter
  symlinks, checkpoint rejection and prepared mesh geometry/materials.
- A separate native Panda3D/EGL test passes for textured OBJ input, an off-centre
  object origin, camera axes, visible texture colors and reconstructed depth.
  CPU preparation tests check metre/millimetre equivalence and scene transforms.
- Lint, formatting, source-distribution and wheel builds pass. The installed
  wheel is tested in an isolated CPU environment from outside the repository.
- The pinned-source helper accepts its exact patched checkout on repeated use.

To run the optional renderer test, install the native dependencies and set
`MEGA_HAT_RENDER_TESTS=1`, `MEGAPOSE_SOURCE` to the fetched MegaPose checkout,
`MEGAPOSE_DATA_DIR` to the model directory, and `CUDA_VISIBLE_DEVICES=0` before
running `pytest tests/test_rendering.py` in the MegaPose environment.

## Fresh inference

Two complete RT500 runs use the prepared textured PROBA-2 GLB:

| Input | Frames | Valid poses | Anchor calls | Valid motion frames |
|---|---:|---:|---:|---:|
| Pixel masks | 300 | 300 | 75 | 280 |
| Bounding boxes, masked motion input | 300 | 300 | 75 | 267 |

Both runs reload and replay exactly from their emitted observation archives.
Each anchor has ten hypotheses and five refinement iterations. Both GPU workers
exit after completion. No inference job is left running.

The MegaPose source is a clean checkout of the pinned upstream revision with
only the distributed compatibility/coordinate patch. Existing GPU Python
dependencies, pretrained weights and compiled DROID extensions were reused.
This is not a fresh installation of every GPU dependency. The CPU installation
and wheel checks use an isolated environment.

These are execution and consistency checks. They do not establish new
benchmark accuracy, sustained FPS, identical historical frontend predictions,
or performance on unseen objects. No detector generation, network training,
parameter tuning or manuscript update was part of this release.

## Fresh installation and subset checks, 26 September 2026

A new checkout and new Python environments were tested on an 8 GB RTX 3070
Laptop GPU. PicoPose used Python 3.9/PyTorch 2.0.0+cu118; MegaPose and DROID
used Python 3.10/PyTorch 2.7.1+cu118. All three DROID extensions were built
from pinned sources with CUDA 11.8 and GCC 11.5. Checkpoint hashes and repeated
source-helper checks passed. See the [dated receipt](release_validation_20260926.json).

The 23 CPU tests, three optional native checks, installed-wheel tests, dependency
checks, lint, formatting and package builds pass. Fresh installation exposed
a missing TinyXML 10 runtime pin and missing `pypng`. Valid sparse CAD meshes
also exposed a MegaPose sampling assertion. The requirements and distributed
patch now fix these failures; native tests cover inference imports, deterministic
sparse sampling, unchanged dense sampling and textured rendering.

Both HAT methods ran through SpaceRGBbenchmark on 300 dense YCB-V inputs
(scene 48/object 6; five scored targets), 36 SwissCube inputs (seq_000400),
and 36 inputs from each of SHIRT's four streams. Each method emitted valid
SE(3) poses for all 480 inputs. Saved observations reproduce every pose exactly
through the API and standalone replay CLI. Five causal prefixes per stream
match exactly (60 checks across both methods).

These checks have material limits:

- SPARK 2024 was not rerun because the restricted dataset and matching inputs
  were unavailable. Earlier validation receipts are separate evidence.
- SwissCube's declared 393216-pixel DROID resolution exhausted this GPU.
  The successful development run explicitly uses 196608 pixels and buffer 64;
  this does not validate the published default on 8 GB. The frozen schedule
  and published resolution remain unchanged.
- SHIRT used a local Tango CAD whose correspondence to the historical calibrated
  origin is unverified. Its accuracy scores are not release evidence.
- YCB-V produced 138 valid DROID motion observations per method but too few
  usable anchors for fusion: 295/300 outputs were held. Pico-HAT also held
  35/36 outputs on SHIRT roe2_synthetic without initializing fusion. Valid
  poses and exact replay do not establish useful tracking accuracy.
- This is not a full-partition accuracy or repeated-FPS experiment. Metadata
  matches all 41 YCB-V streams, 100 SwissCube sequences (8,522 targets) and
  four SHIRT streams (9,484 targets), but neural inference used subsets only.
