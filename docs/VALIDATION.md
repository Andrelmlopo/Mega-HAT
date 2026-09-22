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
