# Installation

GPU inference requires Linux, an NVIDIA CUDA GPU, and the EGL/OpenGL driver
runtime. The release is validated on an A100 with CUDA 11.8 PyTorch wheels.
DROID extensions additionally need a CUDA toolkit, C++ compiler and `nvcc`.
Both models stay resident. Smaller GPUs have not been profiled.

Run commands from the repository root. Separate environments keep MegaPose's
rendering dependencies separate from DROID's compiled extensions.

## Sources

```bash
python3 scripts/fetch_sources.py
```

| Source | Revision |
|---|---|
| MegaPose | `f3b8e1247f133f3d098833a251b8f2d744c03e1f` |
| DROID-SLAM | `2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99` |

The helper applies [megapose-runtime.patch](../patches/megapose-runtime.patch):
virtual-environment Python lookup, NumPy 2 types, headless EGL rendering and
explicit glTF axis preservation. It also permits the requested mesh-point count
for small CAD meshes by sampling vertices with replacement when necessary;
sampling for meshes with enough vertices is unchanged.
It accepts an existing checkout only if its
revision and tracked changes match this release. DROID submodules are fetched
at their recorded revisions. Upstream sources are excluded from this repository.

## CPU coordinator

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m pip install .
.venv/bin/mega-hat config
```

## MegaPose

```bash
python3.10 -m venv .venv-mega
.venv-mega/bin/python -m pip install --upgrade 'setuptools<81' wheel
.venv-mega/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-mega/bin/python -m pip install -r requirements-mega.txt
.venv-mega/bin/python -m pip install --no-deps .
.venv-mega/bin/python -m pip check
.venv-mega/bin/python -c "import pinocchio, png"
```

The adapter imports MegaPose directly from the pinned checkout. MegaPose's
`pin` dependency supplies Pinocchio. Its binary companion versions are pinned
alongside it, including TinyXML 10: the Pinocchio/urdfdom wheels require
`libtinyxml2.so.10`, which TinyXML 11 does not provide. `pypng` is required by
MegaPose's bundled BOP toolkit even for inference. The import check above catches
missing native libraries that `pip check` cannot detect.
Install the NVIDIA EGL driver libraries for hardware rendering.
When necessary, point `__EGL_VENDOR_LIBRARY_FILENAMES` at your NVIDIA EGL vendor
JSON. Do not use a machine-specific file path copied from another system.

Use `mega-hat prepare` in this environment. It converts meshes to a metric,
self-contained GLB while preserving appearance, scene transforms and object
coordinates. No model weights or GPU are needed for this preparation step.
A separate CPU environment can install `.[prepare]` for the same command.

## DROID-SLAM

```bash
python3.10 -m venv .venv-droid
.venv-droid/bin/python -m pip install 'setuptools<81' wheel ninja
.venv-droid/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-droid/bin/python -m pip install -r requirements-droid.txt
.venv-droid/bin/python -m pip install --no-build-isolation \
  ./third_party/DROID-SLAM/thirdparty/lietorch
.venv-droid/bin/python -m pip install --no-build-isolation \
  ./third_party/DROID-SLAM/thirdparty/pytorch_scatter
.venv-droid/bin/python -m pip install --no-build-isolation ./third_party/DROID-SLAM
.venv-droid/bin/python -m pip install --no-deps .
.venv-droid/bin/python -m pip check
```

Set `CUDA_HOME` to your CUDA toolkit and put its `bin` directory on `PATH`
before compiling. `MAX_JOBS=4` limits build memory use. The runtime imports
`droid_backends`, `lietorch` and `torch_scatter`.

## Checkpoints

Obtain MegaPose's RGB models from the
[official model archive](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/),
linked by the pinned upstream README. Keep this directory structure:

```text
weights/
  megapose-models/
    coarse-rgb-906902141/
      config.yaml
      checkpoint.pth.tar
    refiner-rgb-653307694/
      config.yaml
      checkpoint.pth.tar
  droid.pth
```

The four MegaPose files are checked against the release's
[SHA-256 manifest](../src/mega_hat/weights.json) before the upstream loader reads
them. The RGBD model is not needed. Get `droid.pth` from the
[DROID authors' checkpoint](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view).
The validated DROID digest is
`46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526`.
Sources and weights retain their upstream terms and are not bundled here.

Set paths in `configs/runtime.json`. `mega_data` is the `weights` directory,
not the `megapose-models` subdirectory. Relative paths resolve from the runtime
JSON's directory. Select one physical GPU with `CUDA_VISIBLE_DEVICES` when
starting the coordinator. Workers start once and stop after completion or errors.

## Tests and build

```bash
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m build
```

The default suite is CPU-only. GPU integration also needs the weights, compiled
DROID extensions and a calibrated RGB sequence. See [VALIDATION.md](VALIDATION.md)
for the precise scope of the release checks.
