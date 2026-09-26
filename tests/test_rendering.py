"""Optional native EGL check, enabled with MEGA_HAT_RENDER_TESTS=1."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MEGA_HAT_RENDER_TESTS") != "1", reason="Requires native MegaPose and EGL"
)


def test_native_inference_dependencies_import():
    sys.path.insert(0, str(Path(os.environ["MEGAPOSE_SOURCE"]) / "src"))
    from megapose.inference.pose_estimator import PoseEstimator

    assert callable(PoseEstimator)


def test_native_mesh_sampling_supports_small_cad():
    sys.path.insert(0, str(Path(os.environ["MEGAPOSE_SOURCE"]) / "src"))
    import torch
    from megapose.lib3d.mesh_ops import sample_points

    sparse = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    sampled = sample_points(sparse, 2000, deterministic=True)
    assert sampled.shape == (1, 2000, 3)
    assert torch.equal(sampled, sample_points(sparse, 2000, deterministic=True))
    assert all(any(torch.equal(p, v) for v in sparse[0]) for p in sampled[0])
    dense = torch.arange(9000, dtype=torch.float32).reshape(1, 3000, 3)
    original_ids = np.random.RandomState(0).choice(3000, size=2000, replace=False)
    assert torch.equal(sample_points(dense, 2000, deterministic=True), dense[:, original_ids])


def test_native_renderer_preserves_textured_obj_coordinates(tmp_path):
    sys.path.insert(0, str(Path(os.environ["MEGAPOSE_SOURCE"]) / "src"))
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
    from megapose.lib3d.transform import Transform
    from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
    from megapose.panda3d_renderer.types import (
        Panda3dCameraData,
        Panda3dLightData,
        Panda3dObjectData,
    )
    from test_objects import textured_box

    from mega_hat.objects import prepare_object

    mesh = textured_box()
    mesh.export(tmp_path / "model.obj")
    prepare_object(tmp_path / "model.obj", "m", tmp_path / "object")
    objects = RigidObjectDataset(
        [RigidObject(label="target", mesh_path=tmp_path / "object/model.glb", mesh_units="m")]
    )
    renderer = Panda3dSceneRenderer(objects)
    K = np.array([[570.0, 0, 320], [0, 570.0, 240], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, 3] = [-0.4, 0.2, 1.0]
    output = renderer.render_scene(
        [Panda3dObjectData(label="target", TWO=Transform(pose))],
        [Panda3dCameraData(K=K, resolution=(480, 640), TWC=Transform(np.eye(4)))],
        [Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))],
        render_depth=True,
    )[0]
    z = output.depth.squeeze()
    ys, xs = np.nonzero(z > 0)
    assert len(xs) > 100
    depth = z[ys, xs]
    camera = np.stack(
        ((xs - K[0, 2]) * depth / K[0, 0], (ys - K[1, 2]) * depth / K[1, 1], depth), axis=1
    )
    model = camera - pose[:3, 3]
    assert np.max(np.abs(model[:, 2] - 0.10)) < 0.001
    tolerance = 2 * depth.max() / 570
    assert model[:, 0].min() > 0.25 - tolerance and model[:, 0].max() < 0.55 + tolerance
    assert model[:, 1].min() > -0.30 - tolerance and model[:, 1].max() < -0.10 + tolerance
    assert np.std(output.rgb[ys, xs, 0]) > 20
