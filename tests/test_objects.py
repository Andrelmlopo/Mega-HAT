import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from mega_hat.objects import prepare_object


def textured_box():
    mesh = trimesh.creation.box(extents=[0.3, 0.2, 0.1])
    mesh.apply_translation([0.4, -0.2, 0.15])
    uv = (mesh.vertices[:, :2] - mesh.vertices[:, :2].min(0)) / np.ptp(mesh.vertices[:, :2], axis=0)
    color = np.full((16, 16, 3), [240, 70, 20], np.uint8)
    color[:, 8:] = [20, 200, 100]
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=Image.fromarray(color))
    return mesh


def test_obj_materials_units_and_offcenter_origin_survive_preparation(tmp_path):
    mesh = textured_box()
    mesh.export(tmp_path / "model.obj")
    prepare_object(tmp_path / "model.obj", "m", tmp_path / "metres")
    loaded = trimesh.load(tmp_path / "metres/model.glb", force="scene", process=False)
    np.testing.assert_allclose(loaded.bounds, mesh.bounds, atol=1e-7)
    material = next(iter(loaded.geometry.values())).visual.material
    assert np.std(np.asarray(material.baseColorTexture)[..., 0]) > 20
    mesh.apply_scale(1000)
    mesh.export(tmp_path / "millimetres.obj")
    prepare_object(tmp_path / "millimetres.obj", "mm", tmp_path / "millimetres")
    other = trimesh.load(tmp_path / "millimetres/model.glb", force="scene", process=False)
    np.testing.assert_allclose(other.bounds, loaded.bounds, atol=1e-7)
    with pytest.raises(ValueError, match="already exists"):
        prepare_object(tmp_path / "model.obj", "m", tmp_path / "metres")


def test_scene_transforms_are_baked_into_geometry(tmp_path):
    scene = trimesh.Scene()
    mesh = textured_box()
    transform = trimesh.transformations.rotation_matrix(0.7, [1, 0, 0])
    transform[:3, 3] = [1, 2, 3]
    scene.add_geometry(mesh, transform=transform)
    scene.export(tmp_path / "scene.glb")
    result = prepare_object(tmp_path / "scene.glb", "m", tmp_path / "object")
    loaded = trimesh.load(tmp_path / "object/model.glb", force="scene", process=False)
    np.testing.assert_allclose(loaded.bounds, scene.bounds, atol=1e-6)
    for node in loaded.graph.nodes_geometry:
        matrix, _ = loaded.graph[node]
        np.testing.assert_array_equal(matrix, np.eye(4))
    assert json.loads((tmp_path / "object/metadata.json").read_text()) == result
