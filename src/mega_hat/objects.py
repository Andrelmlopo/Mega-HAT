"""Portable textured assets with explicit metric scale and unchanged object axes."""

import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_object(mesh_path, units, output):
    import trimesh

    mesh_path, output = Path(mesh_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}[units]
    source = trimesh.load(mesh_path, force="scene", process=False)
    scene = trimesh.Scene()
    vertices = []
    for name in source.graph.nodes_geometry:
        transform, geometry = source.graph[name]
        mesh = source.geometry[geometry].copy()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("Expected a triangle mesh, not a point cloud")
        mesh.apply_transform(transform)
        mesh.apply_scale(scale)
        vertices.append(np.asarray(mesh.vertices))
        # Bake node transforms into vertices. MegaPose's geometry reader ignores
        # scene transforms, so both geometry and renderer must see identity nodes.
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    if not vertices:
        raise ValueError("The mesh contains no geometry")
    points = np.concatenate(vertices)
    if not np.isfinite(points).all() or np.linalg.norm(np.ptp(points, axis=0)) <= 0:
        raise ValueError("Mesh vertices must be finite with nonzero extent")
    content = scene.export(file_type="glb")
    output.mkdir(parents=True)
    (output / "model.glb").write_bytes(content)
    metadata = {
        "format": "mega-hat.object.v1",
        "mesh": mesh_path.name,
        "mesh_units": units,
        "source_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256(content).hexdigest(),
        "translation_units": "m",
        "pose_convention": "object-to-camera",
        "object_frame": "source scene with node transforms applied; no recentering or axis conversion",
        "bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
