"""Resident MegaPose RGB hypotheses for one prepared textured object."""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from .config import shared_config


class MegaPose:
    def __init__(self, source, data, object_dir, sequence):
        source, data, object_dir = Path(source), Path(data), Path(object_dir)
        sys.path.insert(0, str(source / "src"))
        os.environ["MEGAPOSE_DATA_DIR"] = str(data)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        # The upstream loader reads YAML model metadata. Accept only the
        # published model files checked with this release.
        from .weights import verify_weights

        verify_weights(data)
        import torch

        torch.multiprocessing.set_start_method("spawn", force=True)
        from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
        from megapose.utils.load_model import load_named_model

        from .coarse_cache import install_coarse_cache, precompute_templates

        model_path = object_dir / "model.glb"
        metadata = json.loads((object_dir / "metadata.json").read_text())
        if (
            metadata.get("format") != "mega-hat.object.v1"
            or metadata.get("translation_units") != "m"
        ):
            raise ValueError("Expected an object prepared by mega-hat prepare")
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["prepared_sha256"]:
            raise ValueError("Prepared mesh changed; prepare a fresh object directory")
        self.config = shared_config()
        self.label = "target"
        objects = RigidObjectDataset(
            [RigidObject(label=self.label, mesh_path=model_path, mesh_units="m")]
        )
        self.model = (
            load_named_model(
                "megapose-1.0-RGB-multi-hypothesis", objects, n_workers=4, bsz_images=32
            )
            .cuda()
            .eval()
        )
        # As in the published cached frontend, fixed canonical renders are built
        # once, while each query still computes its own crop and autodepth poses.
        cx, cy = sequence.width / 2, sequence.height / 2
        half = min(sequence.width, sequence.height) * 0.2
        templates = precompute_templates(
            self.model,
            self.label,
            torch.as_tensor(sequence.K).float(),
            torch.tensor([cx - half, cy - half, cx + half, cy + half]).float(),
            (sequence.height, sequence.width),
        )
        install_coarse_cache(self.model, templates)

    def predict(self, image, bbox, K):
        import pandas as pd
        import torch
        from megapose.inference.types import ObservationTensor
        from megapose.utils.tensor_collection import PandasTensorCollection

        box = padded_box(
            bbox, image.shape[1], image.shape[0], self.config["frontend"]["bbox_padding"]
        )
        observation = ObservationTensor.from_numpy(image, None, K).cuda()
        detection = PandasTensorCollection(
            infos=pd.DataFrame([dict(label=self.label, score=1.0, batch_im_id=0, instance_id=0)]),
            bboxes=torch.as_tensor(box, dtype=torch.float32)[None],
        ).cuda()
        with torch.no_grad():
            _, extra = self.model.run_inference_pipeline(
                observation,
                detections=detection,
                n_refiner_iterations=self.config["frontend"]["refiner_iterations"],
                n_pose_hypotheses=self.config["candidate_count"],
            )
        scored = extra["scoring"]["preds"]
        poses = scored.poses.detach().cpu().numpy()
        logits = scored.infos["pose_logit"].values.astype(np.float32)
        count = self.config["candidate_count"]
        if len(poses) == 0:
            raise RuntimeError("MegaPose returned no hypotheses")
        if len(poses) < count:
            poses = np.concatenate([poses, np.repeat(poses[-1:], count - len(poses), axis=0)])
            logits = np.concatenate([logits, np.full(count - len(logits), -1e9, np.float32)])
        return poses[:count], logits[:count]

    def close(self):
        renderers = {
            id(m.renderer): m.renderer for m in (self.model.coarse_model, self.model.refiner_model)
        }
        for renderer in renderers.values():
            renderer.stop()


def padded_box(bbox, width, height, padding):
    box = np.asarray(bbox, dtype=np.float32)
    center = (box[:2] + box[2:]) / 2
    extent = (box[2:] - box[:2]) * (1 + padding) / 2
    return np.clip(
        np.r_[center - extent, center + extent], [0, 0, 0, 0], [width, height, width, height]
    )
