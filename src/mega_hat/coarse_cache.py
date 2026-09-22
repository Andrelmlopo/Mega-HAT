# Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted cached execution for Mega-HAT. SPDX-License-Identifier: Apache-2.0
# See licenses/MegaPose.txt.
"""
Template cache for MegaPose coarse classifier.

MegaPose's coarse path re-renders 576 SO(3) templates per call. The renders
depend on (R, t, K_crop) where R is fixed by the grid but t/K_crop vary with
the input bbox via autodepth. However, because (t, K_crop) are designed
together to produce a canonical object-in-crop layout, the rendered appearance
is approximately bbox-invariant.

This module precomputes the 576 renders once using a canonical bbox, then
patches the pose_estimator to skip rendering on subsequent calls.
"""

from __future__ import annotations

import time

import pandas as pd
import torch
from megapose.lib3d.cosypose_ops import TCO_init_from_boxes_autodepth_with_R
from megapose.lib3d.transform_ops import normalize_T
from megapose.utils.tensor_collection import PandasTensorCollection
from torch.utils.data import DataLoader, TensorDataset


@torch.no_grad()
def precompute_templates(
    pose_estimator,
    label: str,
    image_K: torch.Tensor,  # [3, 3] camera intrinsics of input images
    canonical_bbox: torch.Tensor,  # [4]  (x1, y1, x2, y2) — object-filling bbox
    image_shape: tuple,  # (H, W) of input images (for dummy crop tensor)
) -> torch.Tensor:
    """Render all 576 SO(3) templates using a canonical bbox, once per object.

    Returns a tensor of shape [M, C, H_render, W_render] on CUDA.
    """
    import time as _time

    t0 = _time.time()

    def _log(msg):
        print(f"  [precompute +{_time.time() - t0:5.1f}s] {msg}", flush=True)

    coarse_model = pose_estimator.coarse_model
    SO3_grid = pose_estimator._SO3_grid  # [M, 3, 3] on CUDA
    M = SO3_grid.shape[0]
    device = SO3_grid.device
    _log(f"M={M}, device={device}")

    labels = [label] * M
    meshes = coarse_model.mesh_db.select(labels)
    points = meshes.points  # [M, N, 3]
    _log(f"mesh points ready, shape={tuple(points.shape)}")

    bboxes = canonical_bbox.to(device).unsqueeze(0).repeat(M, 1)  # [M, 4]
    Ks = image_K.to(device).unsqueeze(0).repeat(M, 1, 1)  # [M, 3, 3]

    TCO = TCO_init_from_boxes_autodepth_with_R(bboxes, points, Ks, SO3_grid)
    TCO = normalize_T(TCO).detach()
    tCR = TCO[..., :3, -1]
    _log(f"TCO ready, first z={TCO[0, 2, 3].item():.3f}")

    H, W = image_shape
    C_in = 4 if coarse_model.input_depth else 3

    bsz = pose_estimator.bsz_images
    n_chunks = (M + bsz - 1) // bsz
    _log(f"rendering {M} templates in {n_chunks} chunks of bsz={bsz}")
    all_renders = []
    for c_idx, i in enumerate(range(0, M, bsz)):
        hi = min(i + bsz, M)
        sl = slice(i, hi)
        chunk = hi - i
        # crop_inputs needs dummy images only to know H,W — pixels aren't used for rendering
        dummy = torch.zeros(chunk, C_in, H, W, device=device)
        _log(f"chunk {c_idx + 1}/{n_chunks}: crop_inputs (chunk={chunk})")
        _, K_crop, _, _ = coarse_model.crop_inputs(dummy, Ks[sl], TCO[sl], tCR[sl], labels[i:hi])
        TCO_V = TCO[sl].unsqueeze(1)  # [chunk, 1, 4, 4]
        KV_crop = K_crop.unsqueeze(1)
        _log(f"chunk {c_idx + 1}/{n_chunks}: render_images_multiview")
        renders = coarse_model.render_images_multiview(labels[i:hi], TCO_V, KV_crop)
        _log(f"chunk {c_idx + 1}/{n_chunks}: done, renders={tuple(renders.shape)}")
        all_renders.append(renders)

    templates = torch.cat(all_renders, dim=0)  # [M, C_render, H_r, W_r]
    _log(f"all templates concatenated, shape={tuple(templates.shape)}")
    return templates


@torch.no_grad()
def _forward_coarse_model_cached(
    self,  # bound to pose_estimator via __get__
    observation,
    detections,
    cuda_timer: bool = False,
    return_debug_data: bool = False,
):
    """Drop-in replacement for PoseEstimator.forward_coarse_model that uses
    precomputed templates instead of calling the renderer per frame.

    The cache tensor is expected at self._template_cache (installed by
    install_coarse_cache)."""
    templates = getattr(self, "_template_cache", None)
    if templates is None:
        raise RuntimeError("No coarse template cache installed on pose_estimator")

    start_time = time.time()
    coarse_model = self.coarse_model
    SO3_grid = self._SO3_grid
    bsz_images = self.bsz_images
    B = len(detections)
    M = SO3_grid.shape[0]
    assert templates.shape[0] == M, (
        f"template cache has {templates.shape[0]} entries but SO3 grid has {M}"
    )

    df = detections.infos
    df_concat = []
    for tc_idx, row in df.iterrows():
        df_tmp = pd.DataFrame([row] * M)
        df_tmp["hypothesis_id"] = list(range(M))
        df_tmp["bbox_id"] = tc_idx
        df_concat.append(df_tmp)
    df_hypotheses = pd.concat(df_concat)
    df_hypotheses.reset_index()

    ids = torch.arange(len(df_hypotheses))
    dl = DataLoader(TensorDataset(ids), batch_size=bsz_images)
    device = observation.images.device

    logits_list, scores_list, bboxes_list, TCO_init = [], [], [], []
    model_time = 0.0

    for (batch_ids,) in dl:
        df_ = df_hypotheses.iloc[batch_ids.cpu().numpy()]
        batch_im_ids_ = torch.as_tensor(df_["batch_im_id"].values, device=device)
        m_idx = torch.as_tensor(df_["hypothesis_id"].values, device=device)
        labels_ = df_["label"].tolist()
        bbox_ids_ = torch.as_tensor(df_["bbox_id"].values, device=device)

        images_ = observation.images[batch_im_ids_]
        K_ = observation.K[batch_im_ids_]
        bboxes_ = detections.bboxes[bbox_ids_]
        meshes_ = coarse_model.mesh_db.select(labels_)
        points_ = meshes_.points
        SO3_grid_ = SO3_grid[m_idx]

        TCO_init_ = TCO_init_from_boxes_autodepth_with_R(
            bboxes_,
            points_,
            K_,
            SO3_grid_,
        )

        # ── Replicate forward_coarse body, but skip render_images_multiview ──
        if not coarse_model.input_depth:
            images_rgb = images_[:, coarse_model.input_rgb_dims]
        else:
            images_rgb = images_

        TCO_n = normalize_T(TCO_init_).detach()
        tCR = TCO_n[..., :3, -1]
        images_crop, K_crop, _, _ = coarse_model.crop_inputs(images_rgb, K_, TCO_n, tCR, labels_)

        renders = templates[m_idx]  # [b, C_render, H, W] — cached

        images_crop, renders = coarse_model.normalize_images(images_crop, renders, tCR)
        x = torch.cat((images_crop, renders), dim=1)

        out = coarse_model.forward_coarse_tensor(x, cuda_timer=cuda_timer)
        model_time += out["time"]

        logits_list.append(out["logits"])
        scores_list.append(out["scores"])
        bboxes_list.append(bboxes_)
        TCO_init.append(TCO_init_)

    logits = torch.cat(logits_list).reshape([B, M])
    scores = torch.cat(scores_list).reshape([B, M])
    bboxes = torch.cat(bboxes_list, dim=0)
    TCO = torch.cat(TCO_init)
    TCO_reshape = TCO.reshape([B, M, 4, 4])

    df_hypotheses["coarse_logit"] = logits.flatten().cpu().numpy()
    df_hypotheses["coarse_score"] = scores.flatten().cpu().numpy()
    elapsed = time.time() - start_time

    extra_data = {
        "render_time": 0.0,
        "model_time": model_time,
        "time": elapsed,
        "logits": logits,
        "scores": scores,
        "TCO": TCO_reshape,
        "debug": dict(),
        "n_batches": len(dl),
        "timing_str": f"time: {elapsed:.2f} (cached)",
    }
    data_TCO = PandasTensorCollection(df_hypotheses, poses=TCO, bboxes=bboxes)
    return data_TCO, extra_data


def install_coarse_cache(pose_estimator, templates: torch.Tensor) -> None:
    """Monkey-patch pose_estimator.forward_coarse_model to use cached templates.

    After this, calls to run_inference_pipeline(..., coarse_estimates=None) use
    the cache instead of re-rendering 576 templates per frame.
    """
    pose_estimator._template_cache = templates
    # Bind the function as a method on this specific instance only.
    pose_estimator.forward_coarse_model = _forward_coarse_model_cached.__get__(
        pose_estimator, type(pose_estimator)
    )
