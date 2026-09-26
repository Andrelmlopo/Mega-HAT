"""Causal forward hypothesis scores with the shared Mega-HAT warm-up rule."""

import numpy as np

from .geometry import rotation_distance, valid_pose


class Selector:
    def __init__(self, config):
        self.settings = config["selector"]
        self.count = config["candidate_count"]
        self.scores = None
        self.previous = None
        self.seen = 0

    def step(self, poses, scores, slam):
        poses, scores = np.array(poses, dtype=float), np.array(scores, dtype=float)
        if poses.shape != (self.count, 4, 4) or scores.shape != (self.count,):
            raise ValueError(f"Expected {self.count} candidate poses and scores")
        if not np.isfinite(scores).all():
            raise ValueError("Candidate scores must be finite MegaPose logits")
        if not valid_pose(poses).all():
            raise ValueError("All MegaPose candidate poses must be valid SE(3)")
        rotations = poses[:, :3, :3]
        unary = scores * self.settings["unary_scale"]
        if self.scores is None:
            self.scores = unary.copy()
        else:
            previous_slam, previous_rotations = self.previous
            has_motion = valid_pose(slam) and valid_pose(previous_slam)
            predicted = (
                (slam[:3, :3].T @ previous_slam[:3, :3]) @ previous_rotations
                if has_motion
                else None
            )
            self.scores = np.array(
                [
                    np.max(
                        self.scores
                        - self.settings["motion_weight"]
                        * (
                            np.array([rotation_distance(rotations[k], r) for r in predicted])
                            if has_motion
                            else np.zeros(self.count)
                        )
                    )
                    + unary[k]
                    for k in range(self.count)
                ]
            )
        self.seen += 1
        accepted = self.seen >= self.settings["warmup"]
        index = int(self.scores.argmax() if accepted else scores.argmax())
        self.previous = (slam.copy(), rotations.copy())
        return poses[index].copy(), {
            "accepted": accepted,
            "candidate": index,
            "valid_candidates": self.count,
        }
