"""Command-line entry points for object preparation and sequence tracking."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import shared_config
from .sequence import Sequence
from .tracker import Tracker


def run(args):
    from .runtime import Worker, read_runtime

    sequence = Sequence(args.sequence)
    validation = sequence.validate()
    runtime = read_runtime(args.runtime)
    object_dir = args.object.resolve()
    for name in ("model.glb", "metadata.json"):
        if not (object_dir / name).is_file():
            raise ValueError(f"Missing {object_dir / name}; run mega-hat prepare first")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(shared_config(), indent=2) + "\n")
    (output / "inputs.json").write_text(
        json.dumps(
            dict(
                sequence=sequence.config,
                validation=validation,
                runtime=runtime,
                object_dir=str(object_dir),
            ),
            indent=2,
        )
        + "\n"
    )
    workers, tracker = [], Tracker(len(sequence))
    poses, motions, frames, candidates, scores = [], [], [], [], []
    try:
        mega = Worker("mega", runtime, sequence.path, object_dir, output)
        workers.append(mega)
        droid = Worker("droid", runtime, sequence.path, object_dir, output)
        workers.append(droid)
        with (output / "poses.jsonl").open("w", buffering=1) as stream:
            for index, path in enumerate(sequence.frames):
                motion = np.asarray(droid.call(index), dtype=float)
                observation = None
                if tracker.anchor_due and sequence.localization(index) is not None:
                    observation = tuple(np.asarray(x) for x in mega.call(index))
                    frames.append(index)
                    candidates.append(observation[0])
                    scores.append(observation[1])
                pose, diagnostics = tracker.step(motion, observation)
                poses.append(pose)
                motions.append(motion)
                stream.write(
                    json.dumps(
                        dict(
                            frame=index,
                            image=path.name,
                            pose=pose.tolist() if diagnostics["valid"] else None,
                            **diagnostics,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                if index % 30 == 0 or index + 1 == len(sequence):
                    print(f"{index + 1}/{len(sequence)} frames, {len(frames)} anchors", flush=True)
        np.savez_compressed(
            output / "poses.npz",
            poses=np.array(poses),
            images=np.array([p.name for p in sequence.frames]),
        )
        np.savez_compressed(
            output / "observations.npz",
            slam=np.array(motions),
            frames=np.array(frames, dtype=int),
            candidates=np.array(candidates),
            scores=np.array(scores),
        )
        valid = int(np.isfinite(poses).all(axis=(1, 2)).sum())
        summary = dict(
            frames=len(poses),
            valid_poses=valid,
            anchor_calls=len(frames),
            pose_convention="object-to-camera",
            translation_units="m",
            future_frames=0,
            config="hat-shared-causal",
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if valid == 0:
            raise RuntimeError(
                "No valid pose was recovered. Inspect localization, object and logs."
            )
        print(json.dumps(summary))
    finally:
        tracker.close()
        for worker in reversed(workers):
            worker.close()


def replay(args):
    with np.load(args.observations, allow_pickle=False) as data:
        motion = data["slam"]
        output_frames = data["output_frames"] if "output_frames" in data else np.arange(len(motion))
        frames, poses, scores = data["frames"], data["candidates"], data["scores"]
    if motion.ndim != 3 or motion.shape[1:] != (4, 4) or len(motion) == 0:
        raise ValueError("slam must have shape (N, 4, 4) with N > 0")
    if (
        frames.ndim != 1
        or not np.issubdtype(frames.dtype, np.integer)
        or np.any(frames < 0)
        or np.any(frames >= len(motion))
        or np.any(np.diff(frames) < 4)
        or poses.shape != (len(frames), 10, 4, 4)
        or scores.shape != (len(frames), 10)
    ):
        raise ValueError("Invalid candidate dimensions or anchor schedule")
    if (
        output_frames.ndim != 1
        or not np.issubdtype(output_frames.dtype, np.integer)
        or np.any(output_frames < 0)
        or np.any(output_frames >= len(motion))
        or np.any(np.diff(output_frames) <= 0)
    ):
        raise ValueError("Invalid output_frames grid")
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}")
    observations = {int(f): (p, s) for f, p, s in zip(frames, poses, scores)}
    tracker = Tracker(len(motion))
    try:
        grid = set(output_frames.tolist())
        result = []
        for i, s in enumerate(motion):
            pose, _ = tracker.step(s, observations.get(i), emit=i in grid)
            if i in grid:
                result.append(pose)
    finally:
        tracker.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # A file handle preserves the requested name instead of appending .npz.
    with args.out.open("xb") as stream:
        np.savez_compressed(stream, poses=np.array(result), frames=output_frames)


def main():
    parser = argparse.ArgumentParser(description="Mega-HAT RGB sequence pose tracking")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="Print the shared causal settings from HAT (arXiv:2609.21597)")
    validate = sub.add_parser("validate", help="Check every RGB frame and localization")
    validate.add_argument("sequence", type=Path)
    prepare = sub.add_parser("prepare", help="Prepare a metric textured CAD object")
    prepare.add_argument("--mesh", type=Path, required=True)
    prepare.add_argument("--units", choices=("m", "cm", "mm"), required=True)
    prepare.add_argument("--out", type=Path, required=True)
    infer = sub.add_parser("run", help="Track an RGB sequence with MegaPose and DROID-SLAM")
    infer.add_argument("--sequence", type=Path, required=True)
    infer.add_argument("--runtime", type=Path, required=True)
    infer.add_argument("--object", type=Path, required=True)
    infer.add_argument("--out", type=Path, required=True)
    cached = sub.add_parser("replay", help="Run the temporal method on saved observations")
    cached.add_argument("observations", type=Path)
    cached.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "config":
            print(json.dumps(shared_config(), indent=2))
        elif args.command == "validate":
            print(json.dumps(Sequence(args.sequence).validate(), indent=2))
        elif args.command == "prepare":
            from .objects import prepare_object

            print(json.dumps(prepare_object(args.mesh, args.units, args.out), indent=2))
        elif args.command == "run":
            run(args)
        elif args.command == "replay":
            replay(args)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"mega-hat: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
