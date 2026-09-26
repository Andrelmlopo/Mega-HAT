import json

import numpy as np

from mega_hat.cli import main


def test_config_runs_from_an_unrelated_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["mega-hat", "config"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["anchor_gap"] == 4


def test_replay_preserves_missing_initial_frames_and_refuses_overwrite(tmp_path, monkeypatch):
    motion = np.full((8, 4, 4), np.nan)
    candidates = np.repeat(np.eye(4)[None, None], 10, axis=1)
    candidates[:, :, 2, 3] = 2
    observations = tmp_path / "observations.npz"
    np.savez(
        observations,
        slam=motion,
        frames=np.array([1]),
        candidates=candidates,
        scores=np.ones((1, 10)),
    )
    output = tmp_path / "exact-name.npz"
    monkeypatch.setattr("sys.argv", ["mega-hat", "replay", str(observations), "--out", str(output)])
    assert main() == 0
    with np.load(output) as archive:
        assert np.isnan(archive["poses"][0]).all()
        assert np.isfinite(archive["poses"][1:]).all()
    original = output.read_bytes()
    assert main() == 1
    assert output.read_bytes() == original


def test_replay_uses_declared_output_grid_and_preserves_frame_ids(tmp_path, monkeypatch):
    motion = np.repeat(np.eye(4)[None], 12, axis=0)
    candidates = np.repeat(np.eye(4)[None, None], 10, axis=1)
    candidates[:, :, 2, 3] = 2
    path, output = tmp_path / "observations.npz", tmp_path / "result.npz"
    np.savez(
        path,
        slam=motion,
        frames=np.array([4]),
        candidates=candidates,
        scores=np.zeros((1, 10)),
        output_frames=np.array([2, 4, 9]),
    )
    monkeypatch.setattr("sys.argv", ["mega-hat", "replay", str(path), "--out", str(output)])
    assert main() == 0
    with np.load(output) as data:
        np.testing.assert_array_equal(data["frames"], [2, 4, 9])
        assert data["poses"].shape == (3, 4, 4)
        assert np.isnan(data["poses"][0]).all()
        assert np.isfinite(data["poses"][1:]).all()
