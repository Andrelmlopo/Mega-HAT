import json

import pytest

from mega_hat.weights import verify_weights


def test_changed_pretrained_configuration_is_rejected_before_loading(tmp_path, monkeypatch):
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"megapose-models/coarse/config.yaml": "0" * 64}))
    monkeypatch.setattr("mega_hat.weights.files", lambda name: tmp_path)
    path = tmp_path / "megapose-models/coarse/config.yaml"
    with pytest.raises(ValueError, match="Missing"):
        verify_weights(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("unexpected: configuration")
    with pytest.raises(ValueError, match="differs"):
        verify_weights(tmp_path)
