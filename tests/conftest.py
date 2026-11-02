"""Shared pytest fixtures.

The API tests swap the RT-DETR singleton for a tiny fake so they run without
GPU, ultralytics weights, or network downloads.
"""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


sys.path.insert(0, str(Path(__file__).resolve().parent))
from aq_fakes import FakeBox, FakeModel, FakeResult  # noqa: E402,F401


@pytest.fixture
def make_box():
    return FakeBox


@pytest.fixture
def fake_model(monkeypatch):
    """Install a FakeModel as the API's cached detector and return it."""
    import app.main as main

    model = FakeModel()
    monkeypatch.setattr(main, "_model", model)
    return model


@pytest.fixture
def open_access(monkeypatch):
    """No API keys and no rate limit, so endpoint tests don't depend on the environment."""
    import app.main as main
    from app.security import AccessControl

    ac = AccessControl([], None)
    monkeypatch.setattr(main, "access", ac)
    return ac


@pytest.fixture
def mission_store(monkeypatch, tmp_path):
    import app.missions_api as mapi
    from app.store import MissionStore

    s = MissionStore(tmp_path / "missions")
    monkeypatch.setattr(mapi, "store", s)
    return s


@pytest.fixture
def client(fake_model, open_access, mission_store):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


@pytest.fixture
def jpeg_bytes():
    img = Image.new("RGB", (64, 48), color=(20, 60, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
