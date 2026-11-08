def test_schemas_installed():
    # fails if `pip install /schemas` did not work inside the image
    import mica_schemas  # noqa: F401


def test_health():
    from fastapi.testclient import TestClient

    from src.main import app

    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
