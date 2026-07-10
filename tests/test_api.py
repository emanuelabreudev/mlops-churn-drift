import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

from churn_mlops.models.wrapper import ChurnModel
from churn_mlops.serving.app import app


@pytest.fixture(scope="module")
def client(tmp_path_factory, clean_df, schema, monkeypatch_module):
    model = ChurnModel(
        estimator=LGBMClassifier(n_estimators=30, random_state=0, verbose=-1),
        schema=schema,
        version="test-1",
        trained_at=ChurnModel.now(),
    )
    model.estimator.fit(model.prepare(clean_df), clean_df["Churn"])
    model_dir = tmp_path_factory.mktemp("model")
    model.save(model_dir)
    monkeypatch_module.setenv("MODEL_DIR", str(model_dir))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _payload(clean_df, n=3):
    rows = clean_df.drop(columns=["Churn"]).head(n).to_dict(orient="records")
    return {"customers": rows}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_predict_batch(client, clean_df):
    resp = client.post("/predict", json=_payload(clean_df, n=5))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 5
    assert body["model_version"] == "test-1"
    for p in body["predictions"]:
        assert 0.0 <= p["churn_probability"] <= 1.0
        assert isinstance(p["churn_predicted"], bool)


def test_predict_rejects_missing_field(client, clean_df):
    bad = _payload(clean_df, n=1)
    del bad["customers"][0]["Contract"]
    assert client.post("/predict", json=bad).status_code == 422


def test_model_info_and_metrics(client):
    info = client.get("/model/info").json()
    assert info["model_version"] == "test-1"
    assert len(info["features"]) == 19

    metrics = client.get("/metrics").text
    assert "churn_api_requests_total" in metrics
    assert 'churn_model_info{version="test-1"} 1' in metrics


def test_reload(client):
    resp = client.post("/reload")
    assert resp.status_code == 200
    assert resp.json()["model_version"] == "test-1"


def test_wrapper_prepare_rejects_missing_columns(clean_df, schema):
    model = ChurnModel(estimator=None, schema=schema)
    with pytest.raises(ValueError, match="features ausentes"):
        model.prepare(pd.DataFrame({"tenure": [1.0]}))
