import json

import httpx

from secrag.api import main as api_main


async def _get(path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=api_main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get(path)


async def test_home_serves_chat_page():
    resp = await _get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Ask the" in resp.text


async def test_evals_page_served():
    resp = await _get("/evals")
    assert resp.status_code == 200
    assert "Evaluation Record" in resp.text


async def test_evals_data_sorted_and_stripped(tmp_path, monkeypatch):
    old = {
        "label": "old",
        "created_at": "2026-07-01T00:00:00+00:00",
        "aggregate": {"recall@5": 0.4},
        "per_question": [{"id": "x"}],
    }
    new = {
        "label": "new",
        "created_at": "2026-07-09T00:00:00+00:00",
        "aggregate": {"recall@5": 0.6},
        "per_question": [{"id": "y"}],
    }
    (tmp_path / "b.json").write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "a.json").write_text(json.dumps(new), encoding="utf-8")
    monkeypatch.setattr(api_main, "RESULTS_DIR", tmp_path)

    resp = await _get("/evals/data")
    assert resp.status_code == 200
    runs = resp.json()
    assert [r["label"] for r in runs] == ["old", "new"]
    assert all("per_question" not in r for r in runs)
