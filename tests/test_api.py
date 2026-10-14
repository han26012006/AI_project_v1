# Kiểm thử API bằng TestClient (cần artifact)
import os

import pytest

from conftest import needs_art


@needs_art
def test_api_flow(tmp_path, art):
    os.environ["HS_ART"], os.environ["HS_LOG"] = art, str(tmp_path / "log.sqlite")
    from fastapi.testclient import TestClient
    import importlib
    import api
    importlib.reload(api)
    with TestClient(api.app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["so_dong_nhan"] > 11000
        assert c.post("/classify", json={"mo_ta": "ngắn"}).status_code == 422
        r = c.post("/classify", json={"mo_ta": "Thiết bị thổi nóng cầm tay, vỏ nhựa, gia nhiệt điện trở, 220V 1200W, dùng để làm nóng"})
        assert r.status_code == 200
        res = r.json()
        top = res["ma_de_xuat"][0]["ma8"]
        d = c.post("/decision", json={"ho_so_id": res["ho_so_id"], "ma_chon": top, "ly_do": "Đối chiếu căn cứ, đồng ý"}).json()
        assert d["trung_ma_de_xuat_so_1"] is True
        assert c.post("/decision", json={"ho_so_id": "khong-ton-tai", "ma_chon": top, "ly_do": "thử nghiệm"}).status_code == 404
        e = c.post("/estimate", json={"ma": [m["ma8"] for m in res["ma_de_xuat"]], "tri_gia": 2e9}).json()
        assert "bang" in e and len(e["bang"]) == len(res["ma_de_xuat"])
        assert c.post("/estimate", json={"ma": ["00000000"], "tri_gia": 1e6}).status_code == 422
        items = [{"id": "A", "mo_ta": "máy sấy tóc cầm tay 1200W", "ma_hs": "85163100"},
                 {"id": "B", "mo_ta": "máy sấy tóc cầm tay 1200W", "ma_hs": "85163200"},
                 {"id": "C", "mo_ta": "hạt nhựa PET nguyên sinh", "ma_hs": "39076100"}]
        a = c.post("/audit", json={"items": items, "nguong": 0.85}).json()
        assert a["so_cap_lech"] >= 1 and a["cap_lech"][0]["id_A"] in ("A", "B")


@needs_art
def test_api_key_enforced(tmp_path, art):
    os.environ["HS_ART"], os.environ["HS_LOG"], os.environ["HS_API_KEY"] = art, str(tmp_path / "log2.sqlite"), "khoa-thu"
    from fastapi.testclient import TestClient
    import importlib
    import api
    importlib.reload(api)
    try:
        with TestClient(api.app) as c:
            body = {"mo_ta": "Thiết bị thổi nóng cầm tay, vỏ nhựa, gia nhiệt điện trở, 220V 1200W"}
            assert c.post("/classify", json=body).status_code == 401
            assert c.post("/classify", json=body, headers={"X-API-Key": "khoa-thu"}).status_code == 200
    finally:
        os.environ.pop("HS_API_KEY", None)
        importlib.reload(api)
