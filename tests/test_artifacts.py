import pytest
# Kiểm thử trên artifact thật: lược đồ dữ liệu, rò rỉ nhãn, khớp với notebook, trích dẫn
import json
import os
import re

import numpy as np
import pandas as pd

from conftest import needs_art


@needs_art
def test_label_space_schema(art):
    lab = pd.read_parquet(os.path.join(art, "label_space.parquet"))
    assert lab.ma8.str.fullmatch(r"\d{8}").all() and lab.ma8.is_unique
    assert 11000 <= len(lab) <= 12500
    assert (lab.duong_dan.str.len() > 0).all()


@needs_art
def test_no_label_leakage(art):
    v = pd.read_parquet(os.path.join(art, "precedents.parquet"))
    digits = v.input_text.str.replace(r"\D", "", regex=True)
    leak = [str(c).zfill(8) in d for c, d in zip(v.hs8, digits)]
    assert np.mean(leak) == 0.0
    assert set(v.split) >= {"train", "valid", "test"}


@needs_art
def test_core_reproduces_notebook(art, assistant_train_only):
    pr = json.load(open(os.path.join(art, "predictions.json"), encoding="utf-8"))
    v = assistant_train_only.prec_all
    same, top3 = [], []
    for qi, ref, g in zip(pr["idx_test"], pr["full_test"], pr["gold_test"]):
        _, codes, _, _, _ = assistant_train_only.rank(v.input_text.iloc[qi])
        same.append(codes[0] == ref["codes"][0])
        top3.append(g in codes[:3])
    assert np.mean(same) >= 0.98, f"Chỉ {np.mean(same):.3f} dự đoán top-1 khớp notebook"
    m = json.load(open(os.path.join(art, "metrics.json"), encoding="utf-8"))
    ref_top3 = [r for r in m["ket_qua"] if r["mô hình"].startswith("Mô hình đầy đủ")][0]["top3 8 số"]
    assert np.mean(top3) >= ref_top3 - 0.02


@needs_art
def test_dossier_citations_and_set(art, assistant_train_only):
    v = assistant_train_only.prec_all
    text = v.loc[v.split == "test", "input_text"].iloc[0]
    r = assistant_train_only.classify(text)
    top = r["ma_de_xuat"][0]["ma8"]
    assert all(e["ma"][:4] == top[:4] for e in r["can_cu_tien_le"])
    assert r["nhom_de_xuat"] and all(len(n["nhom"]) == 4 for n in r["nhom_de_xuat"])
    assert len(r["tap_nhom_tin_cay"]) <= 5
    assert all(d["ma8"][:4] == r["nhom_de_xuat"][0]["nhom"] for d in r["dong_8_so_trong_nhom_1"])
    assert isinstance(r["hau_qua_giua_cac_ma"], list) and all("chinh_sach" in x for x in r["hau_qua_giua_cac_ma"])
    assert all(set(x) >= {"chuong", "dieu_khoan", "thuoc_phan_loai_tru"} for x in r["chu_giai_chuong_can_doc"])
    assert "QT2" in r["gioi_han"] and "3 năm" in r["hieu_luc_tien_le"]
    assert r["rui_ro"]["muc"] in {"Thấp", "Trung bình", "Cao"} and isinstance(r["rui_ro"]["ly_do"], list)
    assert all({"cap", "cau_hoi", "lua_chon"} <= set(q) for q in r["cau_hoi_phan_biet"])
    codes = [m["ma8"] for m in r["ma_de_xuat"]]
    est = assistant_train_only.uoc_tinh_thue(codes, 1e9, codes[0])
    assert est["ma_chon"] == codes[0] and est["thieu_toi_da_neu_ma_dung_la_ma_khac"] >= 0
    assert abs(est["phat_neu_hai_quan_phat_hien_20pct"] - 0.2 * est["thieu_toi_da_neu_ma_dung_la_ma_khac"]) < 1e-6
    assert re.fullmatch(r"\d{8}", top)


@needs_art
def test_final_model_steps(tmp_path, art):
    # dựng thư mục artifact tạm có final_model.json với bước M3 (nguyên mẫu nhóm), kiểm tra ứng dụng áp đúng mô hình cuối
    for f in os.listdir(art):
        if os.path.isfile(os.path.join(art, f)):
            os.symlink(os.path.join(art, f), tmp_path / f)
    (tmp_path / "mo_hinh_v4").mkdir()
    json.dump({"ten": "M0 + M3", "base": "M0", "steps": [{"ten": "M3", "loai": "nguyen_mau", "beta": 0.6},
                                                          {"ten": "M5a", "loai": "llm", "beta": 0.3, "provider": "x", "model": "y"}],
               "T_head": 0.15, "tau_head": 0.35, "k_head": 10}, open(tmp_path / "mo_hinh_v4" / "final_model.json", "w"))
    from hs_assistant.core import HSAssistant
    a = HSAssistant(str(tmp_path))
    assert [s["ten"] for s in a.steps_active] == ["M3"] and any("M5a" in x for x in a.steps_skipped)   # LLM tắt mặc định
    r = a.classify("Thiết bị thổi nóng cầm tay, vỏ nhựa, gia nhiệt điện trở, 220V 1200W, dùng để làm nóng")
    assert r["mo_hinh"] == "M0 + M3" and r["nhom_de_xuat"] and sum(n["xac_suat"] for n in r["nhom_de_xuat"]) <= 1.0 + 1e-6
    assert all(d["ma8"][:4] == r["nhom_de_xuat"][0]["nhom"] for d in r["dong_8_so_trong_nhom_1"])


@needs_art
def test_final_matches_predictions_final(art):
    # ứng dụng phải tái lập đúng dự đoán của mô hình cuối Bước 2 (cùng kho tiền lệ train)
    pf = os.path.join(art, "mo_hinh_v4", "predictions_final.json")
    if not os.path.exists(pf):
        pytest.skip("chưa có predictions_final.json (chạy Bước 2)")
    P = json.load(open(pf, encoding="utf-8"))
    if any(s.get("loai") == "llm" for s in P.get("steps", [])):
        pytest.skip("mô hình cuối có bước LLM: không tái lập offline")
    from hs_assistant.core import HSAssistant
    a = HSAssistant(art, precedent_splits=("train",))
    if a.steps_skipped:
        pytest.skip(f"bước không chạy được trên máy này: {a.steps_skipped}")
    same = [a.classify(a.prec_all.input_text.iloc[r["qi"]])["nhom_de_xuat"][0]["nhom"] == r["heads"][0] for r in P["full_test"][:40]]
    assert np.mean(same) >= 0.95
