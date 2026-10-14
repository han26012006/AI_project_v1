# -*- coding: utf-8 -*-
"""Giám sát vận hành: tỷ lệ chấp nhận, tỷ lệ sửa mã, độ trễ, drift dữ liệu, hiệu chỉnh lại conformal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import conformal_qhat


def summary(cls: pd.DataFrame, dec: pd.DataFrame) -> dict:
    out = {"so_lan_phan_loai": int(len(cls)), "so_quyet_dinh": int(len(dec))}
    if len(cls):
        out["ty_le_tin_cay_cao"] = round(float(cls.do_tin_cay_cao.mean()), 3)
        out["ty_le_chua_du_can_cu"] = round(float(1 - cls.do_tin_cay_cao.mean()), 3)
        out["do_tre_p95_ms"] = int(np.percentile(cls.thoi_gian_ms.dropna(), 95)) if cls.thoi_gian_ms.notna().any() else None
        out["do_tuong_dong_tien_le_tb"] = round(float(cls.do_tuong_dong_max.mean()), 3)
    if len(dec):
        out["ty_le_chon_trung_ma_so_1"] = round(float(dec.trung_top1.mean()), 3)
        out["ty_le_sua_ma"] = round(float(1 - dec.trung_top1.mean()), 3)
    return out


def psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    """Population Stability Index giữa hai phân bố rời rạc (ví dụ phân bố chương)."""
    cats = sorted(set(expected) | set(actual))
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).values + eps
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).values + eps
    return float(np.sum((a - e) * np.log(a / e)))


def chapter_drift(train_codes, recent_codes, threshold: float = 0.2) -> dict:
    val = psi(pd.Series(list(train_codes)).str[:2], pd.Series(list(recent_codes)).str[:2])
    return {"psi_chuong": round(val, 3), "canh_bao": val > threshold, "nguong": threshold}


def similarity_drift(baseline_mean: float, recent_max_sims, drop: float = 0.15) -> dict:
    r = float(np.mean(recent_max_sims)) if len(recent_max_sims) else float("nan")
    rel = (baseline_mean - r) / baseline_mean if baseline_mean else float("nan")
    return {"tuong_dong_goc": round(baseline_mean, 3), "tuong_dong_gan_day": round(r, 3),
            "giam_tuong_doi": round(rel, 3), "canh_bao": bool(rel > drop)}


def recalibrate_qhat(ranked_codes: list[list[str]], ranked_probs: list[list[float]], gold: list[str], alpha: float) -> float:
    """Tính lại ngưỡng conformal trên dữ liệu có nhãn gần nhất (ví dụ kết luận chính thức mới)."""
    s = []
    for c, p, g in zip(ranked_codes, ranked_probs, gold):
        s.append(1 - p[c.index(g)] if g in c else 1.0)
    return conformal_qhat(s, alpha)
