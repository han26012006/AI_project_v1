# -*- coding: utf-8 -*-
"""Nhật ký phân loại và quyết định của người khai (SQLite). Lưu đầu vào đã ẩn danh để truy vết căn cứ."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

import pandas as pd


class AuditLog:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path, self._lock = path, threading.Lock()
        with self._lock, sqlite3.connect(self.path) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS phan_loai (
                ho_so_id TEXT PRIMARY KEY, ts REAL, phien_ban TEXT, dau_vao TEXT, top1 TEXT, xac_suat_top1 REAL,
                kich_thuoc_tap INTEGER, do_tin_cay_cao INTEGER, do_tuong_dong_max REAL, thoi_gian_ms INTEGER,
                ket_qua_json TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS quyet_dinh (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ho_so_id TEXT, ts REAL, ma_chon TEXT, ly_do TEXT,
                nguoi_khai TEXT, trung_top1 INTEGER)""")

    def log_classification(self, res: dict) -> None:
        top = res["ma_de_xuat"][0] if res.get("ma_de_xuat") else {"ma8": None, "xac_suat": None}
        with self._lock, sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO phan_loai VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (res["ho_so_id"], time.time(), res.get("phien_ban_mo_hinh"), res.get("dau_vao_an_danh"),
                       top["ma8"], top["xac_suat"], len(res.get("tap_nhom_tin_cay", [])), int(bool(res.get("do_tin_cay_cao"))),
                       res.get("do_tuong_dong_tien_le_max"), res.get("thoi_gian_ms"),
                       json.dumps(res, ensure_ascii=False, default=str)))

    def log_decision(self, ho_so_id: str, ma_chon: str, ly_do: str, nguoi_khai: str | None = None) -> dict | None:
        with self._lock, sqlite3.connect(self.path) as c:
            row = c.execute("SELECT top1 FROM phan_loai WHERE ho_so_id = ?", (ho_so_id,)).fetchone()
            if row is None:
                return None
            trung = int(row[0] == ma_chon)
            c.execute("INSERT INTO quyet_dinh (ho_so_id, ts, ma_chon, ly_do, nguoi_khai, trung_top1) VALUES (?,?,?,?,?,?)",
                      (ho_so_id, time.time(), ma_chon, ly_do, nguoi_khai, trung))
        return {"ho_so_id": ho_so_id, "ma_chon": ma_chon, "trung_ma_de_xuat_so_1": bool(trung), "da_luu": True}

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with sqlite3.connect(self.path) as c:
            return (pd.read_sql("SELECT * FROM phan_loai", c), pd.read_sql("SELECT * FROM quyet_dinh", c))
