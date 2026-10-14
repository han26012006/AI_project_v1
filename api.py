# -*- coding: utf-8 -*-
"""API của Trợ lý mã HS. Chạy: uvicorn api:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from hs_assistant.audit_log import AuditLog
from hs_assistant.core import HSAssistant

ART = os.environ.get("HS_ART", "artifacts")
LOG = os.environ.get("HS_LOG", "logs/decisions.sqlite")
API_KEY = os.environ.get("HS_API_KEY")          # đặt khi chạy ngoài máy cá nhân
state: dict = {}


def check_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Thiếu hoặc sai khóa API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["a"] = HSAssistant(ART)
    state["log"] = AuditLog(LOG)
    yield
    state.clear()


app = FastAPI(title="Trợ lý phân loại mã HS – Nhóm 8", version="0.1.0", lifespan=lifespan,
              description="Đề xuất mã HS 8 số có căn cứ. Kết quả chỉ để tham khảo; người khai quyết định.")


class ClassifyIn(BaseModel):
    mo_ta: str = Field(..., min_length=20, max_length=8000, description="Tên hàng, thành phần, cấu tạo, công dụng, thông số")
    k: int = Field(3, ge=1, le=10)


class AuditItem(BaseModel):
    id: str | None = None
    mo_ta: str = Field(..., min_length=3)
    ma_hs: str = Field(..., min_length=4)


class AuditIn(BaseModel):
    items: list[AuditItem] = Field(..., min_length=2, max_length=5000)
    nguong: float = Field(0.85, ge=0.5, le=0.99)


class EstimateIn(BaseModel):
    ma: list[str] = Field(..., min_length=1, max_length=10)
    tri_gia: float = Field(..., gt=0, description="Trị giá tính thuế (VND)")
    ma_chon: str | None = None
    so_ngay_cham: int = Field(90, ge=0, le=3650)


class DecisionIn(BaseModel):
    ho_so_id: str
    ma_chon: str = Field(..., pattern=r"^\d{8}$")
    ly_do: str = Field(..., min_length=5, max_length=2000)
    nguoi_khai: str | None = None


@app.get("/health")
def health():
    a = state["a"]
    return {"status": "ok", "phien_ban_mo_hinh": a.version, "so_dong_nhan": int(len(a.labels)), "so_tien_le": int(len(a.prec)),
            "vector": a.backend}


@app.post("/classify", dependencies=[Depends(check_key)])
def classify(inp: ClassifyIn):
    res = state["a"].classify(inp.mo_ta, k=inp.k)
    state["log"].log_classification(res)
    return res


@app.post("/audit", dependencies=[Depends(check_key)])
def audit(inp: AuditIn):
    df = pd.DataFrame([i.model_dump() for i in inp.items])
    out = state["a"].audit_catalog(df, text_col="mo_ta", code_col="ma_hs", id_col="id", thr=inp.nguong)
    return {"so_mat_hang": int(len(df)), "so_cap_lech": int(len(out)), "cap_lech": out.to_dict(orient="records")}


@app.post("/estimate", dependencies=[Depends(check_key)])
def estimate(inp: EstimateIn):
    a = state["a"]
    bad = [c for c in inp.ma if c not in a.lab_pos]
    if bad:
        raise HTTPException(status_code=422, detail=f"Mã không có trong Danh mục: {bad}")
    return a.uoc_tinh_thue(inp.ma, inp.tri_gia, inp.ma_chon, inp.so_ngay_cham)


@app.post("/decision", dependencies=[Depends(check_key)])
def decision(inp: DecisionIn):
    r = state["log"].log_decision(inp.ho_so_id, inp.ma_chon, inp.ly_do, inp.nguoi_khai)
    if r is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ phân loại với mã này")
    return r
