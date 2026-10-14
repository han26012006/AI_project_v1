# -*- coding: utf-8 -*-
"""Xuất phiếu phân loại (PDF) và kết quả rà soát danh mục (Excel)."""
from __future__ import annotations

import io
import os
import time
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def _register_fonts() -> str:
    if "DejaVu" in pdfmetrics.getRegisteredFontNames():
        return "DejaVu"
    for d in (_FONT_DIR, "/usr/share/fonts/truetype/dejavu"):
        reg, bold = os.path.join(d, "DejaVuSans.ttf"), os.path.join(d, "DejaVuSans-Bold.ttf")
        if os.path.exists(reg) and os.path.exists(bold):
            pdfmetrics.registerFont(TTFont("DejaVu", reg))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold))
            return "DejaVu"
    raise FileNotFoundError("Thiếu font DejaVuSans.ttf (cần để hiển thị tiếng Việt trong PDF)")


def dossier_pdf(res: dict, decision: dict | None = None) -> bytes:
    f = _register_fonts()
    st = {"h1": ParagraphStyle("h1", fontName=f + "-Bold", fontSize=15, leading=19, spaceAfter=6, textColor=colors.HexColor("#102F78")),
          "h2": ParagraphStyle("h2", fontName=f + "-Bold", fontSize=11, leading=14, spaceBefore=8, spaceAfter=4),
          "p": ParagraphStyle("p", fontName=f, fontSize=9.5, leading=13),
          "small": ParagraphStyle("s", fontName=f, fontSize=8.5, leading=11, textColor=colors.HexColor("#4C5868"))}
    P = lambda t, s="p": Paragraph(escape(str(t)), st[s])
    def table(rows, widths):
        t = Table([[P(c) for c in r] for r in rows], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EFFD")),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9CCEB")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t
    s = [P("Phiếu phân loại mã HS", "h1"),
         P(f"Mã hồ sơ: {res.get('ho_so_id')}  |  Phiên bản mô hình: {res.get('phien_ban_mo_hinh')}  |  "
           f"Thời điểm: {time.strftime('%d/%m/%Y %H:%M')}", "small"),
         P("1. Mô tả hàng hóa (đã ẩn danh)", "h2"), P(res.get("dau_vao_an_danh", "")[:2500]),
         P("2. Mã đề xuất", "h2"),
         table([["Mã 8 số", "Đường dẫn mô tả (3 cấp cuối)", "Xác suất", "NK ưu đãi"]] +
               [[m["ma8"], m["mo_ta"], f"{m['xac_suat']:.3f}", m.get("nk_uu_dai", "")] for m in res.get("ma_de_xuat", [])],
               [24 * mm, 100 * mm, 20 * mm, 22 * mm]),
         Spacer(1, 4),
         P("Nhóm 4 số đề xuất: " + "; ".join(f"{n['nhom']} ({n['xac_suat']:.2f})" for n in res.get("nhom_de_xuat", []))),
         P(f"Tập nhóm có mức tin cậy {res.get('muc_tin_cay_nhom')}: {', '.join(res.get('tap_nhom_tin_cay') or []) or 'chưa xác định được'}"),
         P("3. Căn cứ: tiền lệ cùng nhóm với mã đề xuất", "h2")]
    ev = res.get("can_cu_tien_le") or []
    s.append(table([["Số hiệu", "Năm", "Mã kết luận", "Độ tương đồng", "Trích mô tả"]] +
                   [[e.get("so_hieu"), e.get("nam"), e.get("ma"), e.get("do_tuong_dong"), e.get("trich")] for e in ev],
                   [18 * mm, 14 * mm, 22 * mm, 22 * mm, 90 * mm]) if ev else P("Không có tiền lệ cùng nhóm trong kho tiền lệ."))
    hq = res.get("hau_qua_giua_cac_ma") or []
    if hq:
        s += [P("Hệ quả khác nhau giữa các mã ứng viên (không dùng để chọn mã)", "h2"),
              table([["Mã", "NK ưu đãi", "NK thông thường", "GTGT", "TTĐB", "Chính sách quản lý"]] +
                    [[r["ma8"], r.get("nk_uu_dai"), r.get("nk_thong_thuong"), r.get("vat"), r.get("ttdb"), (r.get("chinh_sach") or "")[:160]] for r in hq],
                    [22 * mm, 18 * mm, 22 * mm, 16 * mm, 14 * mm, 74 * mm])]
    for c in res.get("chu_giai_chuong_can_doc") or []:
        s.append(P(f"Chú giải chương {c['chuong']}{' (thuộc phần loại trừ)' if c['thuoc_phan_loai_tru'] else ''}: {c['dieu_khoan']}"))
    rr = res.get("rui_ro")
    if rr:
        s += [P(f"Mức rủi ro nếu khai theo gợi ý: {rr['muc']}", "h2"), P("; ".join(rr.get("ly_do") or []) or "Không có dấu hiệu đáng chú ý."),
              P(rr.get("ghi_chu", ""), "small")]
    for q in res.get("cau_hoi_phan_biet") or []:
        s.append(P(f"Câu hỏi phân biệt ({q['cap']}): {q['cau_hoi']} " + " | ".join(f"{o['ma']}: {o['mo_ta']}" for o in q["lua_chon"])))
    cg = res.get("chu_giai_chi_tiet") or {}
    s += [P("4. Chú giải chi tiết HS 2022", "h2"),
          P(f"Nhóm {cg.get('nhom')}, trang {cg.get('trang')}: {cg.get('trich') or 'không có đoạn chú giải cho nhóm này'}"),
          P("5. Thông tin còn thiếu", "h2"),
          P(", ".join(res.get("thong_tin_nen_kiem_tra") or []) or "Không phát hiện thiếu theo danh sách kiểm tra (kiểm tra bằng từ khóa, có thể sai)"),
          P("6. Quyết định", "h2"), P(f"Hệ thống: {res.get('quyet_dinh')}")]
    if decision:
        s.append(P(f"Người khai: chọn {decision.get('ma_chon')}; lý do: {decision.get('ly_do')}; người khai: {decision.get('nguoi_khai') or ''}"))
    s += [Spacer(1, 8), P(res.get("hieu_luc_tien_le", ""), "small"), P(res.get("gioi_han", ""), "small"), P(res.get("luu_y", ""), "small")]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                      title="Phiếu phân loại mã HS").build(s)
    return buf.getvalue()


def audit_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="cap_lech_ma")
    return buf.getvalue()
