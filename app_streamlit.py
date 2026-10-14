# -*- coding: utf-8 -*-
"""Giao diện Trợ lý mã HS. Chạy: streamlit run app_streamlit.py"""
import io
import json
import os

import pandas as pd
import streamlit as st

from hs_assistant import monitor
from hs_assistant.audit_log import AuditLog
from hs_assistant.core import DISCLAIMER, HSAssistant
from hs_assistant.export import audit_excel, dossier_pdf

ART = os.environ.get("HS_ART", "artifacts")
LOG = os.environ.get("HS_LOG", "logs/decisions.sqlite")
st.set_page_config(page_title="Trợ lý mã HS – Nhóm 8", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình và kho căn cứ...")
def load_assistant():
    return HSAssistant(ART)


@st.cache_resource
def load_log():
    return AuditLog(LOG)


a, log = load_assistant(), load_log()
st.title("Trợ lý phân loại mã HS có căn cứ")
st.caption(f"{DISCLAIMER}  Phiên bản mô hình {a.version}.")
tab_a, tab_b, tab_c = st.tabs(["Tra một mặt hàng", "Rà soát danh mục", "Giám sát"])

with tab_a:
    with st.form("tra_ma"):
        c1, c2 = st.columns(2)
        ten = c1.text_input("Tên hàng (tên thương mại, tên kỹ thuật)", key="ten")
        cong_dung = c2.text_input("Công dụng", key="cong_dung")
        thanh_phan = c1.text_area("Thành phần, vật liệu", height=90, key="thanh_phan")
        cau_tao = c2.text_area("Cấu tạo, nguyên lý hoạt động", height=90, key="cau_tao")
        thong_so = c1.text_input("Thông số kỹ thuật", key="thong_so")
        tai_lieu = c2.file_uploader("Catalogue hoặc datasheet (PDF, tùy chọn)", type=["pdf"])
        gui = st.form_submit_button("Phân loại", type="primary")
    if gui:
        extra = ""
        if tai_lieu is not None:
            try:
                from pypdf import PdfReader
                extra = " ".join((pg.extract_text() or "") for pg in PdfReader(tai_lieu).pages[:5])[:3000]
            except Exception as e:
                st.warning(f"Không đọc được tệp PDF: {e}")
        mo_ta = (f"1. Hàng hóa đề nghị: {ten}. Thành phần: {thanh_phan}. Cấu tạo: {cau_tao}. Công dụng: {cong_dung}. "
                 f"Thông số: {thong_so}. {extra}")
        if len(" ".join([ten, thanh_phan, cau_tao, cong_dung, thong_so, extra]).strip()) < 20:
            st.error("Nhập ít nhất tên hàng và một trong các thông tin: thành phần, cấu tạo, công dụng, thông số.")
        else:
            res = a.classify(mo_ta)
            log.log_classification(res)
            st.session_state["res"] = res
    res = st.session_state.get("res")
    if res:
        st.caption(f"Mô hình: {res['mo_hinh']}" + (f" | bước không dùng: {', '.join(res['buoc_khong_dung'])}" if res["buoc_khong_dung"] else ""))
        (st.success if res["do_tin_cay_cao"] else st.warning)(res["quyet_dinh"])
        rr = res["rui_ro"]
        {"Cao": st.error, "Trung bình": st.warning, "Thấp": st.info}[rr["muc"]](
            f"Mức rủi ro nếu khai theo gợi ý: {rr['muc']}. " + ("; ".join(rr["ly_do"]) or "Không có dấu hiệu đáng chú ý."))
        for q in res["cau_hoi_phan_biet"]:
            with st.container(border=True):
                st.write(f"**{q['cau_hoi']}** ({q['cap']})")
                for o in q["lua_chon"]:
                    st.write(f"- {o['ma']}: {o['mo_ta']}")
                if q.get("ghi_chu"):
                    st.caption(q["ghi_chu"])
        cols = st.columns(len(res["ma_de_xuat"]))
        for col, m in zip(cols, res["ma_de_xuat"]):
            with col.container(border=True):
                st.metric(f"Mã {m['ma8'][:4]}.{m['ma8'][4:6]}.{m['ma8'][6:]}", f"{m['xac_suat']:.1%}")
                st.write(m["mo_ta"])
                st.caption(f"Thuế NK ưu đãi: {m['nk_uu_dai']}")
        st.write("**Nhóm 4 số đề xuất:** " + "; ".join(f"{n['nhom'][:2]}.{n['nhom'][2:]} ({n['xac_suat']:.0%})" for n in res["nhom_de_xuat"]))
        st.caption("Các dòng 8 số trong nhóm số 1 để so sánh cùng cấp (QT6): " + ", ".join(d["ma8"] for d in res["dong_8_so_trong_nhom_1"]))
        if res["thong_tin_nen_kiem_tra"]:
            st.info("Nên kiểm tra lại thông tin (tự động, có thể sai): " + ", ".join(res["thong_tin_nen_kiem_tra"]))
        if res["hau_qua_giua_cac_ma"]:
            st.caption("Hệ quả khác nhau giữa các mã ứng viên (chỉ để đánh giá rủi ro, không dùng để chọn mã)")
            st.dataframe(pd.DataFrame(res["hau_qua_giua_cac_ma"]), hide_index=True)
        if res.get("uu_dai_fta_ma_1"):
            with st.expander(f"Ưu đãi FTA của mã {res['ma_de_xuat'][0]['ma8']} (cần C/O hợp lệ và đáp ứng quy tắc xuất xứ)"):
                st.dataframe(pd.DataFrame(res["uu_dai_fta_ma_1"]), hide_index=True)
        if res.get("sen_2022_ma_1"):
            with st.expander("Chú giải bổ sung SEN 2022 cho mã số 1"):
                st.write(res["sen_2022_ma_1"])
        for c in res["chu_giai_chuong_can_doc"]:
            st.write(f"Chú giải chương {c['chuong']}{' – phần loại trừ' if c['thuoc_phan_loai_tru'] else ''}: {c['dieu_khoan']}")
        st.subheader("Căn cứ")
        ev = pd.DataFrame(res["can_cu_tien_le"])
        st.dataframe(ev if len(ev) else pd.DataFrame({"ghi_chu": ["Không có tiền lệ cùng nhóm với mã đề xuất"]}), hide_index=True)
        cg = res["chu_giai_chi_tiet"]
        with st.expander(f"Chú giải chi tiết HS 2022, nhóm {cg['nhom']} (trang {cg['trang']})"):
            st.write(cg["trich"] or "Không có đoạn chú giải cho nhóm này.")
        st.caption(res["hieu_luc_tien_le"] + " " + res["gioi_han"])
        with st.expander("Vì sao mã số 1 xếp trên mã số 2 (thống kê)"):
            st.dataframe(pd.DataFrame(res["giai_thich_xep_hang"]), hide_index=True)
        st.subheader("Quyết định của người khai")
        lua_chon = [m["ma8"] for m in res["ma_de_xuat"]] + ["Mã khác"]
        chon = st.selectbox("Mã chọn", lua_chon, key="chon")
        ma_khac = st.text_input("Nhập mã 8 số", key="ma_khac") if chon == "Mã khác" else ""
        ly_do = st.text_area("Lý do (bắt buộc)", key="ly_do")
        nguoi = st.text_input("Người khai", key="nguoi")
        tri_gia = st.number_input("Trị giá tính thuế của lô hàng (VND, tùy chọn) để ước tính hệ quả", min_value=0.0, step=1e6, key="tri_gia")
        if tri_gia > 0:
            ma_cmp = [m["ma8"] for m in res["ma_de_xuat"]]
            est = a.uoc_tinh_thue(ma_cmp, tri_gia, ma_khac.strip() if chon == "Mã khác" and ma_khac.strip() in a.lab_pos else (chon if chon in ma_cmp else None))
            st.dataframe(pd.DataFrame(est["bang"]), hide_index=True)
            st.write(f"Nếu mã đúng là mã khác trong danh sách: thuế có thể thiếu tới {est['thieu_toi_da_neu_ma_dung_la_ma_khac']:,.0f} đồng; "
                     f"phạt ước tính {est['phat_neu_hai_quan_phat_hien_20pct']:,.0f} đồng; chậm nộp {est['cham_nop_uoc_tinh']:,.0f} đồng "
                     f"(giả định {est['so_ngay_cham_gia_dinh']} ngày).")
            st.caption(" ".join(est["gia_dinh"]))
        if st.button("Lưu quyết định", key="luu"):
            ma = ma_khac.strip() if chon == "Mã khác" else chon
            if not (ma.isdigit() and len(ma) == 8) or len(ly_do.strip()) < 5:
                st.error("Cần mã 8 số và lý do tối thiểu 5 ký tự.")
            else:
                st.session_state["dec"] = log.log_decision(res["ho_so_id"], ma, ly_do, nguoi)
                st.success("Đã lưu quyết định vào nhật ký.")
        d1, d2 = st.columns(2)
        d1.download_button("Tải phiếu phân loại (JSON)", json.dumps(res, ensure_ascii=False, indent=1, default=str),
                           file_name=f"phieu_{res['ho_so_id']}.json", mime="application/json")
        d2.download_button("Tải phiếu phân loại (PDF)", dossier_pdf(res, st.session_state.get("dec")),
                           file_name=f"phieu_{res['ho_so_id']}.pdf", mime="application/pdf")

with tab_b:
    st.write("Tải danh mục mặt hàng (CSV hoặc Excel) có cột mô tả và cột mã đang khai.")
    f = st.file_uploader("Danh mục mặt hàng", type=["csv", "xlsx"], key="dm")
    nguong = st.slider("Ngưỡng tương đồng", 0.70, 0.98, 0.85, 0.01)
    if f is not None:
        df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        c1, c2, c3 = st.columns(3)
        col_mo_ta = c1.selectbox("Cột mô tả", list(df.columns), index=list(df.columns).index("mo_ta") if "mo_ta" in df else 0)
        col_ma = c2.selectbox("Cột mã HS", list(df.columns), index=list(df.columns).index("ma_hs") if "ma_hs" in df else 0)
        col_id = c3.selectbox("Cột mã mặt hàng (tùy chọn)", ["(không có)"] + list(df.columns))
        if st.button("Rà soát", key="ra_soat"):
            out = a.audit_catalog(df, col_mo_ta, col_ma, None if col_id == "(không có)" else col_id, nguong)
            st.write(f"Tìm thấy {len(out)} cặp mặt hàng tương tự nhưng khai khác mã.")
            st.dataframe(out, hide_index=True)
            st.download_button("Tải kết quả (Excel)", audit_excel(out), file_name="ra_soat_danh_muc.xlsx")

with tab_c:
    cls, dec = log.frames()
    s = monitor.summary(cls, dec)
    m = st.columns(4)
    m[0].metric("Số lần phân loại", s["so_lan_phan_loai"])
    m[1].metric("Tỷ lệ gợi ý độ tin cậy cao", f"{s.get('ty_le_tin_cay_cao', 0):.0%}")
    m[2].metric("Tỷ lệ người khai sửa mã", f"{s.get('ty_le_sua_ma', 0):.0%}")
    m[3].metric("Độ trễ p95 (ms)", s.get("do_tre_p95_ms") or 0)
    if len(cls) >= 20:
        st.write("Drift phân bố chương so với dữ liệu huấn luyện:",
                 monitor.chapter_drift(a.prec_all.loc[a.prec_all.split == "train", "hs8"], cls.top1.dropna()))
    st.dataframe(cls.drop(columns=["ket_qua_json", "dau_vao"], errors="ignore").tail(50), hide_index=True)
