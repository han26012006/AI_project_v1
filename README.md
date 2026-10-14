# Trợ lý phân loại mã HS có căn cứ — ứng dụng (Nhóm 8)

Phần vận hành của đồ án: API, giao diện, nhật ký quyết định, xuất phiếu phân loại, giám sát, MLflow, kiểm thử tự động.
Mô hình và kho căn cứ được tạo bởi 3 notebook theo thứ tự: `01_DuLieu_MoHinhNen` → `02_MoHinhCaiTien` → `03_DanhGia` (xem HUONG_DAN_CHAY.md). Ứng dụng tự đọc `artifacts/mo_hinh_v4/final_model.json` để dùng đúng mô hình cuối; biến môi trường `HS_USE_CE=0` tắt cross-encoder, `HS_USE_LLM=1` bật LLM.

## Cấu trúc
```
hs_assistant/core.py       lõi suy luận (giống hệt logic notebook): phân loại, phiếu phân loại (hệ quả giữa các mã, chú giải chương cần đọc,
                           phạm vi hiệu lực của tiền lệ), rà soát danh mục
hs_assistant/audit_log.py  nhật ký phân loại và quyết định (SQLite)
hs_assistant/export.py     xuất phiếu phân loại PDF, kết quả rà soát Excel
hs_assistant/monitor.py    tỷ lệ chấp nhận, tỷ lệ sửa mã, độ trễ, drift (PSI chương, độ tương đồng), hiệu chỉnh lại conformal
api.py                     FastAPI: /health, /classify, /audit, /decision
app_streamlit.py           giao diện: tra một mặt hàng, rà soát danh mục, giám sát
track_mlflow.py            ghi run và đăng ký mô hình vào MLflow
tests/                     kiểm thử đơn vị, kiểm thử khớp notebook, kiểm thử API
```

## Chuẩn bị
1. Chạy notebook pipeline; tải thư mục `03_MoHinh/artifacts/` từ Drive về, đặt tên `artifacts/` cạnh tệp này
   (hoặc đặt biến môi trường `HS_ART` trỏ tới thư mục đó).
2. `pip install -r requirements.txt`

## Chạy
- Giao diện: `streamlit run app_streamlit.py`
- API: `uvicorn api:app --host 0.0.0.0 --port 8000` (`/classify`, `/estimate` ước tính hệ quả thuế, `/audit`, `/decision`) (tài liệu tại `http://localhost:8000/docs`). Khi chạy ngoài máy cá nhân, đặt `HS_API_KEY` và gửi header `X-API-Key`.
- Kiểm thử: `pytest -q` (các kiểm thử cần artifact tự bỏ qua nếu chưa có `artifacts/`)
- MLflow: `python track_mlflow.py --art artifacts`, sau đó `mlflow ui --backend-store-uri sqlite:///mlflow.db`

Nếu artifact được tạo với `EMB_BACKEND = "bge-m3"`, lần chạy đầu sẽ tải mô hình BAAI/bge-m3 (cần Internet, khoảng 2 GB).

## Kiểm thử đã có
- Ẩn danh và chặn rò rỉ mã trong đầu vào; BM25 ma trận thưa cho cùng thứ hạng với rank_bm25; conformal đạt độ phủ trên dữ liệu hoán đổi được.
- Lược đồ không gian nhãn; không rò rỉ nhãn trong kho tiền lệ.
- Lõi ứng dụng tái tạo dự đoán của notebook (ít nhất 98% top-1 khớp) và không thấp hơn top-3 của notebook quá 0,02.
- Căn cứ tiền lệ trong phiếu luôn cùng nhóm 4 số với mã đề xuất; phiếu có bảng hệ quả, điều khoản chú giải, phạm vi hiệu lực, giới hạn.
- Luồng API phân loại → quyết định → rà soát; khóa API được kiểm tra; ẩn danh địa chỉ, điện thoại, email, tên người.

CI (`.github/workflows/ci.yml`) chạy `pytest` trên mỗi lần push; trên GitHub không có artifact nên chỉ các kiểm thử đơn vị chạy.

## Tính năng cho người khai
- Chọn nhóm 4 số trước, rồi các dòng 8 số cùng cấp; câu hỏi phân biệt giữa các nhóm và giữa các dòng (QT6).
- Mức rủi ro (Thấp, Trung bình, Cao) theo quy tắc minh bạch: độ tin cậy nhóm, khác biệt chính sách và thuế giữa các mã, thông tin còn thiếu, tiền lệ mâu thuẫn.
- Ước tính hệ quả thuế khi nhập trị giá tính thuế: thuế NK ưu đãi, GTGT, mức thiếu tối đa nếu mã đúng là mã khác, tiền phạt và chậm nộp ước tính.

## Lưu ý
- Thuế và chính sách chỉ dùng để đánh giá hệ quả và quyết định có cần leo thang; không dùng để chọn hay xếp hạng mã.
- Lựa chọn của người dùng là nhãn đồng (chưa kiểm chứng); không dùng để huấn luyện lại mô hình. Chỉ kết luận chính thức của Hải quan hoặc chuyên gia đã duyệt mới là nhãn huấn luyện.
- Kết quả chỉ để tham khảo; người khai chịu trách nhiệm về nội dung khai (Luật Hải quan 2014, Điều 18).
- Nhật ký lưu mô tả đã ẩn danh; không đưa `artifacts/`, `logs/` lên kho mã công khai (đã có trong `.gitignore`).
- Chạy trên máy nội bộ; nếu bật LLM qua API, chỉ gửi mô tả đã ẩn danh.
