# Hướng dẫn chạy — Trợ lý phân loại mã HS (Nhóm 8)

```
01_DuLieu_MoHinhNen.ipynb  →  02_MoHinhCaiTien.ipynb  →  03_DanhGia.ipynb  →  ứng dụng (hs-assistant-app.zip)
   dữ liệu, mô hình nền           chọn mô hình cuối             đánh giá mô hình cuối          dùng mô hình cuối
```
Các notebook cũ (`HS_Code_Pipeline_Data_to_Model`, `HS_Code_Modeling_v4`, `HS_Code_Evaluation`, `Untitled…`) không dùng nữa.


## Cập nhật bản v3 (bắt buộc chạy lại Bước 1 rồi Bước 2)
- **Bước 1:** đặc trưng tiền lệ của train giờ chỉ dùng tiền lệ có ngày *trước* mặt hàng (trước đây dùng cả tiền lệ sau ngày, làm việc chọn mô hình lạc quan sai). Chạy lại từ ô 2; OCR và embedding dùng lại bộ đệm nên nhanh.
- **Bước 2:** tập chọn dùng cửa sổ thời gian mở rộng; chọn theo nhóm top-1; bộ chọn dòng 8 số riêng (L1); độ tin cậy đầu-cuối (tính cả trường hợp bỏ sót ứng viên) kèm bảng tỷ lệ trả lời – độ chính xác; mô hình cuối được **đánh giá độc lập** so với B1 và M0 (chênh lệch, CI95, McNemar); bảng phân loại lỗi; `giai_thich_tung_mat_hang_test.csv`.
- **LLM:** ô M5 kiểm tra kết nối trước, in nguyên văn lỗi và (với Gemini) danh sách mô hình khóa của bạn dùng được. Lần chạy trước, cả 25 lần gọi đều lỗi nên M5 chưa từng được đánh giá; hãy sửa `LLM_MODEL` theo danh sách in ra.
- **CROSS:** chỉ dùng ruling có ngày trước mặt hàng; kết quả không lọc thời gian chỉ in ra như phân tích độ nhạy.



## Dùng OpenAI (bản 14/10/2026)
1. Tạo khóa API tại trang nền tảng OpenAI, nạp số dư (API trả theo token). Trong Colab: biểu tượng chìa khóa → thêm `OPENAI_API_KEY` → bật "Notebook access".
2. 02 đã đặt sẵn: `LLM_PROVIDER, LLM_MODEL = "openai", "gpt-5-mini"`, dự phòng `["gpt-5.4-mini", "gpt-4.1-mini"]`, `LLM_ON_TRAIN = True`, `LLM_LINES = True`. Nếu tên mô hình sai, ô M5 in danh sách mô hình khóa của bạn dùng được.
3. Ước tính: khoảng 200 lần gọi, cỡ 2–3 triệu token vào và dưới 1 triệu token ra; với giá gpt-5-mini (0,25 USD/1 triệu token vào, 2 USD/1 triệu token ra) là vài USD. Ô cuối Bước 2 in số token thực dùng. Kết quả gọi được lưu đệm (`llm_cache.jsonl`): chạy lại không tính tiền lại.
4. Lỗi: giới hạn tốc độ → tự chờ rồi thử lại; mô hình không có → tự chuyển mô hình dự phòng; hết số dư (insufficient_quota) hoặc khóa sai → dừng và báo rõ.
5. 01: ô 16 (thử LLM trên một phiếu) đã chuyển sang OpenAI, mặc định TẮT. 03 không gọi LLM.
6. Ứng dụng: `HS_USE_LLM=1`, `HS_LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` (tùy chọn `HS_LLM_MODEL`, `HS_LLM_FALLBACK`). Chỉ gửi mô tả đã ẩn danh.

## Bản v8 (14/10/2026) — chỉ cần chạy lại 02 rồi 03 (01 giữ nguyên)
- **Lỗi cấu trúc:** điểm nhóm trước đây là TỔNG xác suất của ~260 ứng viên → nhóm có nhiều dòng điểm thấp được ưu tiên. Trên dữ liệu 14/10: nhóm top-1 0,468 → 0,440, mã 8 số 0,294 → 0,248 ngay cả khi không bước nào thay đổi gì. Nay điểm nhóm = xác suất của dòng tốt nhất (`HEAD_AGG = "max"`), nên pipeline tái lập đúng M0.
- **Mô hình nền cố định M0** (`BASE_MODEL = "M0"`): qua 5 lần chạy, M1 chưa lần nào hơn M0 trên test dù tập chọn nói ngược lại.
- **Quy tắc giữ bước chặt hơn:** tăng ≥ 2 điểm % trên tập chọn và ≥ 1 điểm % riêng trên valid.
- **LLM:** hạn mức tính riêng từng mô hình; hết hạn mức (429) ở một mô hình thì tự thử mô hình kế tiếp trong `LLM_FALLBACK`. `LLM_ON_TRAIN = False` mặc định (~60 lần gọi). Tắt ô Gemini thử nghiệm trong 01 để khỏi tốn hạn mức.

## Bản v7 (13/10/2026) — chạy lại cả 01 và 02
**Sửa lỗi gốc (phát hiện khi đọc từng mặt hàng test):**
- 01: 9/98 đầu vào test là phần căn cứ pháp lý chứ không phải mô tả hàng (OCR làm méo tiêu đề mục 3 → lấy nhầm 2.500 ký tự đầu). Nay nhận tiêu đề mục 3 dù méo, lấy mô tả từ mục 1 tới phần kết luận, lấy mã và nhóm từ phần kết luận; loại thông báo đính chính.
- 02: hợp nhất điểm theo thứ hạng (bản cũ min–max làm nhóm "lạ" chen vào top-3); chọn bước theo điểm tổng hợp (nhóm top-1, top-3, mã 8 số); LLM tự chuyển mô hình dự phòng khi gặp 404/503; LLM chọn dòng 8 số (M5c); LLM được đề xuất một nhóm ngoài danh sách (chỉ nhận nhóm có thật trong Danh mục, trọng số thấp hơn); khoảng tin cậy lấy mẫu lại theo doanh nghiệp.

**Cài đặt 02 nên dùng:** `RUN_CE = False`, `LLM_PROVIDER, LLM_MODEL = "gemini", "gemini-3.5-flash"`, `LLM_FALLBACK` như mặc định, `LLM_ON_TRAIN = True`, `LLM_LINES = True`. Khoảng 210 lần gọi API (8 mặt hàng/lần cho chọn nhóm, 6 mặt hàng/lần cho chọn dòng); hạn mức hẹp thì `LLM_ON_TRAIN = False` (~60 lần).

**03 (bản v7):** giữ M0 của Bước 1 làm hệ thống so sánh bắt buộc (bản cũ bị ghi đè nên chỉ so với baseline); CI theo doanh nghiệp cả ở cấp nhóm; báo độ tin cậy đã hiệu chỉnh và áp τ của Bước 2 đúng lên con số này (bản cũ so τ với xác suất thô ở phần công bằng theo chương); thêm cách quyết định trả lời/chuyển theo độ tin cậy đã hiệu chỉnh; in cảnh báo khi tập test đã dùng lúc phát triển.

**Kiểm định độc lập (bắt buộc nếu muốn công bố):** tập test hiện tại đã được xem nhiều lần khi sửa pipeline nên không còn "sạch". Chốt mô hình (giữ nguyên `final_model.json`), tải các thông báo ban hành SAU ngày chốt vào `raw_pdf`, đặt trong 01 `VALID_FROM = "2025-06-01"`, `TEST_FROM = "<ngày chốt>"`, chạy 01 → 02 (không đổi cài đặt) → 03. Chỉ số chính khai báo trước: mã 8 số top-1; phụ: nhóm top-1, top-3, độ chính xác khi chỉ trả lời phần tự tin.

## Lấy thêm dữ liệu (bản v4)
| Dữ liệu | Lấy ở đâu | Đặt vào | Bước 1 xử lý |
|---|---|---|---|
| SEN 2022 (Chú giải bổ sung ASEAN, kèm Công văn 3866/TCHQ-TXNK) | PDF trên LuatVietnam: https://static.luatvietnam.vn/tai-file-cong-van-3866-tchq-txnk-2023-ban-hanh-chu-giai-bo-sung-sen-2022-263023-d6/uploaded/VIETLAWFILE/2023/8/3866_TCHQ-TXNK_090823102642.pdf.aspx | `02_DuLieu/SEN_2022_CV3866.pdf` | Ô 5b tách theo mã 8 số → `sen_2022.json`; in số mã trùng với 971 dòng (SEN). Bộ tách chưa thử trên tệp thật: nếu số mã ít, mở PDF xem cách ghi mã |
| Thông báo xác định trước mã số 2018–2022 | Cùng nguồn nhóm đã tải 767 văn bản, lọc năm 2018–2022; đặt tên giống cũ `SO_TB-TCHQ_YYYY-MM-DD.pdf` | `02_DuLieu/raw_pdf/` | OCR như cũ; đánh dấu Danh mục 2017; chỉ giữ văn bản thuộc nhóm KHÔNG đổi khi sang Danh mục 2022 (bỏ 319 nhóm có mã bỏ/mã mới theo sheet CU, MOI); chỉ làm tiền lệ train, valid/test giữ nguyên |
| Nhãn nhập tay cho ~87 văn bản OCR không đọc được mã | Mở PDF, điền `mau_nhap_tay.csv` | `03_MoHinh/work/nhan_nhap_tay.csv` | Áp dụng tự động |
| Bản chữ của thông báo (để đối chiếu OCR) | Thư Viện Pháp Luật, LuatVietnam có nhiều thông báo dạng văn bản | Không bắt buộc | Dùng để kiểm tra tay, không tải hàng loạt (điều khoản sử dụng của trang) |

Biểu thuế không cần tải thêm: Bước 1 bản v4 đã dùng thêm chú giải PHẦN (trước đây bị gộp nhầm vào chương đứng trước vì tệp gắn nhãn ngược), căn cứ pháp lý và ngày hiệu lực của từng mức thuế, đơn vị tính, mẫu C/O theo hiệp định (sheet HT), mã bỏ/mã mới 2017→2022 (sheet CU, MOI).

## Ba bước

| Bước | Runtime Colab | Tạo ra | Khi nào chạy lại |
|---|---|---|---|
| 1. `01_DuLieu_MoHinhNen` | GPU T4 (mã hóa BGE-M3) | `03_MoHinh/artifacts/` | Khi sửa dữ liệu, thêm nhãn tay. OCR có lưu đệm |
| 2. `02_MoHinhCaiTien` | **CPU được**; GPU T4 nếu muốn thêm cross-encoder | `artifacts/mo_hinh_v4/` (`model_selection_matrix.csv`, `predictions_final.json`, `final_model.json`, `dong_gop_tung_buoc.csv`, `ce_nhat_ky_huan_luyen.csv`) | Sau mỗi lần chạy Bước 1 |
| 3. `03_DanhGia` | CPU | `artifacts/danh_gia/` | Sau mỗi lần chạy Bước 2 |

## Bước 2 trên Colab miễn phí (tránh hết GPU, mất kết nối)
1. **Lần 1 — runtime CPU:** Runtime → Change runtime type → CPU, rồi Run all. Notebook tự nhận không có GPU và bỏ qua cross-encoder; vẫn ra mô hình cuối, Model Selection Matrix và các tệp cho Bước 3.
2. **Lần 2 — khi có GPU (tùy chọn):** đổi sang T4 rồi Run all lại. Các phần đã xong được **dùng lại từ Drive** (`mo_hinh_v4/cache_<phiên bản>`), chỉ phần cross-encoder chạy mới (khoảng 10–30 phút tùy mô hình; là ước tính).
3. **Nếu bị ngắt giữa chừng:** kết nối lại, Run all. Phần đã lưu không chạy lại.
4. **Thiếu dung lượng Drive:** ô 2b liệt kê tệp lớn và bản cũ có thể xóa; đặt `CLEAN = True` để xóa, sau đó **vào Google Drive > Thùng rác > Xóa vĩnh viễn** (xóa trong Colab chưa giải phóng dung lượng ngay). Bản tinh chỉnh cross-encoder giờ là mô hình nhỏ, lưu dạng fp16 (~250 MB) và chỉ chép sang thư mục kết quả khi mô hình cuối dùng nó.

## Bước 2 chọn mô hình cuối như thế nào
- **Tập chọn:** dự đoán *ngoài mẫu* trên train (5 khối theo thời gian) cộng valid, khoảng 550 mặt hàng thay vì ~100. Test chỉ để báo cáo.
- **Tiêu chí:** MRR@10 ở cấp nhóm 4 số (dùng cả thứ hạng, không hy sinh top-3).
- **Mô hình nền:** M0 (hồi quy logistic) hoặc M1 (LightGBM); chỉ chọn M1 khi hơn M0 ít nhất 1 điểm. **Chọn dòng 8 số trong nhóm** dùng bộ xếp hạng riêng theo cùng quy tắc.
- **Xếp chồng:** thử M3 nguyên mẫu nhóm → M2 giám sát yếu (mặc định tắt) → M4 cross-encoder (chỉ giữ một bản: dùng sẵn hoặc tinh chỉnh) → M5 LLM → M6 CROSS. Bước nào tăng MRR trên tập chọn thì giữ. Bước học từ nhãn train (M2, M4 tinh chỉnh) chỉ được chọn trên valid.
- **Giải thích:** `dong_gop_tung_buoc.csv` cho biết mỗi bước đổi nhóm số 1 của bao nhiêu mặt hàng, đổi sai thành đúng hay đúng thành sai; SHAP cho LightGBM.

## Có cần mua API không? Không bắt buộc
- **Không có khóa:** Bước 2 chạy M0–M4 (M4 khi có GPU).
- **Có khóa Gemini miễn phí (Google AI Studio):** đặt `LLM_PROVIDER = "gemini"`, `LLM_MODEL = "<tên mô hình Flash hoặc Flash-Lite đang có trong AI Studio>"`, lưu khóa vào Colab Secrets tên `GEMINI_API_KEY`. Notebook **gộp 8 mặt hàng mỗi lần gọi** (valid + test ≈ 25 lần gọi), có lưu đệm, và **dừng ngay khi hết hạn mức** (không treo); chạy lại hôm sau để làm tiếp. M5 chỉ được đánh giá khi đã có câu trả lời cho ít nhất 95% mặt hàng.
- `LLM_ON_TRAIN = True` cho phép chọn bước LLM trên tập lớn hơn (thêm khoảng 57 lần gọi).
- Chỉ nên trả phí khi cần hạn mức ổn định hoặc dùng LLM trong ứng dụng với dữ liệu thật của doanh nghiệp.

## Ứng dụng
1. Tải `03_MoHinh/artifacts` về máy, đặt cạnh `app_streamlit.py` (hoặc đặt biến `HS_ART`).
2. `pip install -r requirements.txt` rồi `streamlit run app_streamlit.py`.
3. Ứng dụng đọc `artifacts/mo_hinh_v4/final_model.json` và dùng đúng mô hình cuối (mô hình nền, bộ chọn dòng, các bước). Dòng "Mô hình: …" trên giao diện cho biết bước nào đang dùng, bước nào bị tắt.
   - `HS_USE_CE=0` tắt cross-encoder; LLM mặc định tắt, bật bằng `HS_USE_LLM=1`, `HS_LLM_PROVIDER`, `HS_LLM_MODEL` và khóa API trong biến môi trường.
4. Kiểm tra ứng dụng tái lập đúng Bước 2: `pytest -q tests/test_artifacts.py::test_final_matches_predictions_final`.

## Đọc kết quả
- Hai chỉ số chính: nhóm top-1 và nhóm top-3 so với B1 (McNemar, Holm). Top-3 được thêm sau khi đã xem kết quả, nên ghi là **phân tích khám phá**; muốn khẳng định cần một tập test mới (thông báo ban hành sau ngày chốt mô hình).
- Báo cáo riêng: nhóm chưa có trong train, tập kiểm thử chặt, phần trả lời tự tin, tổn thất theo hậu quả (Bước 3).

## Lỗi thường gặp
| Hiện tượng | Cách xử lý |
|---|---|
| Ô 1 báo `Failed to fetch ... poppler-utils ... 404 Not Found` | Danh sách gói Colab bị cũ. Dùng ô 1 bản mới (có `apt-get update`), hoặc chạy `!apt-get update` rồi chạy lại ô 1 |
| `FileNotFoundError ... features_train.npz` | Chưa chạy xong Bước 1 |
| "Không có GPU: bỏ qua cross-encoder" | Bình thường khi chạy CPU; chạy lại với T4 khi có |
| `CUDA out of memory` | Giữ `CE_FT_MODEL` mặc định (mô hình nhỏ); đặt `CE_ZS_ON_TRAIN = False` |
| "Hết hạn mức API" | Chạy lại sau khi hạn mức làm mới; phần đã có không gọi lại |
| Bước 3 ghi "Đánh giá mô hình: M0" | Chưa có `predictions_final.json`: chạy Bước 2 |
