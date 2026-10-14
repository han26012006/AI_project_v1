# Kiểm thử đơn vị: không cần artifact, chạy được trên CI
import numpy as np
import pytest

from hs_assistant.core import FastBM25, _rates, anonymize_and_scrub, conformal_qhat, short_path, softmax, toks, unaccent


def test_scrub_removes_codes_and_identities():
    s = "Công ty TNHH ABC Việt Nam, mã số thuế: 0101234567, đề nghị mã 8516.79.90 thuộc nhóm 85.16"
    out = anonymize_and_scrub(s)
    assert "8516" not in out.replace(" ", "") and "85.16" not in out
    assert "0101234567" not in out and "[MST]" in out and "[DN]" in out


def test_unaccent_and_tokens():
    assert unaccent("Đồ dùng điện") == "Do dung dien"
    t = toks("Máy sấy tóc")
    assert "máy" in t and "may" in t


def test_fast_bm25_matches_rank_bm25_order():
    rank_bm25 = pytest.importorskip("rank_bm25")
    docs = ["máy sấy tóc điện cầm tay", "ấm đun nước điện", "nồi cơm điện gia dụng", "bàn là điện hơi nước",
            "máy sấy tay nhiệt điện", "hạt nhựa PET nguyên sinh"]
    q = "máy sấy tóc cầm tay dùng điện"
    a = FastBM25(docs).get_scores(q)
    b = rank_bm25.BM25Okapi([toks(d) for d in docs]).get_scores(toks(q))
    assert list(np.argsort(-a)[:3]) == list(np.argsort(-b)[:3])


def test_softmax_and_short_path():
    p = softmax(np.array([1.0, 2.0, 3.0]), 1.0)
    assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 2
    assert short_path("A > B > C > D > E") == "C > D > E"


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cal, test = rng.uniform(size=2000), rng.uniform(size=20000)
    q = conformal_qhat(cal, 0.1)
    assert 0.88 <= (test <= q).mean() <= 0.93


def test_scrub_contact_details():
    s = "Địa chỉ: 12 Lê Lợi, Quận 1; Điện thoại: 028 3822 1234; Email: abc@congty.vn; Ông Nguyễn Văn An ký"
    out = anonymize_and_scrub(s)
    assert "Lê Lợi" not in out and "3822" not in out and "abc@" not in out and "Nguyễn Văn An" not in out


def test_vat_rate_parsing():
    assert _rates("8/10") == [8.0, 10.0] and _rates("*/5/8/10") == [5.0, 8.0, 10.0] and _rates("") == []
