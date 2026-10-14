# -*- coding: utf-8 -*-
"""Lõi suy luận của Trợ lý phân loại mã HS (Nhóm 8).

Dùng chung cho API (FastAPI), giao diện (Streamlit), MLflow và kiểm thử.
Logic sinh ứng viên, đặc trưng và xếp hạng giống hệt notebook HS_Code_Pipeline_Data_to_Model.ipynb;
mọi thay đổi ở notebook phải được cập nhật ở đây (kiểm thử test_artifacts.py so sánh hai bên).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import time
import unicodedata

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------- văn bản
_VN = "0-9a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
RE_WORD = re.compile(f"[{_VN}]+")
RE_ANY8 = re.compile(r"\b(\d{4})\s*[.,]\s*(\d{2})\s*[.,]\s*(\d{2})\b")
RE_HEAD = re.compile(r"(ph[âa]n\s*nh[óo]m|nh[óo]m)\s*\d{2,4}\s*[.,]\s*\d{2}", re.I)
RE_MST = re.compile(r"(m[ãa]\s*s[ốôó]\s*thu[ếêé]\s*:?\s*\d[\d\s\-]{8,16})", re.I)
RE_CTY = re.compile(r"C[ôo]ng\s*ty\s+[^\n,;:]{3,80}", re.I)
RE_ADDR = re.compile(r"(?<!\[)đ[ịi]a\s*ch[ỉi]\s*:?[^\n;]{5,150}", re.I)   # không bắt lại chính nhãn "[ĐỊA CHỈ]" (ẩn danh chạy lại vẫn giữ nguyên)
RE_PHONE = re.compile(r"(đi[ệe]n\s*tho[ạa]i|đt|tel|fax)\s*:?\s*[\d\s.\-()+]{8,20}", re.I)
RE_EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
RE_PERSON = re.compile(r"\b(Ông|Bà)\s+[A-ZĐ]\w*(\s+[A-ZĐ]\w*){1,4}")
HQ_COLS = {"nk_uu_dai": "thuế NK ưu đãi", "nk_thong_thuong": "thuế NK thông thường", "vat": "thuế GTGT", "ttdb": "thuế TTĐB",
           "bvmt": "thuế BVMT", "xk": "thuế xuất khẩu", "chinh_sach": "chính sách quản lý"}
REF_COLS = {"dvt": "đơn vị tính", "nk_uu_dai_van_ban": "căn cứ thuế NK ưu đãi", "nk_uu_dai_hieu_luc": "hiệu lực thuế NK ưu đãi",
            "vat_van_ban": "căn cứ thuế GTGT"}
PRECEDENT_NOTE = ("Thông báo xác định trước chỉ có giá trị với tổ chức đã đề nghị, hiệu lực tối đa 3 năm và chấm dứt khi căn cứ "
                  "pháp lý thay đổi; ở đây chỉ dùng làm tham khảo.")
LIMITATION = "Hệ thống chưa tự áp dụng QT2–QT5; phần giải thích xếp hạng là thống kê, không phải lập luận pháp lý."

ATTR = {"thành phần, vật liệu": r"(th[àa]nh ph[ầa]n|v[ậa]t li[ệe]u|ch[ấa]t li[ệe]u|%|nh[ựu]a|th[ép]|g[ỗo]|v[ảa]i)",
        "công dụng": r"(c[ôo]ng d[ụu]ng|d[ùu]ng đ[ểe]|s[ửu] d[ụu]ng)",
        "cấu tạo, nguyên lý": r"(c[ấa]u t[ạa]o|g[ồo]m|nguy[êe]n l[ýy]|ho[ạa]t đ[ộo]ng)",
        "thông số kỹ thuật": r"(\d+\s*(v|w|kg|mm|cm|ml|%)\b)"}

FEATS = ["lab_bm25", "lab_vec", "lab_vec_rr", "prec_bm25", "prec_vec", "prec_bm25_h4", "prec_vec_h4",
         "note_vec_h4", "vote_h4", "vote_h6", "prior", "loai_khac", "head_rrf", "head_rank_rr", "in_head_rr"]
DISCLAIMER = "Kết quả do AI đề xuất, chỉ để tham khảo; người khai chịu trách nhiệm về nội dung khai (Luật Hải quan 2014, Điều 18)."


def unaccent(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")


def words(s) -> list[str]:
    return RE_WORD.findall(str(s).lower())


def toks(s) -> list[str]:
    w = words(s)
    return w + unaccent(" ".join(w)).split()


def anonymize_and_scrub(s: str) -> str:
    """Ẩn danh mã số thuế, tên doanh nghiệp; xóa mã 8 số và tham chiếu nhóm (chặn rò rỉ nhãn)."""
    s = RE_EMAIL.sub("[EMAIL]", str(s))
    s = RE_PHONE.sub("[ĐT]", s)
    s = RE_ADDR.sub("[ĐỊA CHỈ]", s)
    s = RE_PERSON.sub("[NGƯỜI]", s)
    s = RE_MST.sub("[MST]", s)
    s = RE_CTY.sub("[DN]", s)
    s = RE_ANY8.sub("[MÃ]", s)
    s = RE_HEAD.sub("[NHÓM]", s)
    return re.sub(r"\s+", " ", s).strip()


def _num(x):
    try:
        return float(str(x).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _rates(x) -> list[float]:
    """'8/10', '*/5/8/10' -> [8.0, 10.0] (bỏ ký hiệu không phải số)."""
    return [v for v in (_num(t) for t in re.split(r"[/;]", str(x))) if v is not None]


def short_path(path: str, n: int = 3) -> str:
    return " > ".join(str(path).split(" > ")[-n:])


def softmax(z: np.ndarray, T: float) -> np.ndarray:
    z = np.asarray(z, float) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def mix_scores(a: dict, b: dict, beta: float, k: int = 10) -> dict:
    """Giống Bước 2 (bản v6): hợp nhất theo thứ hạng (weighted reciprocal rank fusion), giữ thứ tự hạng 2–10 của mô hình nền."""
    ra = {h: r for r, h in enumerate(sorted(a, key=lambda h: -a[h]))}
    rb = {h: r for r, h in enumerate(sorted(b, key=lambda h: -b[h]))}
    return {h: 10.0 * ((1 - beta) / (k + ra.get(h, 10 ** 6)) + beta / (k + rb.get(h, 10 ** 6))) for h in sorted(set(a) | set(b))}


P_RERANK = (
    "Bạn hỗ trợ phân loại hàng hóa theo Danh mục hàng hóa xuất khẩu, nhập khẩu Việt Nam (HS 2022, AHTN 2022).\n"
    "Áp dụng Quy tắc 1: căn cứ nội dung nhóm và chú giải phần, chương; nếu nhiều nhóm cùng phù hợp, áp dụng Quy tắc 3 theo thứ tự (a), (b), (c).\n"
    "Trường \"nhom\" CHỈ được chọn trong danh sách ứng viên dưới đây. Nếu mô tả không đủ để phân biệt, trả \"chua_du_can_cu\".\n"
    "Nếu CHẮC CHẮN nhóm đúng không có trong danh sách, ghi mã nhóm 4 số đó vào \"nhom_ngoai\"; không chắc thì để \"\".\n"
    "Trả lời đúng một đối tượng JSON: {\"nhom\": \"XXXX\", \"nhom_ngoai\": \"XXXX hoặc rỗng\", \"ly_do\": \"...\", \"quy_tac\": [\"QT1\", ...], \"thong_tin_can_bo_sung\": [\"...\"]}\n\n"
    "Mô tả hàng hóa:\n{mo_ta}\n\nNhóm ứng viên:\n{ung_vien}")


def sim(A, B) -> np.ndarray:
    S = A @ B.T
    return S.toarray() if hasattr(S, "toarray") else np.asarray(S)


def conformal_qhat(scores, alpha: float) -> float:
    """Ngưỡng split conformal: lượng tử bậc ceil((n+1)(1-alpha))/n của điểm không phù hợp."""
    s = np.asarray(scores, float)
    n = len(s)
    if n == 0:
        return 1.0
    return float(np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


class FastBM25:
    """BM25 Okapi bằng ma trận thưa (cùng công thức với notebook)."""

    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.cv = CountVectorizer(analyzer=toks)
        X = self.cv.fit_transform(list(texts)).tocoo().astype(np.float32)
        N = X.shape[0]
        dl = np.asarray(X.tocsr().sum(1)).ravel()
        avg = dl.mean() if N else 1.0
        df = np.bincount(X.col, minlength=X.shape[1])
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        data = idf[X.col] * X.data * (k1 + 1) / (X.data + k1 * (1 - b + b * dl[X.row] / avg))
        self.W = sparse.csr_matrix((data, (X.row, X.col)), shape=X.shape)

    def scores(self, texts) -> np.ndarray:
        return np.asarray((self.cv.transform(list(texts)) @ self.W.T).todense())

    def get_scores(self, text: str) -> np.ndarray:
        return self.scores([text])[0]


class Embedder:
    """Vector BGE-M3 (nạp mô hình khi cần) hoặc TF-IDF ký tự (nạp bộ vector hóa đã lưu từ notebook)."""

    def __init__(self, backend: str, art_dir: str):
        self.backend = backend
        if backend == "bge-m3":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-m3")
            self.model.max_seq_length = 512
        else:
            self.vec = joblib.load(os.path.join(art_dir, "tfidf_vectorizer.joblib"))

    def encode(self, texts, bs: int = 32):
        texts = [str(t)[:2000] for t in texts]
        if self.backend == "bge-m3":
            return self.model.encode(texts, batch_size=bs, normalize_embeddings=True).astype(np.float32)
        return normalize(self.vec.transform([unaccent(t.lower()) for t in texts]))


class HSAssistant:
    """Nạp artifact và trả lời: phân loại một mặt hàng, rà soát danh mục."""

    def __init__(self, art_dir: str, precedent_splits=("train", "valid", "test")):
        self.art = art_dir
        self.labels = pd.read_parquet(os.path.join(art_dir, "label_space.parquet"))
        self.notes = pd.read_parquet(os.path.join(art_dir, "chu_giai_chunks.parquet"))
        self.prec_all = pd.read_parquet(os.path.join(art_dir, "precedents.parquet"))
        for c in ("hs8", "hs6", "hs4", "ch"):
            if c in self.prec_all:
                self.prec_all[c] = self.prec_all[c].astype(str)
        self.prec_all["hs8"] = self.prec_all.hs8.str.zfill(8)
        b = joblib.load(os.path.join(art_dir, "ranker.joblib"))
        self.scaler, self.ranker = b["scaler"], b["ranker"]
        self.TEMP, self.QHAT, self.ALPHA = float(b["TEMP"]), float(b["QHAT"]), float(b["ALPHA"])
        self.K_SRC, self.backend = int(b["K_SRC"]), b["EMB_BACKEND"]
        self.use_muc3 = bool(b.get("USE_MUC3", False))
        self.FEAT_IDX = list(b.get("FEAT_IDX", range(12)))
        self.N_HEAD_EXPAND = int(b.get("N_HEAD_EXPAND", 0))
        self.TAU_HEAD, self.QHAT_HEAD = float(b.get("TAU_HEAD", 1.01)), float(b.get("QHAT_HEAD", 1.0))
        self.ALPHA_HEAD = float(b.get("ALPHA_HEAD", 0.3))
        cpath = os.path.join(art_dir, "chapter_notes.json")
        self.chapters = json.load(open(cpath, encoding="utf-8")) if os.path.exists(cpath) else {}

        self.lab_codes = self.labels.ma8.values
        self.lab_pos = {c: i for i, c in enumerate(self.lab_codes)}
        self.note_pos = {h: i for i, h in enumerate(self.notes.nhom.values)}
        self.nhom_idx = {h: np.asarray(ix) for h, ix in self.labels.groupby("nhom").indices.items()}
        self.is_khac = self.labels.mo_ta.str.startswith("Loại khác").values
        self.heads = np.array(sorted(self.nhom_idx))
        _hp = {h: i for i, h in enumerate(self.heads)}
        self.lab_h = np.array([_hp[h] for h in self.labels.nhom.values])
        self.prior = collections.Counter(self.prec_all.loc[self.prec_all.split == "train", "hs8"])

        self.emb = Embedder(self.backend, art_dir)
        label_text = (self.labels.duong_dan + " | " + self.labels.duong_dan_en).tolist()
        note_text = (self.notes.ten_nhom + ". " + self.notes.noi_dung.str[:1800]).tolist()
        # văn bản tiền lệ: mô tả mục 1–2, cộng "tên gọi theo cấu tạo, công dụng" ở mục 3 nếu notebook bật USE_MUC3
        muc3 = self.prec_all["ten_goi_muc3"].fillna("") if "ten_goi_muc3" in self.prec_all else ""
        docs_all = (self.prec_all.input_text + " " + muc3) if self.use_muc3 else self.prec_all.input_text
        if self.backend == "bge-m3":
            self.E_lab = np.load(os.path.join(art_dir, "E_lab.npy"))
            self.E_note = np.load(os.path.join(art_dir, "E_note.npy"))
            f = "E_doc.npy" if self.use_muc3 else "E_q.npy"
            E_all = np.load(os.path.join(art_dir, f))
        else:
            self.E_lab, self.E_note = self.emb.encode(label_text), self.emb.encode(note_text)
            E_all = self.emb.encode(docs_all.tolist())
        mask = self.prec_all.split.isin(precedent_splits).values
        self.prec = self.prec_all[mask].reset_index(drop=True)
        self.E_prec = E_all[np.where(mask)[0]]
        self.bm_lab = FastBM25(self.labels.duong_dan.tolist())
        _j = lambda f: json.load(open(os.path.join(art_dir, f), encoding="utf-8")) if os.path.exists(os.path.join(art_dir, f)) else {}
        self.fta_co, self.sen = _j("fta_mau_co.json"), _j("sen_2022.json")
        if getattr(self, "clauses", None) is None and os.path.exists(os.path.join(art_dir, "section_notes.json")):
            pass
        self._load_final(art_dir)
        self.bm_prec = FastBM25(docs_all[mask].tolist())
        self.version = hashlib.md5(json.dumps({"n_lab": len(self.labels), "n_prec": len(self.prec), "T": self.TEMP,
                                               "q": self.QHAT, "emb": self.backend}).encode()).hexdigest()[:8]

    # ------------------------------------------------------------ mô hình cuối của Bước 2
    def _load_final(self, art_dir):
        self.final, self.lgbm, self.steps_active, self.steps_skipped = None, None, [], []
        fm = next((p for p in (os.path.join(art_dir, "mo_hinh_v4", "final_model.json"), os.path.join(art_dir, "final_model.json"))
                   if os.path.exists(p)), None)
        if not fm:
            return
        self.final, self.fm_dir = json.load(open(fm, encoding="utf-8")), os.path.dirname(fm)
        self.K_HEAD = int(self.final.get("k_head", 10))
        self.head_desc = self.labels.groupby("nhom").duong_dan.first().str.split(" > ").str[0]
        self.note_txt = self.notes.set_index("nhom").noi_dung.str[:400]
        if self.final.get("lgbm_file"):
            import lightgbm as lgb
            self.lgbm = lgb.Booster(model_file=os.path.join(self.fm_dir, self.final["lgbm_file"]))
        self.base_kind, self.line_kind = self.final.get("base", "M0"), self.final.get("line_ranker", self.final.get("base", "M0"))
        self.T_base, self.T_line = float(self.final.get("T_base", self.TEMP)), float(self.final.get("T_line", self.final.get("T_base", self.TEMP)))
        lm = self.final.get("line_model") or {}
        self.line_model = lm if lm.get("dung") else None
        if self.line_model:
            self.seg = self.labels.duong_dan.str.split(" > ").str[-1].values
            self.bm_seg = FastBM25(list(self.seg))
            self.seg_nums = [self._nums(x) for x in self.seg]
            self.seg_khac = np.array([str(x).startswith("Loại khác") for x in self.seg])
        self.calib = self.final.get("calibration")
        self._prior8 = collections.Counter(self.prec.hs8.values)
        for st in self.final.get("steps", []):
            kind = st.get("loai")
            try:
                if kind == "nguyen_mau":
                    self._build_prototypes(float(st.get("w_note", 2.0)))
                elif kind == "chu_giai_phap_ly":
                    self.clauses = json.load(open(os.path.join(self.fm_dir, st.get("tep", "chu_giai_phap_ly.json")), encoding="utf-8"))
                    self.bm_cl = FastBM25([c["text"] for c in self.clauses])
                elif kind == "phan_loai_nhom" and self.backend == "bge-m3":
                    self.m2 = joblib.load(os.path.join(self.fm_dir, st["tep"]))
                elif kind == "cross_encoder" and os.environ.get("HS_USE_CE", "1") == "1":
                    import torch
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer
                    src = os.path.join(self.fm_dir, st["thu_muc"]) if st.get("thu_muc") else st["mo_hinh"]
                    self._dev = "cuda" if torch.cuda.is_available() else "cpu"
                    self._tok = AutoTokenizer.from_pretrained(src)
                    self._ce = AutoModelForSequenceClassification.from_pretrained(src).to(self._dev).eval()
                elif kind in ("llm", "llm_dong") and os.environ.get("HS_USE_LLM", "0") == "1":
                    pass
                else:
                    self.steps_skipped.append(st["ten"]); continue
                self.steps_active.append(st)
            except Exception as e:  # thiếu thư viện hoặc tệp: bỏ bước, ghi lại để báo người dùng
                self.steps_skipped.append(f"{st['ten']} ({str(e)[:60]})")

    def _build_prototypes(self, w_note=2.0):
        # giống Bước 2: trung bình embedding các dòng 8 số của nhóm + w × chú giải chi tiết của nhóm, rồi chuẩn hóa
        heads_all = sorted(set(self.labels.nhom)); hpos = {h: i for i, h in enumerate(heads_all)}
        rows = np.array([hpos[h] for h in self.labels.nhom.values])
        cnt = np.maximum(np.bincount(rows, minlength=len(heads_all)), 1).astype(np.float32)
        keep = [i for i, h in enumerate(self.notes.nhom.values) if h in hpos]
        nrows = np.array([hpos[self.notes.nhom.values[i]] for i in keep])
        if sparse.issparse(self.E_lab):
            M = sparse.csr_matrix((1.0 / cnt[rows], (rows, np.arange(len(rows)))), shape=(len(heads_all), len(rows)))
            Mn = sparse.csr_matrix((np.full(len(keep), w_note), (nrows, np.array(keep))), shape=(len(heads_all), self.E_note.shape[0]))
            P = normalize(M @ self.E_lab + Mn @ self.E_note)
        else:
            P = np.zeros((len(heads_all), self.E_lab.shape[1]), np.float32)
            np.add.at(P, rows, self.E_lab); P /= cnt[:, None]
            PN = np.zeros_like(P); np.add.at(PN, nrows, self.E_note[keep])
            P = P + w_note * PN; P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
        self.P3, self.heads_all = P, heads_all

    @staticmethod
    def _nums(t):
        return {x.replace(",", ".") for x in re.findall(r"\d+(?:[.,]\d+)?", str(t))}

    def _line_order(self, text, h):
        """Thứ tự các dòng 8 số trong nhóm h: bộ chọn dòng L1 của Bước 2 nếu có, không thì xác suất dòng của mô hình nền."""
        codes = list(self.lab_codes[self.nhom_idx.get(h, np.array([], dtype=int))])
        if not codes:
            return [], np.array([])
        base = np.array([self._last_line.get(c, 0.0) for c in codes]); base = base / (base.sum() or 1.0)
        if not self.line_model:
            o = np.argsort(-base, kind="stable"); return [codes[i] for i in o], base[o]
        ix = [self.lab_pos[c] for c in codes]
        if getattr(self, "_bm_text", None) != text:
            self._bm_text, self._bm_cache = text, self.bm_seg.get_scores(text[:1500])
        bm = self._bm_cache[ix]; bm = bm / max(float(bm.max()), 1e-9); qn = self._nums(text[:1500])
        rows = [[b_, m_, float(self.seg_khac[j]), len(qn & self.seg_nums[j]) / max(1, len(self.seg_nums[j]))] for b_, m_, j in zip(base, bm, ix)]
        lm = self.line_model
        if len(lm["mean"]) == 7:                      # đặc trưng bổ sung của Bước 2 bản v5 (giống notebook)
            vs = np.asarray(sim(self._last_qvec, self.E_lab[ix])).ravel(); vs = vs - vs.max()
            pt = sum(self._prior8[c] for c in codes) or 1
            rows = [r + [float(vs_), self._vote8[c] / 20.0, self._prior8[c] / pt] for r, vs_, c in zip(rows, vs, codes)]
        X = np.array(rows)
        z = ((X - np.array(lm["mean"])) / np.array(lm["scale"])) @ np.array(lm["coef"]) + lm["intercept"]
        p = softmax(z, 1.0); o = np.argsort(-z, kind="stable")
        return [codes[i] for i in o], p[o]

    def _notes_scores(self, text, cand_heads):
        """Giống Bước 2 (M7): điều khoản chú giải nhắc tới nhóm → ủng hộ; điều khoản loại trừ trong chương/phần của nhóm → chống."""
        sims = self.bm_cl.get_scores(str(text)[:1500]); mx = float(sims.max()) if len(sims) else 0.0
        if mx <= 0:
            return {}
        sims = sims / mx; top = np.argsort(-sims)[:10]
        heads = list(dict.fromkeys(list(cand_heads) + [h for j in top for h in self.clauses[j]["nhom"]]))
        pos, neg = collections.defaultdict(float), collections.defaultdict(float)
        for j in np.argsort(-sims)[:60]:
            c = self.clauses[j]
            for h in c["nhom"]:
                pos[h] = max(pos[h], float(sims[j]))
            if c["loai_tru"]:
                for h in heads:
                    if h[:2] in c["chuong"] and h not in c["nhom"]:
                        neg[h] = max(neg[h], float(sims[j]))
        return {h: pos.get(h, 0.0) - neg.get(h, 0.0) for h in heads}

    def _confidence(self, ph):
        if not self.final or not self.calib:
            return None
        p1 = float(ph[0]) if len(ph) else 0.0; p2 = float(ph[1]) if len(ph) > 1 else 0.0
        z = float(np.dot(self.calib["coef"], [p1, p1 - p2]) + self.calib["intercept"])
        return 1.0 / (1.0 + math.exp(-z))

    def _head_text(self, h):
        return f"Nhóm {h[:2]}.{h[2:]}: {self.head_desc.get(h, '')}. {self.note_txt.get(h, '')}"

    def _ce_scores(self, text, heads):
        import torch
        with torch.no_grad():
            enc = self._tok([text[:1500]] * len(heads), [self._head_text(h) for h in heads], padding=True, truncation=True,
                            max_length=384, return_tensors="pt").to(self._dev)
            return self._ce(**enc).logits.view(-1).float().cpu().numpy()

    @staticmethod
    def _llm_text(prov, models, prompt):
        for model in models:
            try:
                if prov == "gemini":
                    from google import genai
                    return genai.Client(api_key=os.environ["GEMINI_API_KEY"]).models.generate_content(model=model, contents=prompt).text
                if prov == "openai":
                    from openai import OpenAI
                    cl, msg = OpenAI(timeout=120), [{"role": "user", "content": prompt}]
                    try:
                        return cl.chat.completions.create(model=model, temperature=0, messages=msg).choices[0].message.content
                    except Exception as e:
                        if "temperature" not in str(e): raise
                        return cl.chat.completions.create(model=model, messages=msg).choices[0].message.content   # mô hình suy luận (GPT-5…)
                if prov == "anthropic":
                    import anthropic
                    return anthropic.Anthropic().messages.create(model=model, max_tokens=600, temperature=0,
                                                                 messages=[{"role": "user", "content": prompt}]).content[0].text
                return None
            except Exception as e:                   # mô hình không còn / quá tải → thử mô hình dự phòng
                if not any(x in str(e) for x in ("404", "NOT_FOUND", "no longer available", "503", "UNAVAILABLE", "overloaded", "model_not_found", "rate_limit")):
                    return None
        return None

    def _llm_line(self, text, h, codes, st):
        """Giống Bước 2 (M5c): LLM chọn MỘT dòng 8 số trong nhóm h; chỉ nhận mã nằm trong danh sách."""
        prov = os.environ.get("HS_LLM_PROVIDER", st.get("provider"))
        models = [os.environ.get("HS_LLM_MODEL", st.get("model"))] + [m for m in os.environ.get("HS_LLM_FALLBACK", "").split(",") if m]
        lst = "\n".join(f"{c}: {' > '.join(str(self.labels.loc[self.lab_pos[c], 'duong_dan']).split(' > ')[-2:])[:160]}"
                        + (f" [SEN: {self.sen[c][:150]}]" if c in self.sen else "") for c in codes[:30])
        prompt = ("Bạn hỗ trợ phân loại hàng hóa theo Danh mục hàng hóa xuất khẩu, nhập khẩu Việt Nam (AHTN 2022). Nhóm 4 số đã được chọn; "
                  "chọn MỘT dòng 8 số trong danh sách, so sánh các phân nhóm cùng cấp (Quy tắc 6), chỉ chọn \"Loại khác\" khi không thuộc dòng cụ thể nào. "
                  "Thiếu thông tin thì trả \"chua_du_can_cu\". Trả lời đúng một đối tượng JSON: {\"ma8\": \"XXXXXXXX\", \"thieu\": \"...\"}\n\n"
                  f"Mô tả: {text[:1000]}\nNhóm {h[:2]}.{h[2:]} — các dòng:\n{lst}")
        ans = self._llm_text(prov, models, prompt)
        m = re.search(r"\{.*\}", str(ans or ""), re.S)
        try:
            pk = re.sub(r"\D", "", str(json.loads(m.group(0)).get("ma8", ""))) if m else ""
        except Exception:
            pk = ""
        return pk if pk in codes else None

    def _llm_pick(self, text, heads, st):
        prov = os.environ.get("HS_LLM_PROVIDER", st.get("provider"))
        models = [os.environ.get("HS_LLM_MODEL", st.get("model"))] + [m for m in os.environ.get("HS_LLM_FALLBACK", "").split(",") if m]
        cg = ""
        if getattr(self, "bm_cl", None) is not None:
            sims = self.bm_cl.get_scores(text[:1500])
            cg = "\nChú giải pháp lý có thể liên quan:\n" + "\n".join(f"- [{self.clauses[j]['pham_vi']}] {self.clauses[j]['text'][:250]}" for j in np.argsort(-sims)[:3] if sims[j] > 0)
        prompt = P_RERANK.replace("{mo_ta}", text[:1500] + cg).replace("{ung_vien}", "\n".join(f"{h}: {self.head_desc.get(h, '')[:200]}" for h in heads))
        ans = self._llm_text(prov, models, prompt)
        if ans is None:
            return None
        m = re.search(r"\{.*\}", str(ans), re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            obj = {}
        pick = re.sub(r"\D", "", str(obj.get("nhom", ""))).zfill(4) if obj.get("nhom") else None
        ng = re.sub(r"\D", "", str(obj.get("nhom_ngoai", "") or ""))
        # đề xuất ngoài danh sách: chỉ nhận nhóm có thật trong Danh mục (giống Bước 2), trọng số thấp hơn nhóm chọn trong danh sách
        self._llm_out = ng if (len(ng) == 4 and ng in self.nhom_idx and ng not in heads) else None
        return pick if pick in heads else None          # chặn hallucination: "nhom" chỉ nhận nhóm trong danh sách ứng viên

    def _final(self, text, qvec, codes, probs):
        """Áp các bước của mô hình cuối (Bước 2) lên điểm nhóm của mô hình nền; trả về mã, xác suất, nhóm đã sắp lại."""
        hp = collections.Counter()
        for c, pp in zip(codes, probs):
            hp[c[:4]] = max(hp[c[:4]], float(pp)) if (self.final and self.final.get("head_agg") == "max") else hp[c[:4]] + float(pp)
        if not self.final:
            return codes, probs, hp.most_common(), False
        cur, base_heads = dict(hp), [h for h, _ in hp.most_common()]
        order = lambda d: sorted(d, key=lambda h: -d[h])
        for st in self.steps_active:
            kind = st["loai"]
            if kind == "nguyen_mau":
                S = np.asarray(sim(qvec, self.P3)).ravel(); top = np.argsort(-S)[:30]
                extra = {self.heads_all[j]: float(S[j]) for j in top}
            elif kind == "chu_giai_phap_ly":
                extra = self._notes_scores(text, order(cur)[:20])
            elif kind == "phan_loai_nhom":
                P = self.m2.predict_proba(qvec)[0]; top = np.argsort(-P)[:30]
                extra = {self.m2.classes_[j]: float(P[j]) for j in top}
            elif kind == "llm_dong":
                continue
            else:
                a = order(cur)[:self.K_HEAD]; cand = a + [h for h in base_heads[:self.K_HEAD] if h not in a]
                if kind == "cross_encoder":
                    extra = dict(zip(cand, self._ce_scores(text, cand).tolist()))
                else:
                    self._llm_out = None; pick = self._llm_pick(text, cand, st)
                    extra = {**({pick: 1.0} if pick else {}), **({self._llm_out: 0.5} if self._llm_out else {})}   # giống Bước 2 (M5a)
            cur = mix_scores(cur, extra, float(st["beta"])) if st["loai"] != "llm_dong" else cur
        hs = order(cur)
        ph = softmax(np.array([cur[h] for h in hs]), float(self.final.get("T_head", 1.0))) if self.steps_active else np.array([cur[h] for h in hs])
        out_c, out_p = [], []
        llm_line = next((st for st in self.steps_active if st["loai"] == "llm_dong"), None)
        for k_, (h, phh) in enumerate(zip(hs[:3], ph[:3])):            # nhóm trước, dòng sau (QT1 → QT6), giống Bước 2
            cs, pl = self._line_order(text, h)
            if k_ == 0 and llm_line and cs:
                pk = self._llm_line(text, h, cs, llm_line)
                if pk:
                    j = cs.index(pk); cs = [pk] + cs[:j] + cs[j + 1:]; pl = np.concatenate([[pl[j]], pl[:j], pl[j + 1:]])
            out_c += cs[:5]; out_p += [float(phh) * float(x) for x in pl[:5]]
        return out_c, np.array(out_p), list(zip(hs, ph.tolist())), True

    # ------------------------------------------------------------ đặc trưng
    def _rank_unique(self, codes, k):
        seen, res = set(), []
        for c in codes:
            if c not in seen:
                seen.add(c)
                res.append(c)
            if len(res) == k:
                break
        return res

    def features_for(self, q: str, qvec, exclude=None):
        lb = self.bm_lab.get_scores(q); lb = lb / (lb.max() + 1e-9)
        lv = np.asarray(sim(qvec, self.E_lab)).ravel()
        pb = self.bm_prec.get_scores(q); pb = pb / (pb.max() + 1e-9)
        pv = np.asarray(sim(qvec, self.E_prec)).ravel()
        if exclude is not None:
            pb[exclude] = -1; pv[exclude] = -1
        nv = np.asarray(sim(qvec, self.E_note)).ravel()
        pc = self.prec.hs8.values
        best_pb, best_pv, best_pb4, best_pv4 = {}, {}, {}, {}
        for j in np.argsort(-pb)[:50]:
            if pb[j] < 0: continue
            c = pc[j]; best_pb[c] = max(best_pb.get(c, -1), pb[j]); best_pb4[c[:4]] = max(best_pb4.get(c[:4], -1), pb[j])
        for j in np.argsort(-pv)[:50]:
            if pv[j] < 0: continue
            c = pc[j]; best_pv[c] = max(best_pv.get(c, -1), pv[j]); best_pv4[c[:4]] = max(best_pv4.get(c[:4], -1), pv[j])
        # xếp hạng nhóm 4 số từ 4 nguồn, hợp nhất RRF (giống notebook)
        hs_lab = np.full(len(self.heads), -9.0); np.maximum.at(hs_lab, self.lab_h, lv)
        lists = []
        for arr in (pv, pb):
            d = {}
            for j in np.argsort(-arr)[:60]:
                if arr[j] > -1:
                    d.setdefault(pc[j][:4], arr[j])
            lists.append(sorted(d, key=lambda h: -d[h]))
        lists += [list(self.heads[np.argsort(-hs_lab)[:60]]), list(self.notes.nhom.values[np.argsort(-nv)[:60]])]
        rrf = collections.Counter()
        for lst in lists:
            for r_, h in enumerate(lst[:60]):
                rrf[h] += 1 / (60 + r_)
        hord = [h for h, _ in rrf.most_common(60)]
        hrank = {h: i for i, h in enumerate(hord)}
        mx = max(rrf.values()) if rrf else 1.0
        top10 = [pc[j] for j in np.argsort(-pv)[:10] if pv[j] > -1]
        v4 = collections.Counter(c[:4] for c in top10); v6 = collections.Counter(c[:6] for c in top10)
        lv_rank = np.empty(len(lv)); lv_rank[np.argsort(-lv)] = np.arange(1, len(lv) + 1)
        K = self.K_SRC
        cand = list(self.lab_codes[np.argsort(-lb)[:K]]) + list(self.lab_codes[np.argsort(-lv)[:K]])
        cand += self._rank_unique(pc[np.argsort(-pb)], 10) + self._rank_unique(pc[np.argsort(-pv)], 10)
        for h in self.notes.nhom.values[np.argsort(-nv)[:3]]:
            sub = self.nhom_idx.get(h, np.array([], dtype=int))
            cand += list(self.lab_codes[sub[np.argsort(-lv[sub])[:5]]])
        for h in hord[:self.N_HEAD_EXPAND]:
            cand += list(self.lab_codes[self.nhom_idx[h]])
        cand = [c for c in dict.fromkeys(cand) if c in self.lab_pos]
        rows = []
        for c in cand:
            li = self.lab_pos[c]; h4 = c[:4]
            rows.append([lb[li], lv[li], 1.0 / lv_rank[li], best_pb.get(c, 0), best_pv.get(c, 0), best_pb4.get(h4, 0),
                         best_pv4.get(h4, 0), nv[self.note_pos[h4]] if h4 in self.note_pos else 0, v4.get(h4, 0) / 10,
                         v6.get(c[:6], 0) / 10, math.log1p(self.prior.get(c, 0)), float(self.is_khac[li]),
                         rrf.get(h4, 0) / mx, 1 / (hrank[h4] + 1) if h4 in hrank else 0,
                         1 / ((lv[self.nhom_idx[h4]] > lv[li]).sum() + 1)])
        return cand, np.asarray(rows, dtype=np.float32), pv

    # ------------------------------------------------------------ phân loại
    def rank(self, text: str):
        text = anonymize_and_scrub(text)
        qvec = self.emb.encode([text])
        cand, X, pv = self.features_for(text, qvec)
        Z = self.scaler.transform(X[:, self.FEAT_IDX])
        s_lr = self.ranker.decision_function(Z)
        s_gb = self.lgbm.predict(X[:, self.FEAT_IDX]) if self.lgbm is not None else None
        if self.final and self.base_kind == "M1":
            p = softmax(s_gb, self.T_base)
        else:
            p = softmax(s_lr, self.final and self.T_base or self.TEMP)
        p_line = (softmax(s_gb, self.T_line) if (self.final and self.line_kind == "M1") else softmax(s_lr, self.TEMP)) if self.final else p
        self._last_line = dict(zip(cand, p_line.tolist()))
        o = np.argsort(-p); self._last_qvec = qvec
        return text, [cand[j] for j in o], p[o], Z[o], pv

    def hau_qua(self, codes):
        """So sánh thuế và chính sách giữa các mã ứng viên. Không dùng để xếp hạng mã; chỉ để quyết định có cần leo thang."""
        rows = [{"ma8": c, **{k: str(self.labels.loc[self.lab_pos[c], k]) if k in self.labels.columns else "" for k in HQ_COLS},
                 **{k: str(self.labels.loc[self.lab_pos[c], k]) for k in REF_COLS if k in self.labels.columns}}
                for c in codes]
        return rows, [HQ_COLS[k] for k in HQ_COLS if len({r[k] for r in rows}) > 1]

    def note_clauses(self, ch):
        out, excl = [], False
        for line in self.chapters.get(ch, {}).get("chu_giai", "").split("\n"):
            l = line.strip()
            if len(l) < 15:
                continue
            if re.match(r"^\d+\.", l):
                excl = bool(re.search(r"không bao gồm|không áp dụng", l, re.I))
            out.append((l, excl or bool(re.search(r"không bao gồm", l, re.I))))
        return out

    def kiem_tra_chu_giai(self, text, code, top=2):
        cl = self.note_clauses(code[:2])
        if not cl:
            return []
        sc = FastBM25([c for c, _ in cl]).get_scores(text)
        return [{"chuong": code[:2], "dieu_khoan": cl[j][0][:350], "thuoc_phan_loai_tru": cl[j][1],
                 "nhom_duoc_nhac_toi": ["".join(x) for x in re.findall(r"(\d{2})\.(\d{2})", cl[j][0])][:6]}
                for j in np.argsort(-sc)[:top] if sc[j] > 0]

    def cau_hoi_phan_biet(self, nhom, codes, k_lines=6):
        """Câu hỏi giúp người khai tự phân biệt: giữa các nhóm 4 số (khi nhóm chưa chắc) và giữa các dòng cùng cấp (QT6)."""
        qs = []
        if len(nhom) >= 2 and nhom[0][1] < self.TAU_HEAD:
            qs.append({"cap": "nhóm 4 số", "cau_hoi": "Theo nội dung nhóm và chú giải chương, mặt hàng thuộc nhóm nào?",
                       "lua_chon": [{"ma": h, "mo_ta": self.labels.loc[self.nhom_idx[h][0], "duong_dan"].split(" > ")[0][:200]}
                                    for h, _ in nhom[:3]]})
        h = nhom[0][0] if nhom else None
        lines = [c for c in codes if h and c[:4] == h][:k_lines]
        if len(lines) >= 2:
            paths = [self.labels.loc[self.lab_pos[c], "duong_dan"].split(" > ") for c in lines]
            i = 0
            while all(len(pth) > i for pth in paths) and len({pth[i] for pth in paths}) == 1:
                i += 1
            opts = {}
            for c, pth in zip(lines, paths):
                seg = pth[i] if len(pth) > i else pth[-1]
                opts.setdefault(seg, []).append(c)
            if len(opts) >= 2:
                qs.append({"cap": "dòng cùng cấp (QT6)", "cau_hoi": "Đặc điểm nào đúng với mặt hàng?",
                           "lua_chon": [{"ma": ", ".join(v), "mo_ta": seg[:200]} for seg, v in opts.items()],
                           "ghi_chu": ("Dòng \"Loại khác\" chỉ chọn khi mặt hàng không thuộc các dòng cụ thể cùng cấp."
                                       if any(seg.startswith("Loại khác") for seg in opts) else "")})
        return qs

    def danh_gia_rui_ro(self, cao, khac, hq, missing, ev, top_code):
        """Phân loại rủi ro theo quy tắc minh bạch (chưa được kiểm chứng thực nghiệm): dùng để quyết định có cần leo thang."""
        diem, ly_do = 0, []
        if not cao:
            diem += 2; ly_do.append("Nhóm 4 số chưa đủ tin cậy")
        if "chính sách quản lý" in khac:
            diem += 2; ly_do.append("Các mã ứng viên khác nhau về chính sách quản lý (giấy phép, kiểm tra chuyên ngành)")
        nk = [v for v in (_num(r.get("nk_uu_dai")) for r in hq) if v is not None]
        if len(nk) >= 2 and max(nk) - min(nk) >= 10:
            diem += 2; ly_do.append(f"Thuế NK ưu đãi giữa các mã ứng viên chênh tới {max(nk) - min(nk):g} điểm phần trăm")
        elif len(nk) >= 2 and max(nk) > min(nk):
            diem += 1; ly_do.append("Thuế NK ưu đãi giữa các mã ứng viên khác nhau")
        if missing:
            diem += 1; ly_do.append("Có thể thiếu thông tin: " + ", ".join(missing))
        if ev:
            e = ev[0]
            if e["ma"][:4] == top_code[:4] and e["ma"] != top_code:
                diem += 1; ly_do.append(f"Tiền lệ gần nhất cùng nhóm nhưng khác dòng 8 số ({e['ma']})")
            elif e["ma"][:4] != top_code[:4] and e["do_tuong_dong"] >= 0.85:
                diem += 1; ly_do.append(f"Tiền lệ rất giống nhưng thuộc nhóm khác ({e['ma']})")
        muc = "Cao" if diem >= 4 else ("Trung bình" if diem >= 2 else "Thấp")
        return {"muc": muc, "diem": diem, "ly_do": ly_do,
                "ghi_chu": "Đánh giá theo quy tắc, chưa kiểm chứng thực nghiệm; dùng để quyết định có cần hỏi chuyên gia hoặc xin xác định trước."}

    def uoc_tinh_thue(self, codes, tri_gia: float, ma_chon: str | None = None, so_ngay_cham: int = 90) -> dict:
        """Ước tính tham khảo thuế NK ưu đãi và GTGT theo từng mã; chênh lệch so với mã đã chọn; tiền phạt, chậm nộp nếu khai thiếu.
        Không tính TTĐB, BVMT (mức tuyệt đối), thuế tự vệ, ưu đãi FTA: chỉ cảnh báo."""
        rows = []
        for c in codes:
            r = self.labels.loc[self.lab_pos[c]]
            nk = _num(r.get("nk_uu_dai")); vat = _rates(r.get("vat", ""))
            t_nk = tri_gia * nk / 100 if nk is not None else None
            base = tri_gia + (t_nk or 0)
            rows.append({"ma8": c, "nk_uu_dai_pct": nk, "thue_nk": t_nk,
                         "gtgt_min": base * min(vat) / 100 if vat else None, "gtgt_max": base * max(vat) / 100 if vat else None,
                         "canh_bao": [x for x, cond in (("có thuế TTĐB", str(r.get("ttdb", "")).strip()),
                                                        ("có thuế BVMT", str(r.get("bvmt", "")).strip()),
                                                        ("có chính sách quản lý", str(r.get("chinh_sach", "")).strip())) if cond]})
        ref = next((x for x in rows if x["ma8"] == ma_chon), rows[0])
        def tong(x, which):
            return (x["thue_nk"] or 0) + (x[which] or 0)
        thieu = max([tong(x, "gtgt_max") - tong(ref, "gtgt_max") for x in rows] + [0])
        return {"tri_gia_tinh_thue": tri_gia, "ma_chon": ref["ma8"], "bang": rows,
                "thieu_toi_da_neu_ma_dung_la_ma_khac": thieu,
                "phat_neu_hai_quan_phat_hien_20pct": 0.2 * thieu,
                "cham_nop_uoc_tinh": thieu * 0.0003 * so_ngay_cham, "so_ngay_cham_gia_dinh": so_ngay_cham,
                "gia_dinh": ["Thuế NK theo mức ưu đãi (MFN) trong biểu thuế; chưa tính ưu đãi FTA, TTĐB, BVMT, thuế tự vệ",
                             "Giá tính thuế GTGT hàng nhập khẩu gồm trị giá tính thuế và thuế nhập khẩu (Luật Thuế GTGT 2024, Điều 7); GTGT lấy theo các mức ghi trong biểu thuế",
                             "Phạt 20% số thuế khai thiếu khi cơ quan hải quan phát hiện; với tổ chức chỉ phạt khi chênh lệch từ 2 triệu đồng/tờ khai (NĐ 128/2020, Điều 9)",
                             "Tiền chậm nộp 0,03%/ngày (Luật Quản lý thuế); số ngày là giả định",
                             "Ước tính tham khảo, không phải tư vấn thuế hay pháp lý"]}

    def uu_dai_fta(self, code):
        """Thuế suất ưu đãi đặc biệt theo FTA của một mã và mẫu C/O tương ứng (chỉ hưởng khi có C/O hợp lệ, đáp ứng quy tắc xuất xứ)."""
        r = self.labels.loc[self.lab_pos[code]]
        out = []
        for c in [c for c in self.labels.columns if c.startswith("fta_")]:
            val = str(r[c]).strip()
            if val and val.lower() not in ("nan", "none", "*"):
                ab = c[4:]; out.append({"hiep_dinh": ab, "thue_suat": val, "mau_co": self.fta_co.get(ab, "")})
        return out

    def classify(self, text: str, k: int = 3) -> dict:
        t0 = time.time()
        text, codes, probs, Z, pv = self.rank(text)
        self._vote8 = collections.Counter(self.prec.hs8.values[j] for j in np.argsort(-pv)[:20])
        codes, probs, nhom_all, final_on = self._final(text, self._last_qvec, codes, probs)
        nhom = nhom_all[:3]
        if final_on:   # mô hình cuối: tập nhóm nhỏ nhất có tổng xác suất >= 1 − α (không phải conformal)
            tap_nhom, acc_p = [], 0.0
            for h, pp in nhom_all:
                tap_nhom.append(h); acc_p += pp
                if acc_p >= 1 - self.ALPHA_HEAD or len(tap_nhom) == 5: break
        else:
            tap_nhom = [h for h, pp in nhom_all if pp >= 1 - self.QHAT_HEAD][:5]
        trong_nhom = [c for c in codes if c[:4] == nhom[0][0]][:5] if nhom else []
        hq, khac = self.hau_qua(codes[:k])
        ev = []
        year = time.localtime().tm_year
        for j in np.argsort(-pv)[:5]:
            r = self.prec.iloc[j]
            m3 = r.get("ten_goi_muc3") if hasattr(r, "get") else None
            ev.append({"so_hieu": int(r.so_hieu) if pd.notna(r.so_hieu) else None,
                       "nam": int(r.nam) if pd.notna(r.nam) else None, "ma": r.hs8,
                       "do_tuong_dong": round(float(pv[j]), 3),
                       "cung_nhom_voi_ma_de_xuat": r.hs8[:4] == codes[0][:4],
                       "qua_3_nam": bool(pd.notna(r.nam) and year - int(r.nam) >= 3),
                       "trich": (m3 if isinstance(m3, str) and m3 else r.input_text)[:200]})
        h4 = codes[0][:4]
        note = self.notes[self.notes.nhom == h4]
        missing = [a for a, pat in ATTR.items() if not re.search(pat, text, re.I)]
        coef = self.ranker.coef_[0]
        contrib = (Z[0] - Z[1]) * coef if len(codes) > 1 else Z[0] * coef
        giai_thich = sorted(zip([FEATS[i] for i in self.FEAT_IDX], contrib.round(3).tolist()), key=lambda x: -abs(x[1]))[:5]
        tau = float(self.final.get("tau_head", 1.01)) if final_on else self.TAU_HEAD
        conf = self._confidence([p for _, p in nhom_all]) if final_on else None
        cao = bool(nhom) and ((conf if conf is not None else nhom[0][1]) >= tau)
        if cao:
            qd = (f"Nhóm {nhom[0][0][:2]}.{nhom[0][0][2:]} có độ tin cậy khá cao; dòng 8 số cần người khai so sánh các dòng "
                  "cùng cấp (QT6) và ghi lý do trước khi khai")
        elif khac:
            qd = ("Chưa đủ căn cứ và các mã ứng viên khác nhau về " + ", ".join(khac) +
                  ": nên hỏi chuyên gia hoặc xin xác định trước mã số")
        else:
            qd = ("Chưa đủ căn cứ; các mã ứng viên giống nhau về thuế và chính sách nhưng vẫn phải khai đúng mã: "
                  "bổ sung thông tin để phân biệt")
        return {
            "ho_so_id": hashlib.md5(f"{text}|{time.time()}".encode()).hexdigest()[:12],
            "mo_hinh": (self.final["ten"] if final_on else "M0 (mô hình nền)"),
            "buoc_khong_dung": self.steps_skipped,
            "do_tin_cay_nhom_1": (round(conf, 3) if conf is not None else None),
            "phien_ban_mo_hinh": self.version,
            "dau_vao_an_danh": text,
            "ma_de_xuat": [{"ma8": c, "xac_suat": round(float(pp), 3),
                            "mo_ta": short_path(self.labels.loc[self.lab_pos[c], "duong_dan"]),
                            "duong_dan": self.labels.loc[self.lab_pos[c], "duong_dan"],
                            "nk_uu_dai": str(self.labels.loc[self.lab_pos[c], "nk_uu_dai"])}
                           for c, pp in zip(codes[:k], probs[:k])],
            "nhom_de_xuat": [{"nhom": h, "xac_suat": round(pp, 3),
                              "mo_ta": self.labels.loc[self.nhom_idx[h][0], "duong_dan"].split(" > ")[0][:160]} for h, pp in nhom],
            "tap_nhom_tin_cay": tap_nhom, "muc_tin_cay_nhom": round(1 - self.ALPHA_HEAD, 2), "do_tin_cay_cao": cao,
            "dong_8_so_trong_nhom_1": [{"ma8": c, "mo_ta": short_path(self.labels.loc[self.lab_pos[c], "duong_dan"])} for c in trong_nhom],
            "hau_qua_giua_cac_ma": hq, "khac_nhau_ve": khac,
            "chu_giai_chuong_can_doc": self.kiem_tra_chu_giai(text, codes[0]),
            "can_cu_tien_le": [e for e in ev if e["cung_nhom_voi_ma_de_xuat"]],
            "tien_le_gan_nhat": ev, "hieu_luc_tien_le": PRECEDENT_NOTE,
            "chu_giai_chi_tiet": {"nhom": h4, "trang": int(note.trang.iloc[0]) if len(note) else None,
                                  "trich": note.noi_dung.iloc[0][:400] if len(note) else None},
            "thong_tin_nen_kiem_tra": missing,
            "giai_thich_xep_hang": [{"dac_trung": f, "dong_gop": c} for f, c in giai_thich],
            "cau_hoi_phan_biet": self.cau_hoi_phan_biet(nhom, codes),
            "uu_dai_fta_ma_1": self.uu_dai_fta(codes[0]) if codes else [],
            "sen_2022_ma_1": (self.sen.get(codes[0], "")[:1200] if codes else ""),
            "rui_ro": self.danh_gia_rui_ro(cao, khac, hq, missing, ev, codes[0]),
            "quyet_dinh": qd, "gioi_han": LIMITATION,
            "do_tuong_dong_tien_le_max": round(float(pv.max()) if len(pv) else 0.0, 3),
            "thoi_gian_ms": int((time.time() - t0) * 1000),
            "luu_y": DISCLAIMER,
        }

    # ------------------------------------------------------------ rà soát danh mục
    def audit_catalog(self, df: pd.DataFrame, text_col: str = "mo_ta", code_col: str = "ma_hs",
                      id_col: str | None = None, thr: float = 0.85) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        E = self.emb.encode(df[text_col].astype(str).map(anonymize_and_scrub).tolist())
        S = sim(E, E)
        np.fill_diagonal(S, 0)
        i, j = np.where(np.triu(S >= thr))
        codes = df[code_col].astype(str).str.replace(r"\D", "", regex=True)
        rows = []
        for a, b in zip(i, j):
            if codes[a] != codes[b]:
                rows.append({"id_A": df.iloc[a][id_col] if id_col else int(a), "mat_hang_A": str(df.iloc[a][text_col])[:120],
                             "ma_A": codes[a], "id_B": df.iloc[b][id_col] if id_col else int(b),
                             "mat_hang_B": str(df.iloc[b][text_col])[:120], "ma_B": codes[b],
                             "do_tuong_dong": round(float(S[a, b]), 3),
                             "cung_nhom_4_so": codes[a][:4] == codes[b][:4]})
        out = pd.DataFrame(rows, columns=["id_A", "mat_hang_A", "ma_A", "id_B", "mat_hang_B", "ma_B", "do_tuong_dong", "cung_nhom_4_so"])
        return out.sort_values("do_tuong_dong", ascending=False).reset_index(drop=True)
