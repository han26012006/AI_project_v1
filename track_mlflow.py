# -*- coding: utf-8 -*-
"""Ghi một lần chạy vào MLflow và đăng ký mô hình. Chạy: python track_mlflow.py --art artifacts"""
import argparse
import json
import os
import re

import mlflow
import mlflow.pyfunc
import pandas as pd


class HSPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from hs_assistant.core import HSAssistant
        self.a = HSAssistant(context.artifacts["art_dir"])

    def predict(self, context, model_input, params=None):
        out = []
        for t in model_input["mo_ta"].astype(str):
            r = self.a.classify(t)
            out.append({"ma_1": r["ma_de_xuat"][0]["ma8"], "xac_suat_1": r["ma_de_xuat"][0]["xac_suat"],
                        "nhom_1": r["nhom_de_xuat"][0]["nhom"] if r["nhom_de_xuat"] else None, "do_tin_cay_cao": r["do_tin_cay_cao"]})
        return pd.DataFrame(out)


def key(s):
    return re.sub(r"[^0-9A-Za-z_\-. /]", "_", s)[:250]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=os.environ.get("HS_ART", "artifacts"))
    ap.add_argument("--uri", default="sqlite:///mlflow.db")
    ap.add_argument("--register", default="hs-assistant")
    args = ap.parse_args()
    mlflow.set_tracking_uri(args.uri)
    mlflow.set_experiment("tro-ly-ma-hs")
    m = json.load(open(os.path.join(args.art, "metrics.json"), encoding="utf-8"))
    with mlflow.start_run(run_name=f"pipeline-{m.get('emb')}"):
        mlflow.log_params({"emb": m.get("emb"), "alpha": m["conformal"]["alpha"], "temp": m.get("temp"),
                           **{f"n_{k}": v for k, v in m.get("so_nhan", {}).items()}})
        mlflow.log_metrics({"ece": m["ece"], "aurc": m["aurc"], "conformal_qhat": m["conformal"]["qhat"],
                            "conformal_coverage": m["conformal"]["do_phu"], "conformal_mean_size": m["conformal"]["kich_thuoc_tb"]})
        for r in m.get("ket_qua", []):
            for col in ("top1 8 số", "top3 8 số", "top1 6 số", "F1 phân cấp"):
                if col in r:
                    mlflow.log_metric(key(f"{r['mô hình']}__{col}".replace(" ", "_")), float(r[col]))
        mlflow.log_artifact(os.path.join(args.art, "metrics.json"))
        if os.path.exists(os.path.join(args.art, "bang_ket_qua.csv")):
            mlflow.log_artifact(os.path.join(args.art, "bang_ket_qua.csv"))
        kw = dict(python_model=HSPyfunc(), artifacts={"art_dir": args.art}, code_paths=["hs_assistant"],
                  registered_model_name=args.register or None)
        try:
            info = mlflow.pyfunc.log_model(name="model", **kw)          # MLflow 3.x
        except TypeError:
            info = mlflow.pyfunc.log_model(artifact_path="model", **kw)  # MLflow 2.x
        print("Đã ghi run và mô hình:", info.model_uri)


if __name__ == "__main__":
    main()
