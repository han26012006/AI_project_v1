import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ART = os.environ.get("HS_ART", os.path.join(ROOT, "artifacts"))
HAS_ART = os.path.exists(os.path.join(ART, "ranker.joblib"))
needs_art = pytest.mark.skipif(not HAS_ART, reason="Chưa có thư mục artifacts (chạy notebook pipeline trước)")


@pytest.fixture(scope="session")
def art():
    return ART


@pytest.fixture(scope="session")
def assistant_train_only():
    from hs_assistant.core import HSAssistant
    return HSAssistant(ART, precedent_splits=("train",))
