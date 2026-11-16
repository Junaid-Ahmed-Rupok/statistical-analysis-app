"""Regression guard: awkward datasets must never produce a traceback or a failed section."""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent))
import awkward_datasets as ad                                   # noqa: E402
from modules.preprocessor import DataPreprocessor               # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
FAST = ["tiny_5rows", "single_column", "single_cat_col", "all_constant", "all_null_col", "half_null",
        "datetime_cols", "unicode_names", "duplicate_colnames", "bool_cols", "multiclass_imbalanced",
        "negatives_inf", "mixed_type_col", "two_rows", "empty_after_dropna", "many_groups_small",
        "long_text", "date_dtype", "category_dtype"]


@pytest.mark.parametrize("name", FAST)
def test_pipeline_survives(name):
    df = pd.read_csv(io.StringIO(ad.D[name].to_csv(index=False)))
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.session_state["uploaded_df"] = df
    at.session_state["loaded_file_name"] = name + ".csv"
    at.session_state["overview"] = DataPreprocessor().get_overview(df)
    at.run()
    for key in ("preprocess_btn", "run_stats_btn"):
        next(b for b in at.button if b.key == key).click()
        at.run(timeout=300)
    assert not at.exception, [e.value[:200] for e in at.exception]
    assert not [w.value for w in at.warning if "could not be completed" in w.value]
