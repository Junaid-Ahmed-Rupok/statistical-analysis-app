import pandas as pd
from streamlit.testing.v1 import AppTest
from pathlib import Path

from modules.privacy import pii_columns

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_flags_by_name_and_by_value():
    df = pd.DataFrame({"Student Name": list("abcdefgh"), "contact_email": ["a@b.com"] * 8,
                       "mobile": ["01712345678"] * 8, "col_x": ["x@y.org"] * 8, "marks": range(8),
                       "grade": list("ABCDEFGH"), "nidal": range(8)})
    flagged = {c for c, _ in pii_columns(df)}
    assert {"Student Name", "contact_email", "mobile", "col_x"} <= flagged
    assert "marks" not in flagged and "grade" not in flagged and "nidal" not in flagged


def test_ordinary_data_not_flagged():
    df = pd.read_csv(Path(APP).parent / "data" / "tips.csv")
    assert pii_columns(df) == []


def test_clear_button_wipes_session():
    at = AppTest.from_file(APP, default_timeout=240).run()
    next(b for b in at.button if "Load Sample Dataset" in b.label).click(); at.run()
    next(b for b in at.button if b.key == "preprocess_btn").click(); at.run(timeout=240)
    assert at.session_state["cleaned_df"] is not None
    next(b for b in at.button if b.key == "clear_data_btn").click(); at.run(timeout=240)
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["cleaned_df"] is None and at.session_state["uploaded_df"] is None
