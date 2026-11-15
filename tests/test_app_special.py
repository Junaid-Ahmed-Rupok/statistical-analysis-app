from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_assumption_doctor_and_writeup_render_after_analysis():
    at = AppTest.from_file(APP, default_timeout=240).run()
    next(b for b in at.button if "Load Sample Dataset" in b.label).click(); at.run()
    for key in ("preprocess_btn", "run_stats_btn"):
        next(b for b in at.button if b.key == key).click(); at.run(timeout=240)
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "Group comparisons" in text and "Thesis-ready Write-up" in text
    assert "Results — Group differences" in text
