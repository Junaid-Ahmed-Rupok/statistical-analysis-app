import io

import pandas as pd
import pytest

from modules import seal
from modules.pdf_generator import PDFReportGenerator
from modules.preprocessor import DataPreprocessor
from modules.statistics import StatisticalAnalyzer
from modules.insights import InsightsGenerator


@pytest.fixture(scope="module")
def analysed():
    raw = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "tips.csv")
    pre = DataPreprocessor()
    ov = pre.get_overview(raw)
    cl, _ = pre.clean(raw)
    res = StatisticalAnalyzer().run_all(cl)
    ins = InsightsGenerator().generate(cl, res, res.get("vif", pd.DataFrame()), ov)
    return raw, ov, cl, res, ins


def test_fingerprint_matches_sha256_of_the_downloadable_file(analysed):
    _, _, cl, _, _ = analysed
    data = seal.csv_bytes(cl)
    assert b"\r\n" not in data                         # identical bytes on Windows and Linux
    assert seal.verify(data, seal.group4(seal.fingerprint(data)))


def test_any_change_breaks_the_seal(analysed):
    _, _, cl, _, _ = analysed
    h = seal.fingerprint(seal.csv_bytes(cl))
    tampered = cl.copy()
    tampered.iloc[0, 0] = tampered.iloc[0, 0] + 0.01
    assert not seal.verify(seal.csv_bytes(tampered), h)
    assert not seal.verify(b"x", "not a hash")


def test_ledger_counts_are_consistent(analysed):
    raw, _, cl, res, _ = analysed
    led = seal.honesty_ledger(res, raw, cl)
    assert led["tests"] >= led["nominal"] >= led["survive"] >= 0
    assert led["expected_by_chance"] == pytest.approx(0.05 * led["tests"])
    assert led["choices"]["rows_removed"] == len(raw) - len(cl)


def test_pdf_ends_with_the_seal_that_matches_the_data(analysed):
    import pdfplumber
    raw, ov, cl, res, ins = analysed
    pdf = PDFReportGenerator().generate(df=raw, overview=ov, stats_results=res, insights=ins,
                                        figures=None, cleaned_df=cl)
    with pdfplumber.open(io.BytesIO(pdf.getvalue())) as doc:
        text = doc.pages[-1].extract_text()
    h = seal.fingerprint(seal.csv_bytes(cl))
    assert "Honesty Ledger" in text and h[:12].upper() in text
    assert seal.group4(h[:32]).split()[0] in text
