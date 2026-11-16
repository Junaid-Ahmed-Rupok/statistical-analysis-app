import io
import pytest
from modules.csv_loader import CSVError, read_csv


def rd(b):
    return read_csv(io.BytesIO(b))


def test_plain_and_bom_and_cp1252():
    assert rd(b"a,b\n1,2\n").shape == (1, 2)
    assert list(rd("﻿a,b\n1,2\n".encode("utf-8")).columns) == ["a", "b"]
    assert rd("name,v\ncaf\xe9,1\n".encode("cp1252")).iloc[0, 0] == "café"


def test_semicolon_and_tab_delimiters():
    assert rd(b"a;b\n1;2\n3;4\n").shape == (2, 2)
    assert rd(b"a\tb\n1\t2\n").shape == (1, 2)


@pytest.mark.parametrize("blob", [b"", b"   \n\n", b"a,b\n", b"\n\n\n"])
def test_empty_or_header_only_gives_friendly_error(blob):
    with pytest.raises(CSVError):
        rd(blob)


def test_blank_and_duplicate_headers_survive():
    df = rd(b"a,,a\n1,2,3\n")
    assert df.shape == (1, 3) and all(isinstance(c, str) and c for c in df.columns)


def test_binary_garbage_does_not_crash():
    try:
        rd(bytes(range(256)) * 20)
    except CSVError:
        pass
