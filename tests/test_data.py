import pytest
from kion.data import validate_tables, ensure_archive
from kion.config import Config


def test_audit_and_bad_schema(tables):
    e, u, i = tables
    assert validate_tables(e, u, i)["users_without_metadata"] == 1
    with pytest.raises(ValueError, match="Missing required"):
        validate_tables(e.drop(columns="total_dur"), u, i)
    with pytest.raises(ValueError, match="Duplicate"):
        import pandas as pd
        validate_tables(pd.concat([e, e.iloc[:1]]), u, i)


def test_invalid_archive_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not the dataset")
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_archive(Config(root=tmp_path), path)
