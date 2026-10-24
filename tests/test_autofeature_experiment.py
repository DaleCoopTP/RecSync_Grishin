import pytest

from kion_autofeatures.baseline import RAW_COLUMNS, feature_families
from kion_autofeatures.experiment import FINAL_NEW, VALIDATION_NEW, variant_features
from kion_autofeatures.grammar import enumerate_features


def test_variants_change_only_predeclared_feature_families():
    families = {"source": ["cand_score"], "raw": ["i_content_type"], "manual": ["u_count_7"]}
    specs = enumerate_features()
    selection = {"selected": [specs[0].name, specs[1].name], "random": [specs[-1].name, specs[-2].name]}
    variants = variant_features(families, selection)
    assert variants["full"] == ["cand_score", "i_content_type", "u_count_7"]
    assert variants["source_raw"] == ["cand_score", "i_content_type"]
    assert variants["auto"] == variants["source_raw"] + selection["selected"]
    assert "cand_score" not in variants["auto_no_source"]
    assert set(VALIDATION_NEW) == {"source_raw", "auto_no_source", "auto", "random"}
    assert set(FINAL_NEW) == {"source", "source_raw", "auto", "random"}
    with pytest.raises(ValueError):
        variant_features(families, {"selected": selection["selected"], "random": []})


def test_frozen_feature_partition_has_no_service_fields():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "results" / "frozen_validation.json"
    frozen = json.loads(path.read_text())
    families = feature_families(frozen["features"])
    assert set(families["raw"]) == set(RAW_COLUMNS)
    flattened = sum(families.values(), [])
    assert len(flattened) == len(set(flattened)) == 82
    assert not {"label", "query_id", "cutoff", "user_id", "item_id"} & set(flattened)
