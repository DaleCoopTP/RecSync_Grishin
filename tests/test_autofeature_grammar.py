import pytest

from kion_autofeatures.grammar import FeatureSpec, enumerate_features, resolve_specs


def test_finite_grammar_serializes_without_data():
    specs = enumerate_features()
    assert len(specs) == len({s.name for s in specs}) == 504
    assert [FeatureSpec.from_dict(s.to_dict()) for s in specs] == specs
    assert resolve_specs([s.name for s in specs]) == specs
    assert all("label" not in s.name and "query_id" not in s.name for s in specs)


def test_invalid_compositions_are_rejected():
    for args in [("user", "share"), ("user_genres", "mean_label"),
                 ("user", "trend", 7, 14), ("item", "recency", 7)]:
        with pytest.raises(ValueError):
            FeatureSpec(*args)
