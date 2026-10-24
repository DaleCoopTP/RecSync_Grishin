import pandas as pd
import pytest

from kion_autofeatures.baseline import attach_features
from kion_autofeatures.config import ResearchConfig


def test_attach_rejects_reordered_pairs_and_preserves_labels():
    base = pd.DataFrame({"user_id": [1, 1], "item_id": [2, 3], "label": [0, 1], "query_id": [1, 1]})
    extra = base[["user_id", "item_id"]].assign(auto__count=[2., 3.])
    pd.testing.assert_frame_equal(attach_features(base, extra)[base.columns], base)
    with pytest.raises(ValueError, match="identity or order"):
        attach_features(base, extra.iloc[::-1])
    with pytest.raises(ValueError, match="declared"):
        attach_features(base, extra.assign(label=1))


def test_configuration_caps_resources_and_paths():
    for kwargs in ({"threads": 3}, {"memory_limit_gib": 7}, {"run_name": "../baseline"}):
        with pytest.raises(ValueError):
            ResearchConfig(**kwargs)
