from dataclasses import replace
import pytest
from kion.config import Config


def test_config_loads():
    cfg = Config.load()
    assert cfg.sample_users == 100000
    assert cfg.fingerprint() == Config.load().fingerprint()


def test_overlapping_labels_rejected():
    with pytest.raises(ValueError, match="overlap"):
        replace(Config(), validation_cutoff="2021-08-03").validate()
