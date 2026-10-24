"""Read-only adapter: no research action writes into baseline directories."""
import importlib.metadata
from pathlib import Path
import platform

from kion.config import Config
from kion.io import sha256
from .io import digest, read_json, verify_files, write_json

RAW_COLUMNS = (
    "u_age", "u_income", "u_sex", "u_kids_flg", "i_content_type", "i_countries",
    "i_directors", "i_genres", "i_id", "i_release_year", "i_release_year_missing",
    "i_for_kids", "i_for_kids_missing", "i_age_rating", "i_age_rating_missing",
)
KEYS = ["user_id", "item_id", "label", "query_id", "cutoff"]


def feature_families(columns):
    source = [c for c in columns if c.startswith("cand_") or c == "rrf_score"]
    raw = [c for c in columns if c in RAW_COLUMNS]
    manual = [c for c in columns if c not in source + raw]
    if (len(source), len(raw), len(manual)) != (13, 15, 54):
        raise ValueError("Expected the frozen 13/15/54 baseline feature partition")
    return {"source": source, "raw": raw, "manual": manual}


class Baseline:
    def __init__(self, research):
        self.research = research
        self.cfg = Config.load(research.root / research.baseline_config)
        if self.cfg.root != research.root:
            raise ValueError("Baseline and research must share the project root")
        from kion.final import load_protocol
        self.protocol, self.frozen = load_protocol(self.cfg)
        self.families = feature_families(self.frozen["features"])

    def snapshots(self, stage="validation"):
        if stage == "validation":
            values = self.frozen["candidate_snapshots"]
            return [s for s in values if s["training"]], next(s for s in values if not s["training"])
        if stage != "test":
            raise ValueError("Unknown snapshot stage")
        final = read_json(self.cfg.path("results_dir") / "test_manifest.json")
        return final["training_snapshots"], final["test_snapshot"]

    def protected_files(self):
        from kion.protocol import prepared_dir
        paths = list(self.protocol["invariant_sources"])
        paths.append(str((self.research.root / self.research.baseline_config).relative_to(self.cfg.root)))
        for directory in (self.cfg.path("results_dir"), prepared_dir(self.cfg)):
            paths.extend(str(p.relative_to(self.cfg.root)) for p in directory.iterdir() if p.is_file())
        for stage in ("validation", "test"):
            train, evaluation = self.snapshots(stage)
            paths.extend(s["path"] for s in [*train, evaluation])
        paths.append(self.frozen["model_artifact"])
        final = read_json(self.cfg.path("results_dir") / "test_manifest.json")
        paths.extend(m["path"] for m in final["models"].values())
        ablation = read_json(self.cfg.path("results_dir") / "feature_ablation_manifest.json")
        for record in ablation["benchmarks"].values():
            paths.extend(record["files"])
        return {p: sha256(self.cfg.root / p) for p in sorted(set(paths))}

    def prepare(self):
        versions = {p: importlib.metadata.version(p) for p in self.frozen["versions"]}
        if versions != self.frozen["versions"] or platform.python_version() != self.frozen["python"]:
            raise ValueError("Use the pinned baseline runtime")
        path = self.research.results / "inputs.json"
        if path.exists():
            record = read_json(path)
            if record["research"] != self.research.to_dict():
                raise ValueError("Run already exists with a different research configuration")
            verify_files(self.cfg.root, record["protected_files"])
            return record
        record = {"research": self.research.to_dict(), "baseline_config": self.cfg.to_dict(),
                  "families": self.families, "parameters": self.frozen["parameters"],
                  "versions": versions, "python": platform.python_version(),
                  "protected_files": self.protected_files(), "test_evaluated": False}
        record["input_id"] = digest(record)
        write_json(path, record)
        return record

    def read_snapshot(self, spec, columns=None):
        import pandas as pd
        path = self.cfg.root / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Baseline snapshot checksum mismatch: {path}")
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != spec["rows"]:
            raise ValueError("Baseline snapshot row count changed")
        assert_pairs(frame)
        if "cutoff" in frame and set(frame.cutoff.unique()) != {spec["cutoff"]}:
            raise ValueError("Snapshot contains a different cutoff")
        return frame


def assert_pairs(frame):
    if not {"user_id", "item_id"}.issubset(frame):
        raise ValueError("Pair keys are required")
    keys = [c for c in ("cutoff", "user_id", "item_id") if c in frame]
    if frame.duplicated(keys).any() or frame[keys].isna().any().any():
        raise ValueError("Pairs must be unique and nonmissing within each cutoff")


def attach_features(base, generated):
    """Positional attachment is allowed only after exact pair identity checks."""
    import pandas as pd
    assert_pairs(base)
    assert_pairs(generated)
    if not base[["user_id", "item_id"]].reset_index(drop=True).equals(
            generated[["user_id", "item_id"]].reset_index(drop=True)):
        raise ValueError("Generated features changed candidate identity or order")
    columns = [c for c in generated if c not in ("user_id", "item_id")]
    if any(not c.startswith("auto__") or c in base for c in columns):
        raise ValueError("Only new declared automatic feature columns may be attached")
    return pd.concat([base.reset_index(drop=True), generated[columns].reset_index(drop=True)], axis=1)
