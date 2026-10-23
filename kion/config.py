from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
import tomllib


@dataclass
class Config:
    seed: int = 42
    sample_users: int = 100000
    threads: int = 6
    data_dir: str = "data"
    artifact_dir: str = "artifacts"
    results_dir: str = "results"
    horizon_days: int = 7
    train_cutoffs: list[str] = field(default_factory=lambda: ["2021-07-19", "2021-07-26", "2021-08-02"])
    validation_cutoff: str = "2021-08-09"
    test_cutoff: str = "2021-08-16"
    train_users_per_cutoff: int = 5000
    candidate_limit: int = 200
    source_limit: int = 100
    preserve_per_source: int = 10
    rrf_constant: int = 60
    knn_neighbors: int = 100
    knn_history_limit: int = 100
    knn_shrinkage: float = 10.0
    batch_users: int = 256
    windows: list[int] = field(default_factory=lambda: [7, 30, 90])
    depths: list[int] = field(default_factory=lambda: [6, 8])
    l2_values: list[int] = field(default_factory=lambda: [3, 10])
    iterations: int = 1500
    learning_rate: float = 0.05
    early_stopping: int = 100
    bootstrap_samples: int = 1000
    root: Path = field(default_factory=lambda: Path.cwd(), repr=False)

    @classmethod
    def load(cls, path="config.toml"):
        path = Path(path).resolve()
        with path.open("rb") as f:
            cfg = cls(**tomllib.load(f), root=path.parent)
        cfg.validate()
        return cfg

    def validate(self):
        from datetime import date, timedelta
        positive = (self.sample_users, self.threads, self.horizon_days, self.train_users_per_cutoff,
                    self.candidate_limit, self.source_limit, self.batch_users, self.iterations)
        if min(positive) < 1 or self.preserve_per_source * 4 > self.candidate_limit:
            raise ValueError("Positive sizes and capacity for preserved candidates are required")
        if (self.preserve_per_source < 0 or self.preserve_per_source > self.source_limit
                or self.knn_neighbors < 1 or self.knn_history_limit < 1 or self.knn_shrinkage <= 0
                or self.rrf_constant < 1 or self.early_stopping < 1 or self.learning_rate <= 0
                or not {7, 30}.issubset(self.windows) or any(w <= 0 for w in self.windows)):
            raise ValueError("Invalid retrieval, time-window or training parameters")
        dates = [date.fromisoformat(x) for x in self.train_cutoffs + [self.validation_cutoff, self.test_cutoff]]
        if any(a + timedelta(days=self.horizon_days) > b for a, b in zip(dates, dates[1:])):
            raise ValueError("Label windows must not overlap or reach beyond the next cutoff")

    def path(self, name):
        return self.root / getattr(self, name)

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "root"}

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
