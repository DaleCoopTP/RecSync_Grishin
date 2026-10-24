from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True)
class ResearchConfig:
    baseline_config: str = "config.toml"
    run_name: str = "main"
    threads: int = 2
    inner_users: int = 1000
    screen_iterations: int = 120
    screen_depth: int = 6
    max_features: int = 64
    prefix_sizes: tuple[int, ...] = (16, 32, 64)
    tie_tolerance: float = 0.0001
    batch_rows: int = 20000
    memory_limit_gib: float = 6.0
    prediction_repeats: int = 3
    root: Path = field(default_factory=Path.cwd, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "prefix_sizes", tuple(self.prefix_sizes))
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.run_name):
            raise ValueError("Use a simple, unique research run name")
        if self.threads not in (1, 2) or not 0 < self.memory_limit_gib <= 6:
            raise ValueError("Research is limited to two threads and 6 GiB RSS")
        if min(self.inner_users, self.screen_iterations, self.screen_depth,
               self.max_features, self.batch_rows, self.prediction_repeats) < 1:
            raise ValueError("Research sizes must be positive")
        if (not self.prefix_sizes or min(self.prefix_sizes) < 1
                or max(self.prefix_sizes) > self.max_features or self.tie_tolerance < 0):
            raise ValueError("Invalid feature-prefix selection settings")

    @classmethod
    def load(cls, path="autofeatures.toml"):
        path = Path(path).resolve()
        with path.open("rb") as stream:
            return cls(**tomllib.load(stream), root=path.parent)

    def to_dict(self):
        value = {k: v for k, v in asdict(self).items() if k != "root"}
        value["prefix_sizes"] = list(self.prefix_sizes)
        return value

    @property
    def artifacts(self):
        return self.root / "artifacts" / "autofeatures" / self.run_name

    @property
    def results(self):
        return self.root / "results" / "autofeatures" / self.run_name
