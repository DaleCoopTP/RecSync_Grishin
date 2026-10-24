"""A typed, finite grammar; enumeration never inspects events or labels."""
from dataclasses import asdict, dataclass

WINDOWS = (3, 7, 10, 14, 30, 60, 90, 0)
TRENDS = ((3, 14), (7, 30), (10, 60))
OPERATIONS = ("count", "sum_log_dur", "mean_log_dur", "mean_pct", "max_pct")
ITEM_CATEGORIES = ("genres", "directors", "countries", "content_type")
USER_CATEGORIES = ("age", "income", "sex", "kids_flg")
PATHS = ("user", "item") + tuple("user_" + c for c in ITEM_CATEGORIES) + tuple(
    "item_" + c for c in USER_CATEGORIES)


@dataclass(frozen=True, order=True)
class FeatureSpec:
    path: str
    operation: str
    window: int = 0
    long_window: int = 0

    def __post_init__(self):
        if self.path not in PATHS or self.window not in WINDOWS:
            raise ValueError("Unknown relational path or time window")
        if self.operation not in (*OPERATIONS, "recency", "share", "trend"):
            raise ValueError("Unknown aggregation")
        if self.operation == "recency" and (self.window or self.long_window):
            raise ValueError("Recency is defined over all available history")
        if self.operation == "share" and self.path in ("user", "item"):
            raise ValueError("Share requires a categorical relation")
        if self.operation == "trend":
            if (self.window, self.long_window) not in TRENDS:
                raise ValueError("Unknown trend window pair")
        elif self.long_window:
            raise ValueError("Only a trend has two windows")

    @property
    def name(self):
        window = str(self.window) if self.window else "all"
        suffix = f"{window}_{self.long_window}" if self.operation == "trend" else window
        return f"auto__{self.path}__{self.operation}__{suffix}"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def describe(self):
        interval = "all history before cutoff" if not self.window else f"[cutoff-{self.window}d, cutoff)"
        if self.operation == "trend":
            return f"{self.path}: (count({self.window}d)+1)/(count({self.long_window}d)+1)"
        if self.operation == "share":
            return f"{self.path}: matched count / parent count over {interval}; empty denominator -> 0"
        return f"{self.path}: {self.operation} over {interval}"


def enumerate_features():
    values = [FeatureSpec(p, op, w) for p in PATHS for op in OPERATIONS for w in WINDOWS]
    values += [FeatureSpec(p, "recency") for p in PATHS]
    values += [FeatureSpec(p, "share", w) for p in PATHS[2:] for w in WINDOWS]
    values += [FeatureSpec(p, "trend", a, b) for p in PATHS for a, b in TRENDS]
    return sorted(values, key=lambda f: f.name)


def resolve_specs(names):
    catalog = {f.name: f for f in enumerate_features()}
    if len(names) != len(set(names)) or any(n not in catalog for n in names):
        raise ValueError("Features must be a unique subset of the declared grammar")
    return [catalog[n] for n in names]
