import pandas as pd
import pytest


@pytest.fixture
def tables():
    rows = [(1, 10, "2021-07-01"), (1, 11, "2021-07-18"), (1, 12, "2021-07-19"),
            (1, 13, "2021-07-26"), (2, 10, "2021-07-02"), (2, 12, "2021-07-10"),
            (2, 14, "2021-07-20"), (3, 15, "2021-07-20"), (4, 11, "2021-07-05")]
    events = pd.DataFrame(rows, columns=["user_id", "item_id", "last_watch_dt"])
    events["total_dur"] = 600.0
    events["watched_pct"] = 50.0
    events["last_watch_dt"] = pd.to_datetime(events.last_watch_dt)
    events["day"] = (events.last_watch_dt - pd.Timestamp("1970-01-01")).dt.days.astype("int32")
    users = pd.DataFrame({"user_id": [1, 2, 4], "age": ["young", "old", None],
                          "income": ["low", "high", None], "sex": ["F", "M", None], "kids_flg": [0, 1, 0]})
    items = pd.DataFrame({"item_id": list(range(10, 16)), "content_type": ["film"] * 6,
                         "genres": ["comedy, drama", "comedy", "drama", "drama", "comedy", "thriller"],
                         "countries": ["RU"] * 6, "directors": ["A", "B", "A", "C", "B", "D"],
                         "release_year": [2020] * 6, "for_kids": [None] * 6, "age_rating": [16] * 6})
    return events, users, items
