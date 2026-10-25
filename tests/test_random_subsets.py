import numpy as np

from scripts.random_subsets import draw_sets, rank_summary


def test_random_draws_follow_frozen_rule_with_a_separate_seed_stream():
    eligible = [f"auto__f{n:03d}" for n in range(50)]
    sets = draw_sets(list(reversed(eligible)), 8, 5, 42)
    assert sets == draw_sets(eligible, 8, 5, 42)  # input order does not matter
    assert all(len(s) == len(set(s)) == 8 and set(s) <= set(eligible) for s in sets)
    assert len({tuple(sorted(s)) for s in sets}) == 5
    frozen = np.random.default_rng(42).choice(sorted(eligible), 8, replace=False).tolist()
    assert frozen not in sets
    assert draw_sets(eligible, 8, 3, 42) == sets[:3]  # a longer run extends, never reshuffles


def test_rank_counts_ties_against_the_reference():
    assert rank_summary([.1, .2, .3], .25) == {"rank": 2, "of": 4, "draws_at_least": 1,
                                               "empirical_p_one_sided": .5}
    assert rank_summary([.1, .2, .3], .2)["rank"] == 3
    assert rank_summary([.1, .2], .5) == {"rank": 1, "of": 3, "draws_at_least": 0,
                                          "empirical_p_one_sided": 1 / 3}
