from lichess_bot import compute_time_allocation


def test_low_time_fast_shallow():
    assert compute_time_allocation(3.0, 0.0) == (0.05, 12)


def test_mid_time():
    assert compute_time_allocation(20.0, 0.0) == (0.5, 18)


def test_high_time_caps_budget():
    alloc, depth = compute_time_allocation(600.0, 0.0)
    assert alloc == 2.0
    assert depth == 20


def test_increment_folds_into_budget():
    big_alloc, big_depth = compute_time_allocation(12.0, 0.0)
    incremented = compute_time_allocation(12.0, 2.0)
    # with increment the bot can afford a strictly larger budget tier
    assert incremented[0] >= big_alloc
    # but a tiny clock (<5s even with increment) stays emergency-fast
    assert compute_time_allocation(1.0, 2.0)[1] == 12
