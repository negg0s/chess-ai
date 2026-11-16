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


def test_negative_and_extreme_values_return_safe_minimum():
    safe_min = (0.05, 12)
    assert compute_time_allocation(-1.0, 0.0) == safe_min
    assert compute_time_allocation(-100000.0, 0.0) == safe_min
    assert compute_time_allocation(0.0, -50.0) == safe_min
    assert compute_time_allocation(-9999.0, -9999.0) == safe_min
    assert compute_time_allocation(float("-inf"), 0.0) == safe_min
    assert compute_time_allocation(0.0, float("-inf")) == safe_min
    assert compute_time_allocation(float("nan"), 0.0) == safe_min
    assert compute_time_allocation(0.0, float("nan")) == safe_min
    assert compute_time_allocation(None, None) == safe_min
    assert compute_time_allocation("invalid", 0.0) == safe_min
    assert compute_time_allocation(0.0, "bad") == safe_min

