import math

import numpy as np
import pytest

from judgejitter import stats


def test_fleiss_kappa_matches_fleiss_1971_example():
    # The 10 items x 14 raters x 5 categories example used on Wikipedia.
    counts = np.array(
        [
            [0, 0, 0, 0, 14],
            [0, 2, 6, 4, 2],
            [0, 0, 3, 5, 6],
            [0, 3, 9, 2, 0],
            [2, 2, 8, 1, 1],
            [7, 7, 0, 0, 0],
            [3, 2, 6, 3, 0],
            [2, 5, 3, 2, 2],
            [6, 5, 2, 1, 0],
            [0, 2, 2, 3, 7],
        ]
    )
    # P-bar = 0.378022, P-e = 0.212755 (worked by hand), kappa = 0.20993
    assert stats.fleiss_kappa(counts) == pytest.approx(0.209931, abs=1e-6)


def test_fleiss_kappa_hand_worked_two_categories():
    # 4 items, 3 repeats each. Rows are [PASS, FAIL] counts.
    counts = np.array([[3, 0], [0, 3], [2, 1], [1, 2]])
    # P_i = 1, 1, 1/3, 1/3 -> P-bar = 2/3
    # p_PASS = 6/12, p_FAIL = 6/12 -> P-e = 1/2
    # kappa = (2/3 - 1/2) / (1/2) = 1/3
    assert stats.fleiss_kappa(counts) == pytest.approx(1 / 3)


def test_fleiss_kappa_perfect_and_undefined():
    assert stats.fleiss_kappa(np.array([[5, 0], [0, 5], [5, 0]])) == pytest.approx(1.0)
    # Every verdict PASS: chance agreement is 1, so kappa is undefined.
    assert math.isnan(stats.fleiss_kappa(np.array([[5, 0], [5, 0]])))


def test_fleiss_kappa_uneven_rows_equals_even_when_rows_are_even():
    even = np.array([[4, 1], [2, 3], [5, 0], [1, 4]])
    assert stats.fleiss_kappa(even) == pytest.approx(stats.fleiss_kappa(even.astype(float)))
    # A row with fewer than 2 ratings is ignored.
    with_short = np.vstack([even, [[1, 0]]])
    assert stats.fleiss_kappa(with_short) == pytest.approx(stats.fleiss_kappa(even))


def test_icc_matches_shrout_fleiss_1979():
    # Shrout and Fleiss (1979), Table 2: 6 targets x 4 judges.
    x = np.array(
        [
            [9, 2, 5, 8],
            [6, 1, 3, 2],
            [8, 4, 6, 8],
            [7, 1, 2, 6],
            [10, 5, 6, 9],
            [6, 2, 4, 7],
        ]
    )
    icc = stats.icc1(x)
    assert icc.single == pytest.approx(0.17, abs=0.005)  # published ICC(1,1)
    assert icc.average == pytest.approx(0.44, abs=0.005)  # published ICC(1,k)
    assert icc.single == pytest.approx(0.165742, abs=1e-6)


def test_icc_drops_rows_with_missing():
    x = np.array([[1, 1, 1], [5, 5, 4], [3, np.nan, 3], [2, 2, 2]], dtype=float)
    assert stats.icc1(x).n_items == 3


def test_flip_rate_and_majority():
    assert stats.flip_rate(["PASS"] * 5) == 0
    assert stats.flip_rate(["PASS", "PASS", "PASS", "FAIL", "FAIL"]) == pytest.approx(0.4)
    assert stats.flip_rate(["PASS", None, "FAIL", "FAIL"]) == pytest.approx(1 / 3)
    assert math.isnan(stats.flip_rate([None, None]))
    assert stats.majority(["PASS", "FAIL", "PASS"]) == "PASS"
    assert stats.majority(["PASS", "FAIL"]) is None
    assert stats.majority([4, 4, 5, 3]) == 4


def test_rerun_interval_is_zero_when_judge_never_flips():
    p_hat = np.array([1.0] * 30 + [0.0] * 70)
    iv = stats.rerun_interval(p_hat, 1)
    assert iv.estimate == pytest.approx(0.3)
    assert iv.width_points == 0
    k, _ = stats.recommend_k(p_hat)
    assert k == 1


def test_rerun_interval_single_run_matches_binomial():
    # 400 coin-flip items: a single run's pass rate is Binomial(400, .5)/400,
    # so its 95% interval is about 2 * 1.96 * sqrt(.25 / 400) = 9.8 points.
    p_hat = np.full(400, 0.5)
    iv = stats.rerun_interval(p_hat, 1, n_boot=20000, seed=1)
    assert iv.estimate == pytest.approx(0.5, abs=0.002)
    assert iv.width_points == pytest.approx(9.8, abs=0.6)


def test_majority_vote_narrows_interval_for_leaning_items():
    rng = np.random.default_rng(3)
    p_hat = rng.choice([0.2, 0.8], size=300)
    w1 = stats.rerun_interval(p_hat, 1).width_points
    w5 = stats.rerun_interval(p_hat, 5).width_points
    w15 = stats.rerun_interval(p_hat, 15).width_points
    assert w1 > w5 > w15


def test_recommend_k_finds_smallest_odd_k():
    rng = np.random.default_rng(5)
    p_hat = np.concatenate([np.ones(150), np.zeros(130), rng.choice([0.2, 0.4, 0.6, 0.8], 20)])
    k, tried = stats.recommend_k(p_hat, target_points=2.0)
    assert k is not None and k % 2 == 1
    assert tried[-1].width_points <= 2.0
    assert all(iv.width_points > 2.0 for iv in tried[:-1])
    assert stats.recommend_k(p_hat, target_points=100.0)[0] == 1


def test_recommend_k_gives_up_for_coin_flips():
    k, tried = stats.recommend_k(np.full(100, 0.5), target_points=2.0, max_k=9)
    assert k is None
    assert [iv.k for iv in tried] == [1, 3, 5, 7, 9]


def test_rerun_interval_rejects_even_k():
    with pytest.raises(ValueError):
        stats.rerun_interval(np.array([0.5]), 4)


def test_full_interval_is_wider_than_rerun_interval():
    rng = np.random.default_rng(9)
    p_hat = rng.choice([0.0, 1.0, 0.6], size=200)
    assert stats.full_interval(p_hat, 5).width_points > stats.rerun_interval(p_hat, 5).width_points


def test_mcnemar_exact_hand_worked():
    # 10 items went PASS->FAIL, 2 went FAIL->PASS. Two-sided exact p is
    # 2 * P(X <= 2 | n=12, p=.5) = 2 * (1 + 12 + 66) / 4096 = 0.0385742.
    res = stats.mcnemar_exact(10, 2)
    assert res.p_value == pytest.approx(158 / 4096)
    assert stats.mcnemar_exact(0, 0).p_value == 1.0


def test_paired_diff_interval():
    a = np.array([1, 1, 0, 0, 1, 0, 1, 1])
    b = np.array([1, 0, 0, 0, 1, 0, 0, 1])
    diff, lo, hi = stats.paired_diff_interval(a, b)
    assert diff == pytest.approx(-0.25)
    assert lo <= diff <= hi


def test_summarise_labels():
    verdicts = {
        "a": ["PASS"] * 5,
        "b": ["FAIL"] * 5,
        "c": ["PASS", "FAIL", "PASS", "PASS", "FAIL"],
        "d": ["FAIL", "FAIL", None, "FAIL", "PASS"],
    }
    s = stats.summarise(verdicts)
    assert s.n_items == 4 and s.k == 5 and s.n_calls == 20 and s.n_invalid == 1
    assert s.items_flipped == 2
    assert s.exact_agreement == pytest.approx(0.5)
    assert s.mean_flip_rate == pytest.approx((0 + 0 + 0.4 + 0.25) / 4)
    # Repeat 1: PASS, FAIL, PASS, FAIL -> 0.5. Repeat 3: PASS, FAIL, PASS -> 2/3.
    assert s.per_repeat_pass_rates[0] == pytest.approx(0.5)
    assert s.per_repeat_pass_rates[2] == pytest.approx(2 / 3)
    counts = np.array([[5, 0], [0, 5], [3, 2], [1, 3]])
    assert s.kappa == pytest.approx(stats.fleiss_kappa(counts))
    by_id = {it.item_id: it for it in s.items}
    assert by_id["c"].counts == {"PASS": 3, "FAIL": 2}
    assert by_id["d"].majority == "FAIL"


def test_summarise_scores():
    verdicts = {
        "a": [5, 5, 5],
        "b": [1, 2, 1],
        "c": [4, 3, 4],
        "d": [2, 2, 2],
    }
    s = stats.summarise(verdicts, scale="1-5", pass_threshold=4)
    assert s.kappa is None and s.icc is not None
    assert s.icc.single == pytest.approx(stats.icc1(np.array(list(verdicts.values()))).single)
    by_id = {it.item_id: it for it in s.items}
    assert by_id["c"].pass_share == pytest.approx(2 / 3)
    assert s.per_repeat_pass_rates == pytest.approx([0.5, 0.25, 0.5])
