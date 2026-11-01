"""Statistics for judge self-agreement.

Everything here works on plain numpy arrays so it can be tested without
an API key or a database.

Terms used throughout:
- an item is one row of the eval (one input and one output to judge);
- a repeat is one independent judge call on that item;
- a verdict is PASS/FAIL (labels) or 1 to 5 (scores).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binomtest

# ---------------------------------------------------------------------------
# Agreement


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa from an items x categories matrix of counts.

    Rows may have different totals (each row needs at least 2 ratings), which
    uses the usual generalisation: per-item agreement P_i is computed with that
    row's own total, and category shares are pooled over all ratings.

    Returns nan when agreement by chance is 1 (every rating in one category),
    because kappa is undefined there.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("counts must be a 2-D items x categories matrix")
    n_i = counts.sum(axis=1)
    keep = n_i >= 2
    counts, n_i = counts[keep], n_i[keep]
    if counts.shape[0] == 0:
        return float("nan")
    p_i = ((counts**2).sum(axis=1) - n_i) / (n_i * (n_i - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / n_i.sum()
    p_e = float((p_j**2).sum())
    if np.isclose(p_e, 1.0):
        return float("nan")
    return float((p_bar - p_e) / (1.0 - p_e))


@dataclass
class ICC:
    single: float  # ICC(1,1): agreement of one judge call with another
    average: float  # ICC(1,k): reliability of the mean of k calls
    n_items: int
    k: int


def icc1(scores: np.ndarray) -> ICC:
    """One-way random-effects ICC (Shrout and Fleiss ICC(1,1) and ICC(1,k)).

    Repeats of the same judge are exchangeable, not distinct raters, so the
    one-way model is the right one: items are random, repeats are random.
    Rows with any missing value (nan) are dropped.
    """
    x = np.asarray(scores, dtype=float)
    if x.ndim != 2:
        raise ValueError("scores must be a 2-D items x repeats matrix")
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        return ICC(float("nan"), float("nan"), n, k)
    grand = x.mean()
    row_means = x.mean(axis=1)
    msb = k * ((row_means - grand) ** 2).sum() / (n - 1)
    msw = ((x - row_means[:, None]) ** 2).sum() / (n * (k - 1))
    if np.isclose(msb + (k - 1) * msw, 0.0):
        return ICC(float("nan"), float("nan"), n, k)
    single = (msb - msw) / (msb + (k - 1) * msw)
    average = (msb - msw) / msb if not np.isclose(msb, 0.0) else float("nan")
    return ICC(float(single), float(average), n, k)


# ---------------------------------------------------------------------------
# Per-item stability


def flip_rate(verdicts: Sequence[Hashable]) -> float:
    """Share of an item's verdicts that disagree with its most common verdict.

    With 5 repeats the possible values are 0, 0.2 and 0.4. Missing verdicts
    (None) are ignored. Returns nan with no valid verdicts.
    """
    valid = [v for v in verdicts if v is not None]
    if not valid:
        return float("nan")
    top = Counter(valid).most_common(1)[0][1]
    return 1.0 - top / len(valid)


def majority(verdicts: Sequence[Hashable]) -> Hashable | None:
    """Most common verdict, or None when the top two are tied or none are valid."""
    valid = [v for v in verdicts if v is not None]
    if not valid:
        return None
    ranked = Counter(valid).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


# ---------------------------------------------------------------------------
# Pass rates and how many repeats are needed


def _quantiles(samples: np.ndarray, level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


@dataclass
class Interval:
    k: int
    estimate: float  # mean pass rate (0 to 1) over the simulated reruns
    low: float
    high: float

    @property
    def width_points(self) -> float:
        return round((self.high - self.low) * 100.0, 9)


def rerun_interval(
    p_hat: np.ndarray,
    k: int,
    *,
    n_boot: int = 4000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Pass rate if you reran the judge on the same items with k calls each.

    Items are held fixed; only judge noise varies. For each item the chance of
    a PASS on one call is its observed share of PASS verdicts (p_hat). Each
    simulated rerun draws k calls per item and takes the majority (k must be
    odd so there are no ties); k=1 is a single run. The interval is the
    middle 95% of the simulated pass rates.
    """
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd number")
    p_hat = np.asarray(p_hat, dtype=float)
    rng = np.random.default_rng(seed)
    passes = rng.binomial(k, p_hat[None, :], size=(n_boot, p_hat.size))
    rates = (passes > k / 2).mean(axis=1)
    lo, hi = _quantiles(rates, level)
    return Interval(k, float(rates.mean()), lo, hi)


def full_interval(
    p_hat: np.ndarray,
    k: int,
    *,
    n_boot: int = 4000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Like rerun_interval, but also resamples the items with replacement.

    This adds the uncertainty from having only these items, which more
    repeats cannot remove.
    """
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd number")
    p_hat = np.asarray(p_hat, dtype=float)
    n = p_hat.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    passes = rng.binomial(k, p_hat[idx])
    rates = (passes > k / 2).mean(axis=1)
    lo, hi = _quantiles(rates, level)
    return Interval(k, float(rates.mean()), lo, hi)


def recommend_k(
    p_hat: np.ndarray,
    *,
    target_points: float = 2.0,
    max_k: int = 51,
    n_boot: int = 4000,
    seed: int = 0,
) -> tuple[int | None, list[Interval]]:
    """Smallest odd k whose rerun interval is at most target_points wide.

    Returns (k or None if no k up to max_k is enough, the intervals tried).
    """
    tried: list[Interval] = []
    for k in range(1, max_k + 1, 2):
        iv = rerun_interval(p_hat, k, n_boot=n_boot, seed=seed + k)
        tried.append(iv)
        if iv.width_points <= target_points + 1e-9:
            return k, tried
    return None, tried


# ---------------------------------------------------------------------------
# Comparing two runs


@dataclass
class McNemar:
    pass_to_fail: int  # PASS in run A, FAIL in run B
    fail_to_pass: int  # FAIL in run A, PASS in run B
    p_value: float


def mcnemar_exact(pass_to_fail: int, fail_to_pass: int) -> McNemar:
    """Exact (binomial) McNemar test on the discordant pairs."""
    n = pass_to_fail + fail_to_pass
    p = 1.0 if n == 0 else float(binomtest(min(pass_to_fail, fail_to_pass), n, 0.5).pvalue)
    return McNemar(pass_to_fail, fail_to_pass, p)


def paired_diff_interval(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 4000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap interval for mean(b) - mean(a) over paired 0/1 item outcomes."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = _quantiles(diffs, level)
    return float(b.mean() - a.mean()), lo, hi


# ---------------------------------------------------------------------------
# One call that summarises a whole run


@dataclass
class ItemStat:
    item_id: str
    counts: dict  # verdict -> count, e.g. {"PASS": 3, "FAIL": 2}
    n_valid: int
    n_invalid: int
    flip_rate: float
    majority: Hashable | None
    pass_share: float  # share of valid verdicts that count as a pass


@dataclass
class RunStats:
    scale: str
    n_items: int
    k: int
    n_calls: int
    n_invalid: int
    kappa: float | None
    icc: ICC | None
    exact_agreement: float  # share of items where every valid verdict agrees
    items_flipped: int  # items with at least one disagreeing verdict
    mean_flip_rate: float
    per_repeat_pass_rates: list[float]
    single: Interval
    majority_k: Interval
    majority_k_full: Interval
    single_full: Interval
    recommended_k: int | None
    target_points: float
    curve: list[Interval]
    items: list[ItemStat] = field(default_factory=list)


LABELS = ("PASS", "FAIL")


def is_pass(verdict, scale: str, pass_threshold: int) -> bool:
    if scale == "pass-fail":
        return verdict == "PASS"
    return int(verdict) >= pass_threshold


def summarise(
    verdicts: dict[str, list],
    *,
    scale: str = "pass-fail",
    pass_threshold: int = 4,
    target_points: float = 2.0,
    max_k: int = 51,
    curve_max_k: int = 15,
    n_boot: int = 4000,
    seed: int = 0,
) -> RunStats:
    """Summarise a run.

    verdicts maps item id -> list of verdicts in repeat order (index 0 is
    repeat 1). A verdict is "PASS"/"FAIL" or an int 1..5, or None when the
    call failed or could not be parsed.
    """
    if scale not in ("pass-fail", "1-5"):
        raise ValueError("scale must be 'pass-fail' or '1-5'")
    ids = list(verdicts)
    k = max((len(v) for v in verdicts.values()), default=0)
    cats = list(LABELS) if scale == "pass-fail" else [1, 2, 3, 4, 5]

    items: list[ItemStat] = []
    counts_matrix = []
    score_rows = []
    for item_id in ids:
        vs = verdicts[item_id]
        valid = [v for v in vs if v is not None]
        c = Counter(valid)
        counts_matrix.append([c.get(cat, 0) for cat in cats])
        if scale == "1-5":
            row = [float(v) if v is not None else np.nan for v in vs]
            row += [np.nan] * (k - len(row))
            score_rows.append(row)
        n_pass = sum(1 for v in valid if is_pass(v, scale, pass_threshold))
        items.append(
            ItemStat(
                item_id=item_id,
                counts={cat: c.get(cat, 0) for cat in cats if c.get(cat, 0)},
                n_valid=len(valid),
                n_invalid=len(vs) - len(valid),
                flip_rate=flip_rate(valid),
                majority=majority(valid),
                pass_share=(n_pass / len(valid)) if valid else float("nan"),
            )
        )

    usable = [it for it in items if it.n_valid > 0]
    p_hat = np.array([it.pass_share for it in usable], dtype=float)
    flips = np.array([it.flip_rate for it in usable], dtype=float)

    kappa = fleiss_kappa(np.array(counts_matrix)) if scale == "pass-fail" and counts_matrix else None
    icc = icc1(np.array(score_rows)) if scale == "1-5" and score_rows else None

    per_repeat = []
    for r in range(k):
        col = [verdicts[i][r] for i in ids if r < len(verdicts[i]) and verdicts[i][r] is not None]
        if col:
            per_repeat.append(sum(is_pass(v, scale, pass_threshold) for v in col) / len(col))

    k_odd = k if k % 2 == 1 else max(k - 1, 1)
    rec, tried = recommend_k(p_hat, target_points=target_points, max_k=max_k, n_boot=n_boot, seed=seed)
    curve_ks = list(range(1, curve_max_k + 1, 2))
    for extra in (k_odd, rec):
        if extra is not None and extra not in curve_ks:
            curve_ks.append(extra)
    curve = [rerun_interval(p_hat, kk, n_boot=n_boot, seed=seed + kk) for kk in sorted(curve_ks)]

    return RunStats(
        scale=scale,
        n_items=len(usable),
        k=k,
        n_calls=sum(len(v) for v in verdicts.values()),
        n_invalid=sum(it.n_invalid for it in items),
        kappa=kappa,
        icc=icc,
        exact_agreement=float((flips == 0).mean()) if flips.size else float("nan"),
        items_flipped=int((flips > 0).sum()),
        mean_flip_rate=float(flips.mean()) if flips.size else float("nan"),
        per_repeat_pass_rates=per_repeat,
        single=rerun_interval(p_hat, 1, n_boot=n_boot, seed=seed + 1),
        majority_k=rerun_interval(p_hat, k_odd, n_boot=n_boot, seed=seed + k_odd),
        majority_k_full=full_interval(p_hat, k_odd, n_boot=n_boot, seed=seed + 1000 + k_odd),
        single_full=full_interval(p_hat, 1, n_boot=n_boot, seed=seed + 1001),
        recommended_k=rec,
        target_points=target_points,
        curve=curve,
        items=items,
    )
