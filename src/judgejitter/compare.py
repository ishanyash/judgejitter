"""Compare two runs on the same items: did the verdicts move?"""

from __future__ import annotations

import numpy as np

from .report import analyse
from .stats import is_pass, mcnemar_exact, paired_diff_interval
from .store import RunData


def compare_runs(a: RunData, b: RunData, *, target_points: float = 2.0, n_boot: int = 4000, seed: int = 0) -> dict:
    """Paired comparison of per-item majority verdicts (run A versus run B)."""
    sa = analyse(a, target_points=target_points, n_boot=n_boot, seed=seed)
    sb = analyse(b, target_points=target_points, n_boot=n_boot, seed=seed)
    ma = {it.item_id: it for it in sa.items}
    mb = {it.item_id: it for it in sb.items}
    shared = [i for i in ma if i in mb]
    only_a = sorted(set(ma) - set(mb))
    only_b = sorted(set(mb) - set(ma))

    def as_pass(run: RunData, verdict) -> bool:
        return is_pass(verdict, run.meta["scale"], run.meta.get("pass_threshold") or 4)

    pa, pb, changed, tied = [], [], [], []
    for item_id in shared:
        va, vb = ma[item_id].majority, mb[item_id].majority
        if va is None or vb is None:
            tied.append(item_id)
            continue
        xa, xb = as_pass(a, va), as_pass(b, vb)
        pa.append(xa)
        pb.append(xb)
        if va != vb:
            changed.append(
                {
                    "id": item_id,
                    "a": str(va),
                    "b": str(vb),
                    "a_counts": {str(k): v for k, v in ma[item_id].counts.items()},
                    "b_counts": {str(k): v for k, v in mb[item_id].counts.items()},
                    "pass_changed": xa != xb,
                }
            )
    pa_arr, pb_arr = np.array(pa, dtype=float), np.array(pb, dtype=float)
    pass_to_fail = int(((pa_arr == 1) & (pb_arr == 0)).sum())
    fail_to_pass = int(((pa_arr == 0) & (pb_arr == 1)).sum())
    test = mcnemar_exact(pass_to_fail, fail_to_pass)
    if pa_arr.size:
        diff, lo, hi = paired_diff_interval(pa_arr, pb_arr, n_boot=n_boot, seed=seed)
    else:
        diff = lo = hi = float("nan")

    def side(run: RunData, st) -> dict:
        return {
            "run_id": run.meta["run_id"],
            "name": run.meta.get("name"),
            "model": run.meta["model"],
            "api_version": run.meta.get("api_version"),
            "k": run.meta["k"],
            "fleiss_kappa": st.kappa,
            "icc_single": st.icc.single if st.icc else None,
            "items_flipped": st.items_flipped,
            "mean_flip_rate": st.mean_flip_rate,
            "majority_pass_rate": float(np.mean(pa_arr if run is a else pb_arr)) if pa_arr.size else None,
        }

    changed.sort(key=lambda c: (not c["pass_changed"], c["id"]))
    return {
        "a": side(a, sa),
        "b": side(b, sb),
        "items_compared": int(pa_arr.size),
        "items_tied_skipped": tied,
        "only_in_a": only_a,
        "only_in_b": only_b,
        "pass_to_fail": pass_to_fail,
        "fail_to_pass": fail_to_pass,
        "mcnemar_p": test.p_value,
        "pass_rate_diff": diff,
        "pass_rate_diff_low": lo,
        "pass_rate_diff_high": hi,
        "changed": changed,
    }
