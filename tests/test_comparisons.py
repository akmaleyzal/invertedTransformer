"""Table 6 -- the pair matrix, its FWER control, and the confidence set.

`D56` supplied the models §7 calls mandatory; `D60g` found that nothing ever
called a test on them. These pin the three things that would otherwise go wrong
quietly: the wrong statistic on a nested pair, an uncorrected 45-test matrix, and
a confidence set that keeps everything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from itransformer_btc.comparisons import (
    NAIVE,
    build_panel,
    cluster_bootstrap_t,
    differential,
    is_nested,
    label,
    mcs_table,
    model_confidence_set,
    pair_matrix,
    romano_wolf,
)

#: The 684-run grid output (`D60f`). Repo-root ``artifacts/`` holds one stale
#: 2026-08-06 CPU smoke run and is deliberately not this path.
ARTIFACTS = (
    Path(__file__).resolve().parent.parent / "notebooks" / "outputs" / "artifacts"
)

needs_grid = pytest.mark.skipif(
    not (ARTIFACTS / "preds" / "itr_o01_K08_H024_s42.parquet").exists(),
    reason="grid output not present in this checkout",
)


# -- which statistic (`D29`) -------------------------------------------------


def test_nested_pairs_are_recognised() -> None:
    """The comparisons that carry the paper are nested: the ladder is cumulative,
    so K=1's feature set is a strict subset of K=8's under one architecture, and
    standard DM is not asymptotically N(0,1) there."""
    assert is_nested(("itr", 1), ("itr", 8))
    assert is_nested(("rdg", 1), ("rdg", 8))
    assert is_nested(("itr", 8), NAIVE)
    assert is_nested(NAIVE, ("dlin", 8))


def test_cross_architecture_pairs_are_not_nested() -> None:
    assert not is_nested(("itr", 8), ("dlin", 8))
    assert not is_nested(("itr", 8), ("ptst", 8))
    assert not is_nested(("rdg", 8), ("itr", 8))


def test_the_uniform_attention_arm_nests_against_nothing_along_k() -> None:
    """`D50`: ``itru`` is ``itr`` at K=8 with attention forced uniform. It sees
    exactly the same eight variates, so the arm varies *what attention selects*,
    not what the model can see -- and a K-nesting rule applied to it would put a
    Clark-West adjustment on a pair whose information sets are identical."""
    assert not is_nested(("itru", 8), ("itr", 8))
    assert not is_nested(("itru", 8), ("itr", 1))


# -- multiplicity (`D35`) ----------------------------------------------------


def test_romano_wolf_is_monotone_and_respects_its_own_floor() -> None:
    rng = np.random.default_rng(42)
    per_origin = rng.standard_normal((15, 6)) * 0.01
    per_origin[:, 0] += 0.05  # one pair with a real effect
    p = romano_wolf(per_origin, B=999, seed=7)
    assert p.shape == (6,)
    assert p[0] == p.min()
    assert (p >= 1 / (1 + 999)).all(), "no bootstrap p may sit below its floor (D53d)"
    assert (p <= 1.0).all()


def test_romano_wolf_controls_fwer_under_a_complete_null() -> None:
    """The unadjusted matrix at 20 pairs expects ~1 spurious rejection at
    alpha = 0.05 -- which `D35` says SPA and Reality Check cannot fix, because
    they test a one-against-many null and say nothing about all-pairs
    comparisons."""
    rng = np.random.default_rng(1)
    p = romano_wolf(rng.standard_normal((15, 20)) * 0.01, B=999, seed=3)
    assert (p < 0.05).sum() <= 1


def test_romano_wolf_is_never_smaller_than_the_unadjusted_p() -> None:
    """A stepdown that produced a smaller p than the raw one would be reporting
    multiplicity as a discount rather than a cost."""
    rng = np.random.default_rng(11)
    per_origin = rng.standard_normal((15, 8)) * 0.01
    per_origin[:, 2] += 0.08

    t_obs, t_boot = cluster_bootstrap_t(per_origin, B=999, seed=5)
    raw = np.array(
        [
            (1 + int((np.abs(t_boot[:, j]) >= abs(t_obs[j])).sum())) / (1 + 999)
            for j in range(per_origin.shape[1])
        ]
    )
    adjusted = romano_wolf(per_origin, B=999, seed=5)
    assert (adjusted >= raw - 1e-12).all()


def test_model_confidence_set_drops_a_clearly_worse_model() -> None:
    rng = np.random.default_rng(2)
    losses = rng.standard_normal((15, 4)) * 0.001
    losses[:, 3] += 0.5
    keep = model_confidence_set(losses, alpha=0.10, B=999, seed=5)
    assert 3 not in keep
    assert len(keep) >= 1


def test_model_confidence_set_keeps_everything_when_nothing_separates() -> None:
    rng = np.random.default_rng(4)
    keep = model_confidence_set(rng.standard_normal((15, 5)) * 1e-9, alpha=0.10, B=999)
    assert len(keep) == 5


# -- the panel and the table, on the real artifact ---------------------------


@needs_grid
def test_build_panel_aligns_every_model_on_identical_windows() -> None:
    """`D45`. Verified on the artifact: at origin 1 ``itr``, ``ptst`` and ``rdg``
    hold 88,992 rows over 3,708 timestamps with identical timestamp sets."""
    keys = [("itr", 8), ("ptst", 8), ("rdg", 8), NAIVE]
    panel = build_panel(keys, [ARTIFACTS], origin_indices=(1,))
    assert panel.origin_indices == (1,)
    n = len(panel.y_true[1])
    assert n == 88_992
    for key in keys:
        assert len(panel.y_pred[(key, 1)]) == n
    assert np.isclose(panel.y_pred[(NAIVE, 1)].std(), 0.0), "Naive-RW is constant in z"


@needs_grid
def test_missing_cell_is_a_loud_failure_not_a_short_matrix() -> None:
    with pytest.raises(FileNotFoundError, match="no run at origin"):
        build_panel([("itr", 8), ("dlin", 12)], [ARTIFACTS], origin_indices=(1,))


@needs_grid
def test_nested_differential_carries_the_clark_west_adjustment() -> None:
    """The adjustment term is non-negative by construction, so a nested pair's
    differential must sit at or above its unadjusted twin everywhere."""
    panel = build_panel([("itr", 1), ("itr", 8)], [ARTIFACTS], origin_indices=(1,))
    adjusted = differential(panel, ("itr", 1), ("itr", 8), 1)
    y = panel.y_true[1]
    small = panel.y_pred[(("itr", 1), 1)]
    large = panel.y_pred[(("itr", 8), 1)]
    plain = (np.square(y - small) - np.square(y - large)).reshape(-1, 24).mean(axis=1)
    assert (adjusted >= plain - 1e-12).all()
    assert adjusted.mean() > plain.mean()


@needs_grid
def test_pair_matrix_names_its_statistic_per_pair_and_states_t_and_h() -> None:
    """Root §9.2 requires ``T`` beside every p-value and the statistic named per
    pair. ``T`` counts window starts, so a full block is 720 and never 17,280."""
    keys = [("itr", 1), ("itr", 8), ("ptst", 8), NAIVE]
    panel = build_panel(keys, [ARTIFACTS], origin_indices=(1, 2, 3))
    table = pair_matrix(panel, B=499, seed=13)

    assert table.height == 6
    named = dict(
        zip(
            zip(table.get_column("left"), table.get_column("right")),
            table.get_column("statistic_name"),
        )
    )
    assert named[("itr-K1", "itr-K8")] == "Clark-West"
    # Oriented restricted-model-first, so the key is not the enumeration order.
    assert named[("Naive-RW", "itr-K1")] == "Clark-West"
    assert named[("itr-K8", "ptst-K8")] == "DM-HLN"

    assert (table.get_column("h") == 24).all()
    assert (table.get_column("T_min").to_numpy() <= 720).all()
    assert (table.get_column("T_min").to_numpy() > 100).all()
    assert (table.get_column("n_cells") == 18).all(), "3 origins x 6 blocks"
    assert (table.get_column("G") == 3).all()
    assert (table.get_column("p_romano_wolf").to_numpy() >= 1 / 500).all()


@needs_grid
def test_mcs_table_ranks_by_mean_loss_and_reports_origin_dispersion() -> None:
    """`D30`: a row aggregated across origins carries the SE across origins, not
    the seed standard deviation."""
    keys = [("itr", 8), ("rdg", 8), ("dlin", 8), NAIVE]
    panel = build_panel(keys, [ARTIFACTS], origin_indices=(1, 2, 3))
    table = mcs_table(panel, B=499, seed=17)
    assert table.get_column("rank").to_list() == [1, 2, 3, 4]
    assert (np.diff(table.get_column("mean_loss").to_numpy()) > 0).all()
    assert (table.get_column("se_across_origins").to_numpy() > 0).all()
    assert table.get_column("in_mcs_90").any()


def test_label_names_the_sentinel_readably() -> None:
    assert label(NAIVE) == "Naive-RW"
    assert label(("itr", 8)) == "itr-K8"


# -- orientation, which decides the sign of every nested statistic ------------


def test_nesting_order_puts_the_restricted_model_first() -> None:
    """Clark-West is ``(y - y_small)^2 - (y - y_large)^2 + (y_small - y_large)^2``.
    The adjustment is symmetric and the first two terms are not, so swapping the
    roles reports ``-(first two) + adjustment`` -- not a Clark-West statistic.

    Against Naive-RW the error is loud in the wrong direction: every model would
    return a large positive statistic and appear to beat the baseline while its
    own sample MSE is worse.
    """
    from itransformer_btc.comparisons import nesting_order

    assert nesting_order(("itr", 8), NAIVE) == (NAIVE, ("itr", 8))
    assert nesting_order(NAIVE, ("itr", 8)) == (NAIVE, ("itr", 8))
    assert nesting_order(("itr", 8), ("itr", 1)) == (("itr", 1), ("itr", 8))
    assert nesting_order(("rdg", 1), ("rdg", 8)) == (("rdg", 1), ("rdg", 8))
    assert nesting_order(("itr", 8), ("ptst", 8)) is None
    assert nesting_order(("itr", 8), ("itru", 8)) is None


@needs_grid
def test_differential_refuses_a_misoriented_nested_pair() -> None:
    panel = build_panel([("itr", 8), NAIVE], [ARTIFACTS], origin_indices=(1,))
    with pytest.raises(ValueError, match="nests the other way"):
        differential(panel, ("itr", 8), NAIVE, 1)
    assert differential(panel, NAIVE, ("itr", 8), 1).shape == (3708,)


@needs_grid
def test_pair_matrix_orients_naive_rw_as_the_restricted_model() -> None:
    """Naive-RW is nested inside every model in root §7, so it is always ``left``
    and a positive ``t_cluster`` means the larger model helps."""
    keys = [("itr", 8), ("rdg", 8), NAIVE]
    panel = build_panel(keys, [ARTIFACTS], origin_indices=(1, 2, 3))
    table = pair_matrix(panel, B=499, seed=19)
    against_naive = table.filter(
        (pl_col_eq(table, "left", "Naive-RW")) | (pl_col_eq(table, "right", "Naive-RW"))
    )
    assert against_naive.height == 2
    assert against_naive.get_column("left").to_list() == ["Naive-RW", "Naive-RW"]


def pl_col_eq(frame, column: str, value: str):
    """``frame[column] == value`` as a mask, spelled once for readability."""
    return frame.get_column(column) == value
