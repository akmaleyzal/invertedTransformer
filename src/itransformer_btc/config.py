"""Design constants and the walk-forward origin grid.

Every number here is fixed by ``CLAUDE.md`` and none of it may be tuned. The
module exists so that no magic number is buried in pipeline code (root §16) and
so that a change to the design is a one-line diff with a visible blast radius.

The origin grid is *derived*, never transcribed: transcribing it is how the
13-origin figure survived in four documents after `D26` replaced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

# -- data window (root §4.1) -------------------------------------------------

DATA_START: Final = datetime(2018, 1, 1, tzinfo=timezone.utc)
DATA_END: Final = datetime(2026, 8, 1, tzinfo=timezone.utc)  # EXCLUSIVE

BARS_EXPECTED: Final = 75_216
BARS_ACTUAL: Final = 75_094
MISSING_BARS: Final = 122
GAP_BLOCKS: Final = 27

# -- model geometry (root §6.2) ----------------------------------------------

SEQ_LEN: Final = 96   # L — 4 days of lookback
PRED_LEN: Final = 24  # H — headline horizon

#: A window occupies ``L + H`` consecutive bars, so a break inside a span
#: destroys ``L + H - 1`` *start positions*, not ``L + H``. Root §4.3 turns on
#: this off-by-one: across 30 breaks it is a 30-window difference in the
#: assertion target, the same size as the drift the assertion exists to catch.
WINDOW_SPAN: Final = SEQ_LEN + PRED_LEN         # 120
STARTS_LOST_PER_BREAK: Final = WINDOW_SPAN - 1  # 119

# -- walk-forward protocol (root §8.1) ---------------------------------------

TRAIN_MONTHS: Final = 24      # fixed rolling window, never expanding
VAL_MONTHS: Final = 3         # final 3 months of the training window
TRAIN_SUB_MONTHS: Final = TRAIN_MONTHS - VAL_MONTHS  # 21 — where the scaler is fit
TEST_BLOCKS: Final = 6
BLOCK_DAYS: Final = 30
BLOCK_HOURS: Final = BLOCK_DAYS * 24  # 720 window starts per block

#: 5, not 6. The calendar month a block lands on is ``m0 + s*i + (b-1) mod 12``,
#: so for fixed ``b`` the months visited form a coset of size ``12/gcd(s,12)``.
#: At s=6 that is 2 months, and a significant beta1 becomes observationally
#: equivalent to "February and August are harder" — a bias no post-hoc analysis
#: removes. Only s coprime to 12 fully decouples; 5 maximises the origin count
#: among those. Root §8.1 / `D26`.
ORIGIN_SPACING_MONTHS: Final = 5

FIRST_ORIGIN: Final = datetime(2020, 1, 1, tzinfo=timezone.utc)

# -- variate ladder (root §5.2) ----------------------------------------------

K_LADDER: Final = (1, 4, 8, 12)
SEEDS: Final = (42, 43, 44, 45, 46)

#: Horizons swept in root §10.2's 192-run arm. H=24 is the headline.
HORIZONS: Final = (1, 3, 24, 168)

#: Origins the horizon sweep runs at, **named in advance** (`D48`). Choosing
#: them after the main grid would be origin selection.
SWEEP_ORIGIN_INDICES: Final = (1, 5, 10, 15)

#: Offset of the falsification arm's fresh model (root §8.1). Exactly 90 days,
#: not 3 calendar months: test blocks are 30 **days**, so only 90 days lands the
#: fresh origin precisely on the aged model's block-4 boundary, which is what
#: makes "the *same* calendar blocks 4-6" true rather than approximately true.
FRESH_OFFSET_DAYS: Final = 90


def add_months(when: datetime, months: int) -> datetime:
    """Shift ``when`` by whole calendar months, keeping the day of month.

    Every boundary in this study falls on the first of a month, so the
    day-clamping question a general implementation must answer never arises.
    This raises rather than clamping if it ever does: a silently clamped
    boundary moves a split by a day and no assertion downstream would notice.

    Args:
        when: A timezone-aware datetime on the first of some month.
        months: Whole months to add; may be negative.

    Returns:
        The shifted datetime, same tzinfo and time of day.

    Raises:
        ValueError: If ``when`` is not on the first of a month.
    """
    if when.day != 1:
        raise ValueError(
            f"add_months is only used on month boundaries in this study; got "
            f"day={when.day}. Clamping rules would silently move a split."
        )
    total = when.month - 1 + months
    return when.replace(year=when.year + total // 12, month=total % 12 + 1)


@dataclass(frozen=True, slots=True)
class Origin:
    """One walk-forward origin and every boundary derived from it.

    All boundaries are half-open ``[start, end)``. The origin is both the end of
    validation and the start of testing: a forecaster standing at ``o`` has seen
    everything before ``o`` and nothing after it.
    """

    index: int
    origin: datetime

    @property
    def train_start(self) -> datetime:
        """Start of the 24-month rolling training window."""
        return add_months(self.origin, -TRAIN_MONTHS)

    @property
    def train_sub_end(self) -> datetime:
        """End of the 21-month sub-block; equivalently ``val_start``.

        The scaler is fit on this sub-block and nothing else, and training
        windows are enumerated over it and nothing else (`D25`). The 24-month
        count — ~17,400 windows, ~80 MB — is what you get by training on the
        validation months too, which is `D24`'s leak wearing a sample-count
        disguise.
        """
        return add_months(self.origin, -VAL_MONTHS)

    @property
    def val_start(self) -> datetime:
        return self.train_sub_end

    @property
    def val_end(self) -> datetime:
        return self.origin

    @property
    def test_start(self) -> datetime:
        return self.origin

    @property
    def test_end(self) -> datetime:
        return self.origin + timedelta(days=BLOCK_DAYS * TEST_BLOCKS)

    def block(self, b: int) -> tuple[datetime, datetime]:
        """Half-open bounds of test block ``b``, one-indexed as in the paper."""
        if not 1 <= b <= TEST_BLOCKS:
            raise ValueError(f"block index must be in 1..{TEST_BLOCKS}, got {b}")
        start = self.origin + timedelta(days=BLOCK_DAYS * (b - 1))
        return start, start + timedelta(days=BLOCK_DAYS)

    def blocks(self) -> list[tuple[int, datetime, datetime]]:
        """Every test block as ``(label, start, end)``, label one-indexed.

        The label is carried rather than inferred from position because the
        falsification arm evaluates blocks 4-6 and nothing else: there, the
        first tensor in the tuple is block **4**, and writing it out as block 1
        would silently re-index the arm the comparison depends on.
        """
        return [(b, *self.block(b)) for b in range(1, TEST_BLOCKS + 1)]

    @property
    def label(self) -> str:
        """``YYYY-MM`` — the form used in every table and figure."""
        return self.origin.strftime("%Y-%m")


@dataclass(frozen=True, slots=True)
class FalsificationOrigin:
    """A model trained fresh at ``o_i + 90 days``, scored on blocks 4-6.

    Root §8.1's pre-registered falsification arm, and **the only design in the
    study that identifies decay directly**. If the aged-minus-fresh gap is zero
    while beta1 < 0, then beta1 is calendar, not age — the aged model is not
    decaying, the market simply got harder in months 4-6, and RQ2's headline
    would be an artefact.

    Every training boundary is the base origin's, shifted by the same 90 days,
    so the fresh model trains on a window of **identical duration** to the aged
    one. Re-deriving the window from a 24-month subtraction instead would land
    on 2020-03-31-style dates, where the day-of-month clamping question
    :func:`add_months` refuses to answer silently would arise for the first time
    in this study — and a clamped boundary moves a split by a day with no
    assertion downstream to notice.

    Its validation sub-block overlaps the aged model's test blocks 1-3. That is
    not a leak: this is a *different* model, standing at a later origin, and a
    forecaster there has legitimately seen everything before ``o_i + 90 days``.
    """

    base: Origin
    offset_days: int = FRESH_OFFSET_DAYS

    @property
    def index(self) -> int:
        return self.base.index

    @property
    def _shift(self) -> timedelta:
        return timedelta(days=self.offset_days)

    @property
    def origin(self) -> datetime:
        return self.base.origin + self._shift

    @property
    def train_start(self) -> datetime:
        return self.base.train_start + self._shift

    @property
    def train_sub_end(self) -> datetime:
        return self.base.train_sub_end + self._shift

    @property
    def val_start(self) -> datetime:
        return self.train_sub_end

    @property
    def val_end(self) -> datetime:
        return self.origin

    @property
    def test_start(self) -> datetime:
        return self.origin

    @property
    def test_end(self) -> datetime:
        return self.base.test_end

    def blocks(self) -> list[tuple[int, datetime, datetime]]:
        """The **base** origin's blocks 4-6, keeping their original labels.

        ``o_i + 90 days`` is exactly where base block 4 opens, so these are the
        same calendar hours the aged model was scored on — which is the entire
        content of the comparison.
        """
        return [(b, *self.base.block(b)) for b in (4, 5, 6)]

    @property
    def label(self) -> str:
        return f"{self.base.label}+{self.offset_days}d"


#: Anything :func:`itransformer_btc.splits.build_origin_tensors` accepts. The
#: two share an interface rather than an inheritance chain because they share no
#: implementation: one derives its boundaries from calendar months, the other by
#: shifting another origin's.
OriginLike = Origin | FalsificationOrigin


def origin_grid(
    first: datetime = FIRST_ORIGIN,
    spacing_months: int = ORIGIN_SPACING_MONTHS,
    data_start: datetime = DATA_START,
    data_end: datetime = DATA_END,
) -> list[Origin]:
    """Derive every origin that fits inside the data window.

    An origin is admissible when its training window starts no earlier than the
    data and its sixth test block ends no later than the data:
    ``o - 24 months >= data_start`` and ``o + 180 days <= data_end``.

    Under the committed constants this yields **15** origins, 2020-01 … 2025-11.
    The count is derived rather than written down so that changing the spacing
    changes the grid instead of leaving a stale integer behind — which is
    exactly what happened to the superseded 13.

    Returns:
        Origins in chronological order, ``index`` one-based.

    Raises:
        ValueError: If the first origin would need data from before the window.
    """
    grid: list[Origin] = []
    candidate = first
    while True:
        origin = Origin(index=len(grid) + 1, origin=candidate)
        if origin.train_start < data_start:
            raise ValueError(
                f"origin {origin.label} needs training data from "
                f"{origin.train_start.date()}, before the data window opens at "
                f"{data_start.date()}"
            )
        if origin.test_end > data_end:
            break
        grid.append(origin)
        candidate = add_months(candidate, spacing_months)
    return grid


#: Materialised once. Import this rather than rebuilding it — a second call with
#: different arguments silently produces a different study.
ORIGINS: Final = origin_grid()


# -- provenance of every algorithm this study runs (root §12, §13.3) ----------


@dataclass(frozen=True, slots=True)
class Upstream:
    """Where one algorithm in this repository came from, and what was changed.

    Root §1's deliverable is a manuscript, and a manuscript that runs a named
    architecture is asked where its code came from. Author-year alone does not
    answer that: `D16` shows this project already carries mis-dated references
    assembled from memory, and an examiner comparing a cell against an upstream
    release needs the *repository*, the *licence* and the list of deliberate
    departures, not a surname.

    The convention is the one the LTSF literature uses when it reports a
    baseline it did not write: cite the paper by venue, footnote the official
    code URL with an access date, and state in one clause whether the code was
    **used**, **adapted** or **reimplemented**. That third distinction is the
    load-bearing one here, because nothing in ``src/`` is a copy of an upstream
    file --- every model is written from the published description against this
    study's own tensor contract, which is a stronger claim than a fork and a
    checkable one.

    Attributes:
        component: The names in this package the row accounts for.
        module: The ``src/itransformer_btc`` file they live in. The binding test
            reads this, so a row can never describe a module it is not in.
        status: ``reimplemented`` (written here from the published description),
            ``library`` (imported and called), or ``own`` (no upstream code
            exists --- the algorithm is defined in a paper and implemented here).
        reference: IEEE-style, so the row lifts straight into the bibliography.
        repo: Official implementation, empty when the algorithm has none.
        licence: Upstream licence. Empty when there is no upstream code.
        accessed: ISO date ``repo`` was last opened and confirmed.
        adapted: Every deliberate departure, with the divergence ID that forced
            it. Empty only where nothing was changed.
        verified: True when the repository page or an indexing service was read
            and the fields above were taken from it. False means root §13.3
            still owes this row a check --- recorded rather than assumed,
            because `D16` is what silence buys.
    """

    component: str
    module: str
    status: str
    reference: str
    repo: str = ""
    licence: str = ""
    accessed: str = ""
    adapted: str = ""
    verified: bool = False


#: Read as a table by the notebook's provenance cell and asserted against the
#: module docstrings by ``tests/test_provenance.py`` --- two copies that must
#: agree, with something checking that they do (`D54a`, `D69`).
SOURCE_PROVENANCE: Final[tuple[Upstream, ...]] = (
    Upstream(
        component="ITransformer, InvertedEmbedding, VariateAttention, EncoderLayer",
        module="model.py",
        status="reimplemented",
        reference=(
            "Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, "
            '"iTransformer: Inverted transformers are effective for time series '
            'forecasting," in Proc. 12th Int. Conf. Learn. Represent. (ICLR), '
            "2024. arXiv:2310.06625."
        ),
        repo="https://github.com/thuml/iTransformer",
        licence="MIT",
        accessed="2026-09-03",
        adapted=(
            "d_model 512 -> 128, because attention here runs over N <= 12 variate "
            "tokens rather than L timesteps and 512 over-parameterises ~14k "
            "training windows (`D25`); loss on the target channel only, where the "
            "reference defaults to all channels, which would make K itself vary "
            "the number of supervised tasks (`D39`); lr halved every 4 epochs "
            "rather than every epoch (`D47`); a runtime `capture` attribute for "
            "the attention maps, deliberately not a config field so a captured "
            "run stays bit-identical (`D62d`). Every other hyperparameter is "
            "adopted unchanged and never tuned (`D38`)."
        ),
        verified=True,
    ),
    Upstream(
        component="DLinear, SeriesDecomposition",
        module="baselines.py",
        status="reimplemented",
        reference=(
            "A. Zeng, M. Chen, L. Zhang, and Q. Xu, "
            '"Are transformers effective for time series forecasting?," in Proc. '
            "37th AAAI Conf. Artif. Intell., 2023, pp. 11121-11128. "
            "arXiv:2205.13504."
        ),
        repo="https://github.com/cure-lab/LTSF-Linear",
        licence="Apache-2.0",
        accessed="2026-09-03",
        adapted=(
            "The published all-channel objective and channel-shared weights are "
            "kept deliberately: trained on the target channel alone this model "
            "would be K=1 wearing a K=8 label (`D40`, `D56`). Its centred moving "
            "average is retained as published and is confined to the 96-bar "
            "lookback, so root §8.3's no-embargo argument is untouched (`D56`)."
        ),
        verified=True,
    ),
    Upstream(
        component="PatchTST",
        module="baselines.py",
        status="reimplemented",
        reference=(
            "Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "
            '"A time series is worth 64 words: Long-term forecasting with '
            'transformers," in Proc. 11th Int. Conf. Learn. Represent. (ICLR), '
            "2023. arXiv:2211.14730."
        ),
        repo="https://github.com/yuqinie98/PatchTST",
        licence="Apache-2.0",
        accessed="2026-09-03",
        adapted=(
            "Reuses this study's own EncoderLayer with iTransformer's d_model, "
            "d_ff, e_layers, n_heads and dropout, so the two models differ in "
            "what a token is and in nothing else --- the cleanest form of the "
            "contrast, and it extends `D38`'s no-tuning posture to the baselines "
            "instead of quietly exempting them. Patch 16 / stride 8 as published."
        ),
        verified=True,
    ),
    Upstream(
        component="use_norm instance normalisation (ITransformer, PatchTST)",
        module="model.py",
        status="reimplemented",
        reference=(
            "T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, "
            '"Reversible instance normalization for accurate time-series '
            'forecasting against distribution shift," in Proc. 10th Int. Conf. '
            "Learn. Represent. (ICLR), 2022."
        ),
        repo="https://github.com/ts-kim/RevIN",
        licence="MIT",
        accessed="2026-09-03",
        adapted=(
            "Normalisation and denormalisation only; the learnable affine "
            "transform is not used, matching iTransformer's own use_norm. Held "
            "True at every rung as a fixed property of the design rather than a "
            "tuning knob (root §6.2), which is what makes the outer "
            "StandardScaler algebraically inert (root §6.3, `D03`). The upstream "
            "README dates the paper ICLR 2021; dblp records conf/iclr/KimKTPCC22, "
            "so 2022 is used here."
        ),
        verified=True,
    ),
    Upstream(
        component="LSTMForecaster",
        module="baselines.py",
        status="library",
        reference=(
            "S. Hochreiter and J. Schmidhuber, "
            '"Long short-term memory," Neural Computation, vol. 9, no. 8, '
            "pp. 1735-1780, 1997."
        ),
        repo="https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html",
        licence="BSD-3-Clause (PyTorch)",
        accessed="2026-09-03",
        adapted=(
            "torch.nn.LSTM is called directly; only the forecasting head around "
            "it is written here. Genuinely multivariate with a target-channel "
            "loss, so unlike DLinear and PatchTST its K=8 means what the ladder's "
            "K means and its best_val_mse is comparable to theirs (`D64`)."
        ),
        verified=False,
    ),
    Upstream(
        component="RidgeForecaster",
        module="baselines.py",
        status="own",
        reference=(
            "A. E. Hoerl and R. W. Kennard, "
            '"Ridge regression: Biased estimation for nonorthogonal problems," '
            "Technometrics, vol. 12, no. 1, pp. 55-67, 1970."
        ),
        adapted=(
            "NOT scikit-learn. The normal equations are solved once in float64 "
            "and the Gram matrix reused across the alpha grid, because at "
            "L*K = 1152 a float32 Gram makes the smallest alphas report noise. "
            "The intercept is fitted by centring and left unpenalised: shrinking "
            "it would shrink the forecast toward r = mu_g, the constant-drift "
            "model `D31` spent a section removing from Naive-RW. alpha is the "
            "only hyperparameter selected anywhere in this study."
        ),
        verified=False,
    ),
    Upstream(
        component="NaiveForecaster (naive-persist, seasonal-naive), Naive-RW",
        module="baselines.py",
        status="own",
        reference=(
            "R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and "
            "Practice, 3rd ed. Melbourne, Australia: OTexts, 2021."
        ),
        adapted=(
            "Closed forms; nothing is trained. Naive-RW is not the last return "
            "but y_raw = 0, mapped into scaler space as y_z = -mu_g/sigma_g so "
            "the EMH baseline does not quietly become a constant-drift model "
            "(`D31`), and it needs no run at all."
        ),
        verified=False,
    ),
    Upstream(
        component="walk-forward with purging (Origin, build_origin_tensors)",
        module="splits.py",
        status="own",
        reference=(
            "M. Lopez de Prado, Advances in Financial Machine Learning. "
            "Hoboken, NJ: Wiley, 2018, ch. 7; L. J. Tashman, "
            '"Out-of-sample tests of forecasting accuracy: An analysis and '
            'review," Int. J. Forecast., vol. 16, no. 4, pp. 437-450, 2000; '
            "C. Bergmeir and J. M. Benitez, "
            '"On the use of cross-validation for time series predictor '
            'evaluation," Information Sciences, vol. 191, pp. 192-213, 2012.'
        ),
        adapted=(
            "Purging adopted; embargo and CPCV deliberately NOT, with the "
            "written arguments in root §8.3 and §8.4 rather than a protocol "
            "element left silently absent (`D15`). The purge runs at BOTH "
            "boundaries, train/validation as well as train/test, which the "
            "source is not read as requiring and which is the one that governs "
            "model selection (`D24`). Origin spacing is 5 months, not 6, "
            "because only a spacing coprime to 12 decouples the block index "
            "from the calendar month (`D26`)."
        ),
        verified=False,
    ),
    Upstream(
        component="Scaler",
        module="splits.py",
        status="own",
        reference=(
            "No upstream publication: the z-score is arithmetic. The row exists "
            "because a reader is entitled to know this is NOT "
            "sklearn.preprocessing.StandardScaler."
        ),
        adapted=(
            "Fitted on the 21-month training sub-block only, at every origin. "
            "Under use_norm=True it cancels algebraically (root §6.3, `D03`), "
            "so what it controls is the reporting scale and the baselines with "
            "no internal normalisation. RobustScaler and MinMaxScaler are "
            "rejected on correctness rather than preference (root §6.3)."
        ),
        verified=False,
    ),
    Upstream(
        component="Adam optimiser and StepLR schedule (train_one)",
        module="train.py",
        status="library",
        reference=(
            "D. P. Kingma and J. Ba, "
            '"Adam: A method for stochastic optimization," in Proc. 3rd Int. '
            "Conf. Learn. Represent. (ICLR), 2015. arXiv:1412.6980."
        ),
        repo="https://docs.pytorch.org/docs/stable/optim.html",
        licence="BSD-3-Clause (PyTorch)",
        accessed="2026-09-03",
        adapted=(
            "torch.optim.Adam at lr = 1e-4, adopted unchanged (`D38`). StepLR "
            "halves every FOUR epochs, not every epoch: per-epoch halving "
            "reaches ~4e-7 by epoch 9, so the 30-epoch budget could never bind "
            "(`D47`). The loop itself takes nothing from a reference "
            "implementation --- there is no Dataset and no DataLoader, the "
            "split is GPU-resident and batching is index-slicing, which is what "
            "puts the grid inside the weekly quota at all (`D19`, `D57`)."
        ),
        verified=False,
    ),
    Upstream(
        component="dm_test, clark_west_test (HLN correction, rectangular LRV)",
        module="metrics.py",
        status="own",
        reference=(
            "F. X. Diebold and R. S. Mariano, "
            '"Comparing predictive accuracy," J. Bus. Econ. Statist., vol. 13, '
            "no. 3, pp. 253-263, 1995; D. Harvey, S. Leybourne, and P. Newbold, "
            '"Testing the equality of prediction mean squared errors," Int. J. '
            "Forecast., vol. 13, no. 2, pp. 281-291, 1997; T. E. Clark and "
            'K. D. West, "Approximately normal tests for equal predictive '
            'accuracy in nested models," J. Econometrics, vol. 138, no. 1, '
            "pp. 291-311, 2007."
        ),
        repo="https://pkg.robjhyndman.com/forecast/reference/dm.test.html",
        licence="GPL-3 (R forecast --- validation target, not a dependency)",
        accessed="2026-09-03",
        adapted=(
            "Written on numpy, not taken from a package. The long-run variance is "
            "the truncated rectangular estimator at lag h-1, NOT Newey-West "
            "Bartlett, which would shrink the lag-22 autocovariance by ~92% and "
            "manufacture optimistic p-values (`D34`) --- statsmodels' cov_hac is "
            "Bartlett by default, which is why it is not used here. Nested pairs "
            "take Clark-West, non-nested pairs DM with the HLN correction "
            "(`D29`)."
        ),
        verified=False,
    ),
    Upstream(
        component="wild cluster restricted bootstrap (WCR) for beta1",
        module="metrics.py",
        status="own",
        reference=(
            "A. C. Cameron, J. B. Gelbach, and D. L. Miller, "
            '"Bootstrap-based improvements for inference with clustered errors," '
            "Rev. Econ. Statist., vol. 90, no. 3, pp. 414-427, 2008; "
            "J. G. MacKinnon, M. O. Nielsen, and M. D. Webb, "
            '"Cluster-robust inference: A guide to empirical practice," '
            "J. Econometrics, vol. 232, no. 2, pp. 272-299, 2023."
        ),
        adapted=(
            "NOT the wildboottest package, which root §9.2 names as the reference "
            "implementation but which this package does not import. Restricted "
            "(the null is imposed when generating samples), bootstrapping the "
            "cluster-robust t rather than beta-hat, B = 99,999, Rademacher and "
            "Webb weights both reported, and p computed as (1 + count)/(1 + B) "
            "because the observed statistic belongs to its own reference "
            "distribution and mean(t* <= t_obs) returned a literal p = 0 "
            "(`D42`, `D53d`)."
        ),
        verified=False,
    ),
    Upstream(
        component="romano_wolf",
        module="comparisons.py",
        status="own",
        reference=(
            "J. P. Romano and M. Wolf, "
            '"Stepwise multiple testing as formalized data snooping," '
            "Econometrica, vol. 73, no. 4, pp. 1237-1282, 2005."
        ),
        adapted=(
            "Written here; it did not exist in this package until `D62a`. Applied "
            "to the all-pairs matrix, where White's Reality Check and Hansen's SPA "
            "do not apply because they test a one-against-many null (`D35`). "
            "Reported in two columns after `D79`: the all-pairs family, and the "
            "declared claim family."
        ),
        verified=False,
    ),
    Upstream(
        component="model_confidence_set, mcs_table",
        module="comparisons.py",
        status="own",
        reference=(
            "P. R. Hansen, A. Lunde, and J. M. Nason, "
            '"The model confidence set," Econometrica, vol. 79, no. 2, '
            "pp. 453-497, 2011."
        ),
        adapted=(
            "Written here (`D62a`). Reported at 90% and 75% as a membership "
            "column inside Table 6 rather than as a table of its own."
        ),
        verified=False,
    ),
    Upstream(
        component="deflated_sharpe",
        module="economics.py",
        status="own",
        reference=(
            "D. H. Bailey and M. Lopez de Prado, "
            '"The deflated Sharpe ratio: Correcting for selection bias, backtest '
            'overfitting, and non-normality," J. Portfolio Manage., vol. 40, '
            "no. 5, pp. 94-107, 2014."
        ),
        adapted=(
            "Computed per origin from that origin's non-overlapping 24-hour "
            "strategy returns and their per-period Sharpe, never the annualised "
            "one, which would inflate it by sqrt(periods per year). N is the "
            "number of configurations evaluated on that origin's own test span, "
            "not the full run total, because DSR counts candidates selected on "
            "the SAME return series; the full total is reported separately as the "
            "development trial count (`D46`)."
        ),
        verified=False,
    ),
    Upstream(
        component="variance_ratio and adf inside efficiency_table",
        module="efficiency.py",
        status="library",
        reference=(
            "A. W. Lo and A. C. MacKinlay, "
            '"Stock market prices do not follow random walks: Evidence from a '
            'simple specification test," Rev. Financial Stud., vol. 1, no. 1, '
            "pp. 41-66, 1988."
        ),
        repo="https://github.com/bashtage/arch",
        licence="NCSA (arch), BSD-3-Clause (statsmodels)",
        accessed="2026-09-03",
        adapted=(
            "arch.unitroot.VarianceRatio and statsmodels.tsa.stattools.adfuller "
            "are called directly. These two and scipy.stats are the ONLY "
            "third-party statistics anywhere in the package, and they sit at root "
            "§16's named boundary --- imported inside the function that needs "
            "them, never at module level."
        ),
        verified=False,
    ),
    Upstream(
        component="hurst_rs",
        module="efficiency.py",
        status="own",
        reference=(
            "H. E. Hurst, "
            '"Long-term storage capacity of reservoirs," Trans. Amer. Soc. Civil '
            "Eng., vol. 116, no. 1, pp. 770-799, 1951."
        ),
        adapted=(
            "Rescaled-range implementation written here; no package provides one "
            "under a licence and an API this project already depends on."
        ),
        verified=False,
    ),
    Upstream(
        component="participation_ratio, stable_rank, lookback_correlation_pr",
        module="keff.py",
        status="own",
        reference=(
            "L. Laloux, P. Cizeau, J.-P. Bouchaud, and M. Potters, "
            '"Noise dressing of financial correlation matrices," Phys. Rev. '
            "Lett., vol. 83, no. 7, pp. 1467-1470, 1999; V. Plerou, "
            "P. Gopikrishnan, B. Rosenow, L. A. N. Amaral, T. Guhr, and "
            'H. E. Stanley, "Random matrix approach to cross correlations in '
            'financial data," Phys. Rev. E, vol. 65, no. 6, 066126, 2002.'
        ),
        adapted=(
            "PR is taken on the CORRELATION matrix, never the covariance one: the "
            "covariance spectrum is not monotone in K and its ordering is a "
            "statement about units, since log_quote_volume's deviations sit two "
            "orders of magnitude above r's (`D53a`, `D53b`, `D81`). Measured per "
            "origin on that origin's own 21-month training sub-block, so RQ1's "
            "regressor never reads the test period (`D44`)."
        ),
        verified=False,
    ),
    Upstream(
        component="parkinson, garman_klass, rogers_satchell (family F2)",
        module="features.py",
        status="own",
        reference=(
            "M. Parkinson, "
            '"The extreme value method for estimating the variance of the rate '
            'of return," J. Business, vol. 53, no. 1, pp. 61-65, 1980; '
            'M. B. Garman and M. J. Klass, "On the estimation of security price '
            'volatilities from historical data," J. Business, vol. 53, no. 1, '
            "pp. 67-78, 1980; L. C. G. Rogers and S. E. Satchell, "
            '"Estimating variance from high, low and closing prices," Ann. Appl. '
            "Probab., vol. 1, no. 4, pp. 504-512, 1991."
        ),
        adapted=(
            "Per-bar, with NO trailing average --- which is what makes the "
            "center=True leakage class structurally unrepresentable and licenses "
            "root §8.3's no-embargo argument (`D13`, `D15`). Rogers-Satchell "
            "vanishes on the 33 shadowless bars in the sample, so it is taken as "
            "log(RS + 1e-9); Parkinson and Garman-Klass need no floor (`D52a`)."
        ),
        verified=False,
    ),
)
