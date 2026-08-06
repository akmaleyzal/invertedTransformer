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
