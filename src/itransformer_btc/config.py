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

    @property
    def label(self) -> str:
        """``YYYY-MM`` — the form used in every table and figure."""
        return self.origin.strftime("%Y-%m")


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
