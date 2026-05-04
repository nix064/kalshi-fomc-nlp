#!/usr/bin/env python3
"""
build_signals.py — Per-FOMC meeting LM signal aggregation

Reads:    data/corpus_scored.parquet   (output of score_corpus.py)
          data/fomc_outcomes.csv        (68 FOMC meetings, manually verified)
Writes:   data/fomc_signals.parquet

Output schema
─────────────
    fomc_date          str    YYYY-MM-DD meeting date
    [fomc_outcomes cols]      decision_bps etc. carried through verbatim
    n_docs             int    documents in primary signal window
    total_words        int    total words across primary-window docs
    lm_signal          float  PRIMARY: equal-weighted mean lm_composite, T-42
    lm_signal_t11      float  ROBUSTNESS: equal-weighted, T-11 window
    lm_signal_spkwt    float  ROBUSTNESS: all speakers, Powell=2× weight, T-21
    n_speeches         int    count by doc_type in primary window
    n_testimony        int
    n_statements       int
    n_minutes          int
    n_pressconf        int
    signal_start       str    window start date
    signal_end         str    window end date (always T-1)

Signal definitions (pre-registered, v4 Section 4.1 + 4.4)
───────────────────────────────────────────────────────────
    Primary corpus:  Powell speeches + Powell testimony
                     + FOMC statements + minutes + pressconf (prepared only)
    Primary agg:     equal-weighted mean lm_composite   [PRE-REGISTERED]
    T-42 window:     full inter-meeting period (calendar days T-42 to T-1)
    T-11 robustness: same corpus, window [T-11, T-1]    [ROBUSTNESS]
    Speaker-wt:      all speakers, Powell weight=2.0,
                     others weight=1.0, T-42 window     [ROBUSTNESS]

    "qa" segments are excluded from all three signals.
    Higher lm_signal → more negative/uncertain language → more DOVISH.

Usage
─────
    cd ~/ft370_project
    source venv/bin/activate
    python pipeline/build_signals.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SCORED_PATH   = PROJECT_ROOT / "data" / "corpus_scored.parquet"
OUTCOMES_PATH = PROJECT_ROOT / "data" / "fomc_outcomes.csv"
OUTPUT_PATH   = PROJECT_ROOT / "data" / "fomc_signals.parquet"

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "build_signals.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

WINDOW_T42 = 42      # primary window: full inter-meeting period (~6 weeks)
WINDOW_T11 = 11      # robustness window (outside Fed blackout ≈ T-10)

POWELL_WEIGHT = 2.0
OTHER_WEIGHT  = 1.0

# Institutional doc types: always included regardless of speaker
INSTITUTIONAL_TYPES = frozenset({"statements", "minutes", "pressconf"})

# Only score these segments (exclude Q&A — Powell hedges deliberately there)
VALID_SEGMENTS = frozenset({"full", "prepared"})

# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_scored(path: Path) -> pd.DataFrame:
    """
    Load corpus_scored.parquet, convert dates, drop Q&A segments.
    Returns the full scored corpus (all speakers) ready for window queries.
    """
    log.info(f"Loading corpus_scored: {path}")
    if not path.exists():
        log.error(f"File not found: {path}  (run score_corpus.py first)")
        sys.exit(1)

    df = pd.read_parquet(path)
    log.info(f"  {len(df)} rows loaded")

    df["date"] = pd.to_datetime(df["date"])

    # Drop Q&A segments before any further processing
    n_before = len(df)
    df = df[df["segment"].isin(VALID_SEGMENTS)].copy()
    log.info(
        f"  After dropping 'qa' segments: {len(df)} rows "
        f"(removed {n_before - len(df)})"
    )

    log.info(f"  doc_type counts: {df['doc_type'].value_counts().to_dict()}")
    log.info(f"  speaker counts:  {df['speaker'].value_counts().to_dict()}")
    log.info(
        f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


def load_outcomes(path: Path) -> pd.DataFrame:
    """
    Load fomc_outcomes.csv. Detects and normalises the date column to 'date'.
    Sorts ascending by date.
    """
    log.info(f"Loading fomc_outcomes: {path}")
    if not path.exists():
        log.error(f"File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info(f"  {len(df)} meetings, columns: {list(df.columns)}")

    # Detect date column (handle common naming variants)
    date_col = None
    for candidate in ["date", "meeting_date", "fomc_date", "Date", "Meeting_Date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError(
            f"Cannot find date column in fomc_outcomes.csv. "
            f"Columns present: {list(df.columns)}"
        )

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
        log.info(f"  Renamed '{date_col}' → 'date'")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}"
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Primary corpus filter
# ──────────────────────────────────────────────────────────────────────────────

def apply_primary_filter(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Primary signal corpus (pre-registered, v4 Section 4.1):
        - Powell speeches (speaker == 'powell', doc_type == 'speeches')
        - Powell testimony (speaker == 'powell', doc_type == 'testimony')
        - All FOMC statements  (institutional — any speaker tag)
        - All FOMC minutes     (institutional)
        - All press conferences (institutional, prepared segment already enforced)

    Note: other Fed governors' speeches are excluded from primary signal;
    they contribute to the speaker-weighted robustness signal instead.
    """
    mask = (
        scored["doc_type"].isin(INSTITUTIONAL_TYPES)
        | (
            scored["doc_type"].isin({"speeches", "testimony"})
            & (scored["speaker"] == "powell")
        )
    )
    result = scored[mask].copy()
    log.info(
        f"Primary corpus: {len(result)} docs "
        f"({len(scored) - len(result)} non-Powell speeches excluded)"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Signal computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def equal_weighted_signal(docs: pd.DataFrame) -> float:
    """
    Pre-registered primary aggregation: equal-weighted mean lm_composite.
    Each document contributes equally regardless of length or type.
    Returns np.nan if no documents in window.
    """
    if docs.empty:
        return np.nan
    return float(docs["lm_composite"].mean())


def speaker_weighted_signal(docs: pd.DataFrame) -> float:
    """
    Robustness aggregation: weighted mean lm_composite with
    Powell = 2.0, all others = 1.0 (v4 Section 4.4).
    Uses ALL speakers (not just Powell) — this is the robustness extension.
    Returns np.nan if no documents in window.
    """
    if docs.empty:
        return np.nan
    weights = docs["speaker"].apply(
        lambda s: POWELL_WEIGHT if s == "powell" else OTHER_WEIGHT
    )
    return float(np.average(docs["lm_composite"], weights=weights))


# ──────────────────────────────────────────────────────────────────────────────
# Per-meeting window computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_meeting_signal(
    fomc_date: pd.Timestamp,
    primary_docs: pd.DataFrame,
    all_docs: pd.DataFrame,
) -> dict:
    """
    Compute all signal values for a single FOMC meeting.

    primary_docs : pre-filtered primary corpus (Powell + institutional)
    all_docs     : full corpus including all speakers (for spkwt robustness)

    Window conventions:
        T-1  = day before meeting (lock-in date, inclusive)
        T-42 = 42 calendar days before meeting (full inter-meeting period)
        T-11 = 11 calendar days before meeting (robustness, outside blackout)
    """
    window_end = fomc_date - pd.Timedelta(days=1)

    t42_start = fomc_date - pd.Timedelta(days=WINDOW_T42)
    t11_start = fomc_date - pd.Timedelta(days=WINDOW_T11)

    def in_window(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
        return df[(df["date"] >= start) & (df["date"] <= window_end)]

    # Primary window docs
    primary_t42 = in_window(primary_docs, t42_start)
    primary_t11 = in_window(primary_docs, t11_start)

    # All-speaker docs for speaker-weighted robustness
    all_t42 = in_window(all_docs, t42_start)

    # Doc type breakdown within primary T-42 window
    type_counts = primary_t42["doc_type"].value_counts()

    return {
        "fomc_date":       fomc_date.strftime("%Y-%m-%d"),
        "signal_start":    t42_start.strftime("%Y-%m-%d"),
        "signal_end":      window_end.strftime("%Y-%m-%d"),
        "n_docs":          len(primary_t42),
        "total_words":     int(primary_t42["total_words"].sum()),
        # ── Signals ──
        "lm_signal":       equal_weighted_signal(primary_t42),
        "lm_signal_t11":   equal_weighted_signal(primary_t11),
        "lm_signal_spkwt": speaker_weighted_signal(all_t42),
        # ── Doc type breakdown ──
        "n_speeches":      int(type_counts.get("speeches",   0)),
        "n_testimony":     int(type_counts.get("testimony",  0)),
        "n_statements":    int(type_counts.get("statements", 0)),
        "n_minutes":       int(type_counts.get("minutes",    0)),
        "n_pressconf":     int(type_counts.get("pressconf",  0)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("build_signals.py  —  start")
    log.info(f"Primary window : T-{WINDOW_T42} to T-1 (full inter-meeting period)")
    log.info(f"T-11 robustness: T-{WINDOW_T11} to T-1")
    log.info("=" * 60)

    # ── Load ───────────────────────────────────────────────────────────────
    scored   = load_scored(SCORED_PATH)
    outcomes = load_outcomes(OUTCOMES_PATH)

    # ── Build primary corpus ───────────────────────────────────────────────
    primary = apply_primary_filter(scored)

    # ── Compute per-meeting signals ────────────────────────────────────────
    log.info(f"\nProcessing {len(outcomes)} FOMC meetings...")

    records = []
    for _, meeting_row in outcomes.iterrows():
        fomc_date = meeting_row["date"]

        window_data = compute_meeting_signal(fomc_date, primary, scored)

        # Carry through all fomc_outcomes columns except 'date'
        # (fomc_date string from window_data is the canonical date column)
        outcome_passthrough = {
            col: meeting_row[col]
            for col in outcomes.columns
            if col != "date"
        }

        records.append({**outcome_passthrough, **window_data})

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    # Column order: fomc_date → outcome cols → signal cols
    outcome_cols  = [c for c in outcomes.columns if c != "date"]
    signal_cols   = [
        "n_docs", "total_words",
        "lm_signal", "lm_signal_t11", "lm_signal_spkwt",
        "n_speeches", "n_testimony", "n_statements", "n_minutes", "n_pressconf",
        "signal_start", "signal_end",
    ]
    col_order = ["fomc_date"] + outcome_cols + signal_cols
    # Filter to columns that actually exist (guards against fomc_outcomes variants)
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # ── Sanity checks ──────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("OUTPUT SUMMARY")
    log.info("=" * 60)
    log.info(f"Total rows : {len(df)}")

    # Empty windows
    empty = df[df["n_docs"] == 0]
    if not empty.empty:
        log.warning(
            f"  {len(empty)} meetings with 0 documents in primary window:\n"
            + empty[["fomc_date", "signal_start", "signal_end"]].to_string()
        )
    else:
        log.info("  No empty windows ✓")

    # NaN signals (should match empty windows)
    nan_count = df["lm_signal"].isna().sum()
    if nan_count > 0:
        log.warning(f"  {nan_count} NaN lm_signal values")
    else:
        log.info("  No NaN signal values ✓")

    # Signal distribution
    log.info(f"\nlm_signal (primary, T-21, equal-weighted):")
    log.info(
        f"  mean={df['lm_signal'].mean():.4f}  "
        f"std={df['lm_signal'].std():.4f}  "
        f"min={df['lm_signal'].min():.4f}  "
        f"max={df['lm_signal'].max():.4f}"
    )
    log.info(f"\nlm_signal_t11 (T-11 robustness):")
    log.info(
        f"  mean={df['lm_signal_t11'].mean():.4f}  "
        f"std={df['lm_signal_t11'].std():.4f}"
    )
    log.info(f"\nlm_signal_spkwt (speaker-weighted robustness):")
    log.info(
        f"  mean={df['lm_signal_spkwt'].mean():.4f}  "
        f"std={df['lm_signal_spkwt'].std():.4f}"
    )

    log.info(
        f"\nDocs per meeting — "
        f"mean={df['n_docs'].mean():.1f}  "
        f"min={df['n_docs'].min()}  "
        f"max={df['n_docs'].max()}"
    )

    # Correlation between primary and robustness signals (should be high)
    corr_t11   = df["lm_signal"].corr(df["lm_signal_t11"])
    corr_spkwt = df["lm_signal"].corr(df["lm_signal_spkwt"])
    log.info(f"\nRobustness correlations with primary signal:")
    log.info(f"  lm_signal vs lm_signal_t11  : r = {corr_t11:.3f}")
    log.info(f"  lm_signal vs lm_signal_spkwt: r = {corr_spkwt:.3f}")

    # Full table preview
    display_cols = [c for c in ["fomc_date", "n_docs", "lm_signal",
                                 "lm_signal_t11", "lm_signal_spkwt"]
                    if c in df.columns]
    log.info(f"\nFull signal table:\n{df[display_cols].to_string()}")

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"\nSaved → {OUTPUT_PATH}")
    log.info("=" * 60)
    log.info("build_signals.py  —  complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
