#!/usr/bin/env python3
"""
score_corpus.py — LM sentiment scoring of the full FOMC/Powell corpus

Reads:    data/corpus_manifest.csv   (254 documents)
          data/lm_dictionary.csv     (LM 2020 Master Dictionary)
Writes:   data/corpus_scored.parquet

Output schema
─────────────
    doc_id          str   e.g. "powell20181206a"
    doc_type        str   speeches | testimony | statements | minutes | pressconf
    date            str   "YYYY-MM-DD"
    source_file     str   relative path from manifest
    speaker         str   inferred from doc_id / category
    lm_positive     int   non-negated positive word count
    lm_negative     int   negative word count + negation-flipped positives
    lm_uncertainty  int   uncertainty word count
    lm_litigious    int   litigious word count
    lm_constraining int   constraining word count
    lm_modal_strong int   strong modal word count
    lm_modal_weak   int   weak modal word count
    lm_composite    float (lm_negative + lm_uncertainty - lm_positive) / total_words
    total_words     int   token count after clean tokenization
    segment         str   "full" | "prepared" | "qa"

LM composite sign convention
─────────────────────────────
    Higher value  → more negative/uncertain language → more DOVISH signal
    Lower value   → more positive language           → more HAWKISH signal

Press conferences produce TWO rows each: segment="prepared" and segment="qa".
All other doc types produce one row: segment="full".
LM scores the ENTIRE extracted text; the signal keyword filter from the LLM
labeling step is NOT applied here.

Usage
─────
    cd ~/ft370_project
    source venv/bin/activate
    python pipeline/score_corpus.py
"""

import re
import sys
import logging
from pathlib import Path

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # pipeline/ lives inside ft370_project/

MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus_manifest.csv"
LM_DICT_PATH  = PROJECT_ROOT / "data" / "lm_dictionary.csv"
OUTPUT_PATH   = PROJECT_ROOT / "data" / "corpus_scored.parquet"
PDF_DIR       = PROJECT_ROOT / "data" / "raw" / "pressconf"

# ──────────────────────────────────────────────────────────────────────────────
# Logging  (stdout + file)
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "score_corpus.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NEGATIONS        = {"no", "not", "none", "neither", "never", "nobody"}
NEGATION_WINDOW  = 3          # how many tokens to look back before a Positive word
MIN_TOKENS       = 20         # rows with fewer tokens are skipped (broken parse)

# Stop conditions for commencement / off-topic passages in speeches.
# If any paragraph contains one of these phrases, discard it and all remaining
# paragraphs from that document.
SPEECH_STOP_PHRASES = [
    "if you are a student",
    "as you graduate",
    "congratulations to the",
    "commencement",
    "graduating class",
    "your careers ahead",
    "as a student",
    "welcome to the graduating",
]

# Q&A boundary patterns for press conference PDFs.
# A stripped line must MATCH (from the start) one of these regexes.
#
# NOTE: "CHAIR POWELL" is intentionally EXCLUDED.
#   Many Fed press conf PDFs begin the prepared statement with "CHAIR POWELL."
#   including it as a stop condition caused all 64 PDFs to produce 0 text
#   in the prior LLM labeling pipeline. We stop on journalist / reporter /
#   moderator patterns only — Powell's Q&A answers are reached naturally
#   after a journalist line has already triggered the boundary.
QA_BOUNDARY_RE = re.compile(
    r"^(?:"
    r"journalist[\s:.]"          # JOURNALIST: / JOURNALIST.
    r"|reporter[\s:.]"           # REPORTER: / REPORTER.
    r"|moderator[\s:.]"          # MODERATOR: / MODERATOR.
    r"|q\.\s"                    # Q. [question text]
    r"|(?:ms|mr)\.\s+[a-z]"     # MS. SMITH / MR. TIMIRAOS  (name must follow)
    r")",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────────────
# LM Dictionary
# ──────────────────────────────────────────────────────────────────────────────

def load_lm_dict(path: Path) -> dict:
    """
    Load LM 2020 Master Dictionary CSV.
    Returns dict mapping category label → frozenset of lowercase words.

    LM CSV column names (confirmed):
        Word, Negative, Positive, Uncertainty, Litigious,
        Strong_Modal, Weak_Modal, Constraining
    Output keys use the output-schema naming convention:
        Positive, Negative, Uncertainty, Litigious,
        Strong_Modal, Weak_Modal, Constraining
    """
    log.info(f"Loading LM dictionary: {path}")
    df = pd.read_csv(path)

    required_cols = [
        "Word", "Negative", "Positive", "Uncertainty",
        "Litigious", "Strong_Modal", "Weak_Modal", "Constraining",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"LM dictionary missing expected columns: {missing}")

    lm = {}
    for col in ["Positive", "Negative", "Uncertainty", "Litigious",
                "Strong_Modal", "Weak_Modal", "Constraining"]:
        lm[col] = frozenset(df.loc[df[col] > 0, "Word"].str.lower())
        log.info(f"  {col:15s}: {len(lm[col]):,} words")

    return lm


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """
    Lowercase → strip all non-alphabetic characters → split on whitespace.
    No lemmatization: the LM dictionary uses inflected forms.
    """
    clean = re.sub(r"[^a-z\s]", " ", text.lower())
    return clean.split()


# ──────────────────────────────────────────────────────────────────────────────
# LM Scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_tokens(tokens: list, lm: dict) -> dict:
    """
    Score a token list against all LM categories.

    Negation rule (LM standard):
        If one of {no, not, none, neither, never, nobody} appears within
        NEGATION_WINDOW tokens immediately before a Positive word, that word
        is counted as Negative instead of Positive.

    Positive and Negative are treated as mutually exclusive via `elif` to
    prevent a word appearing in both sets from being double-counted.
    All other categories (Uncertainty, Litigious, etc.) are checked
    independently with plain `if` — they are orthogonal to sentiment.

    Returns a dict matching the output schema column names.
    """
    pos = neg = unc = lit = strong_m = weak_m = con = 0

    pos_set  = lm["Positive"]
    neg_set  = lm["Negative"]
    unc_set  = lm["Uncertainty"]
    lit_set  = lm["Litigious"]
    sm_set   = lm["Strong_Modal"]
    wm_set   = lm["Weak_Modal"]
    con_set  = lm["Constraining"]

    for i, tok in enumerate(tokens):

        # ── Sentiment: Positive / Negative (mutually exclusive) ─────────────
        if tok in pos_set:
            # Check negation window: tokens[i-3 : i]  (excludes i itself)
            window_start = max(0, i - NEGATION_WINDOW)
            negated = any(tokens[j] in NEGATIONS for j in range(window_start, i))
            if negated:
                neg += 1    # negated positive flips to negative
            else:
                pos += 1
        elif tok in neg_set:
            neg += 1

        # ── All other categories: independent of sentiment ───────────────────
        if tok in unc_set:
            unc += 1
        if tok in lit_set:
            lit += 1
        if tok in sm_set:
            strong_m += 1
        if tok in wm_set:
            weak_m += 1
        if tok in con_set:
            con += 1

    n = len(tokens)
    composite = (neg + unc - pos) / n if n > 0 else 0.0

    return {
        "lm_positive":     pos,
        "lm_negative":     neg,
        "lm_uncertainty":  unc,
        "lm_litigious":    lit,
        "lm_constraining": con,
        "lm_modal_strong": strong_m,
        "lm_modal_weak":   weak_m,
        "lm_composite":    round(composite, 8),
        "total_words":     n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Speaker Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_speaker(doc_id: str, category: str) -> str:
    """
    Infer the speaker from doc_id and document category.

    Speeches / testimony: extract the leading alphabetic prefix from doc_id.
        "powell20181206a" → "powell"
        "brainard20200101a" → "brainard"

    Press conferences: always "powell" — all Fed press conferences 2018–2026
        were chaired by Jerome Powell.

    Statements / minutes: "fomc" — institutional voice, no single speaker.
    """
    if category in ("speeches", "testimony"):
        m = re.match(r"^([a-z]+)", doc_id.lower())
        return m.group(1) if m else "unknown"
    elif category == "pressconf":
        return "powell"
    else:
        return "fomc"


# ──────────────────────────────────────────────────────────────────────────────
# HTML Parser  (speeches, testimony, statements, minutes)
# ──────────────────────────────────────────────────────────────────────────────

def parse_html(filepath: Path) -> str:
    """
    Extract clean policy text from a Fed website HTML file.

    Selector strategy (confirmed against browser devtools):
        PRIMARY  : div#article
        FALLBACK : div.col-xs-12.col-sm-8.col-md-8
        STRIP    : div.heading.col-xs-12.col-sm-8.col-md-8  (date/title banner)
        STRIP    : div.col-xs-12.col-sm-4.col-md-4.hidden-sm  (sidebar)

    Extracts all <p> tags from the content div and joins with double newlines.
    Stops (discards paragraph and all following) on commencement / off-topic
    language (e.g. Spelman College Dec 2023 speech).

    Returns a single string of clean text, or "" on failure.
    """
    try:
        raw = filepath.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        log.warning(f"  FILE NOT FOUND: {filepath}")
        return ""

    soup = BeautifulSoup(raw, "lxml")

    # Remove sidebar first (before removing heading, to avoid confusion)
    for el in soup.select("div.col-xs-12.col-sm-4.col-md-4.hidden-sm"):
        el.decompose()

    # Remove heading block (has an extra "heading" class; safe to target specifically)
    for el in soup.select("div.heading.col-xs-12.col-sm-8.col-md-8"):
        el.decompose()

    # Try primary selector
    content = soup.select_one("div#article")

    # Fallback: main content column (3-class match; heading already removed above)
    if content is None:
        content = soup.select_one("div.col-xs-12.col-sm-8.col-md-8")

    # Last-resort: use the whole body
    if content is None:
        log.warning(f"  {filepath.name}: no content div found — falling back to full body")
        content = soup.body or soup

    paragraphs = []
    for p_tag in content.find_all("p"):
        text = p_tag.get_text(separator=" ", strip=True)
        if not text:
            continue

        lower = text.lower()
        if any(phrase in lower for phrase in SPEECH_STOP_PHRASES):
            log.debug(f"  {filepath.name}: commencement stop at: \"{text[:80]}\"")
            break

        paragraphs.append(text)

    return "\n\n".join(paragraphs)


# ──────────────────────────────────────────────────────────────────────────────
# PDF Parser  (press conferences only)
# ──────────────────────────────────────────────────────────────────────────────

def parse_pdf(filepath: Path) -> tuple:
    """
    Extract prepared statement and Q&A text from a Fed press conference PDF.

    Strategy:
        1. Extract all page text via pdfplumber, join pages with newline.
        2. Split into lines.
        3. Scan lines for the Q&A boundary (journalist / reporter / moderator
           speaker tag at start of line).
        4. Return (prepared_text, qa_text).

    "CHAIR POWELL" is intentionally NOT a boundary trigger — see QA_BOUNDARY_RE
    comment above.

    Diagnostic warnings are logged if:
        - pdfplumber extracts 0 text
        - prepared segment is < 100 words (boundary may have fired too early)
        - no Q&A boundary is found at all

    Returns ("", "") on pdfplumber failure.
    """
    try:
        with pdfplumber.open(filepath) as pdf:
            page_texts = [pg.extract_text() or "" for pg in pdf.pages]
    except Exception as exc:
        log.error(f"  pdfplumber FAILED on {filepath.name}: {exc}")
        return "", ""

    full_text = "\n".join(page_texts)
    if not full_text.strip():
        log.warning(f"  {filepath.name}: pdfplumber extracted 0 text")
        return "", ""

    lines = full_text.splitlines()

    # Find Q&A boundary
    boundary_idx = None
    for i, line in enumerate(lines):
        if QA_BOUNDARY_RE.match(line.strip()):
            boundary_idx = i
            log.debug(
                f"  {filepath.name}: Q&A boundary at line {i}: "
                f'"{line.strip()[:70]}"'
            )
            break

    if boundary_idx is None:
        # No journalist tag found — entire transcript treated as prepared
        log.info(f"  {filepath.name}: no Q&A boundary found; full text → prepared")
        prepared_text = full_text.strip()
        qa_text = ""
    else:
        prepared_text = "\n".join(lines[:boundary_idx]).strip()
        qa_text       = "\n".join(lines[boundary_idx:]).strip()

    # Diagnostic: short prepared text suggests boundary fired too early
    prep_words = len(prepared_text.split())
    if prep_words < 100:
        boundary_line = (
            lines[boundary_idx].strip()[:70] if boundary_idx is not None else "N/A"
        )
        log.warning(
            f"  {filepath.name}: prepared segment only {prep_words} words "
            f"(boundary line: \"{boundary_line}\")"
        )

    return prepared_text, qa_text


# ──────────────────────────────────────────────────────────────────────────────
# PDF Diagnostic  (run before main loop)
# ──────────────────────────────────────────────────────────────────────────────

def run_pdf_diagnostic(pdf_dir: Path) -> None:
    """
    Attempt text extraction on the first 3 PDFs in pressconf/.
    Logs word counts so we know pdfplumber is functional before the full run.
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        log.warning(f"PDF DIAGNOSTIC: no PDFs found in {pdf_dir}")
        return

    log.info("=" * 60)
    log.info("PDF DIAGNOSTIC (first 3 files)")
    log.info("=" * 60)
    for pdf_path in pdfs[:3]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
            wc = len(text.split())
            status = "OK" if wc > 100 else "WARNING: very short"
            log.info(f"  {pdf_path.name:<45} {wc:>6,} words  [{status}]")
        except Exception as exc:
            log.error(f"  {pdf_path.name:<45} FAILED: {exc}")
    log.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("score_corpus.py  —  start")
    log.info(f"Project root : {PROJECT_ROOT}")
    log.info(f"Output path  : {OUTPUT_PATH}")
    log.info("=" * 60)

    # ── Load manifest ──────────────────────────────────────────────────────
    log.info(f"Loading manifest: {MANIFEST_PATH}")
    if not MANIFEST_PATH.exists():
        log.error(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    manifest = pd.read_csv(MANIFEST_PATH)
    log.info(f"  {len(manifest)} documents")
    log.info(f"  Category counts: {manifest['category'].value_counts().to_dict()}")

    # ── Load LM dictionary ─────────────────────────────────────────────────
    lm = load_lm_dict(LM_DICT_PATH)

    # ── PDF diagnostic ─────────────────────────────────────────────────────
    run_pdf_diagnostic(PDF_DIR)

    # ── Score corpus ───────────────────────────────────────────────────────
    records  = []
    skipped  = 0
    n_docs   = len(manifest)

    for idx, row in manifest.iterrows():
        doc_id   = row["doc_id"]
        doc_type = row["category"]
        date     = row["date"]
        src_file = row["local_path"]
        speaker  = extract_speaker(doc_id, doc_type)
        filepath = PROJECT_ROOT / src_file

        if (idx + 1) % 50 == 0 or (idx + 1) == n_docs:
            log.info(f"  Progress: {idx + 1}/{n_docs}")

        # ── Press conference PDFs → two segments ──────────────────────────
        if doc_type == "pressconf":
            prepared_text, qa_text = parse_pdf(filepath)
            for segment, text in [("prepared", prepared_text), ("qa", qa_text)]:
                tokens = tokenize(text)
                if len(tokens) < MIN_TOKENS:
                    if text.strip():   # only warn if parse produced something
                        log.warning(
                            f"  {doc_id} [{segment}]: {len(tokens)} tokens < "
                            f"{MIN_TOKENS} minimum — skipping"
                        )
                    skipped += 1
                    continue
                scores = score_tokens(tokens, lm)
                records.append({
                    "doc_id":      doc_id,
                    "doc_type":    doc_type,
                    "date":        date,
                    "source_file": src_file,
                    "speaker":     speaker,
                    "segment":     segment,
                    **scores,
                })

        # ── HTML documents → single segment ───────────────────────────────
        else:
            text   = parse_html(filepath)
            tokens = tokenize(text)
            if len(tokens) < MIN_TOKENS:
                log.warning(
                    f"  {doc_id}: {len(tokens)} tokens < {MIN_TOKENS} minimum "
                    f"— skipping (parse may have failed)"
                )
                skipped += 1
                continue
            scores = score_tokens(tokens, lm)
            records.append({
                "doc_id":      doc_id,
                "doc_type":    doc_type,
                "date":        date,
                "source_file": src_file,
                "speaker":     speaker,
                "segment":     "full",
                **scores,
            })

    log.info(f"Scored {len(records)} segments | Skipped {skipped} segments")

    if not records:
        log.error("No records produced — check parser output above. Aborting.")
        sys.exit(1)

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    # Enforce exact output column order from spec
    col_order = [
        "doc_id", "doc_type", "date", "source_file", "speaker",
        "lm_positive", "lm_negative", "lm_uncertainty", "lm_litigious",
        "lm_constraining", "lm_modal_strong", "lm_modal_weak",
        "lm_composite", "total_words", "segment",
    ]
    df = df[col_order]

    # ── Sanity checks ──────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("OUTPUT SUMMARY")
    log.info("=" * 60)
    log.info(f"Total rows    : {len(df)}")
    log.info(f"\nBy doc_type:\n{df['doc_type'].value_counts().to_string()}")
    log.info(f"\nBy segment:\n{df['segment'].value_counts().to_string()}")
    log.info(
        f"\nlm_composite  — "
        f"mean={df['lm_composite'].mean():.4f}  "
        f"std={df['lm_composite'].std():.4f}  "
        f"min={df['lm_composite'].min():.4f}  "
        f"max={df['lm_composite'].max():.4f}"
    )
    log.info(
        f"total_words   — "
        f"mean={df['total_words'].mean():.0f}  "
        f"min={df['total_words'].min()}  "
        f"max={df['total_words'].max()}"
    )

    # Flag any doc_types with suspiciously few rows (may indicate parser failure)
    expected_min = {"speeches": 50, "testimony": 15, "statements": 30,
                    "minutes": 30, "pressconf": 40}
    for dtype, min_rows in expected_min.items():
        actual = (df["doc_type"] == dtype).sum()
        if actual < min_rows:
            log.warning(
                f"  SANITY: {dtype} has only {actual} rows (expected >= {min_rows}) "
                f"— check parser"
            )

    # Check lm_composite is bounded in a reasonable range
    extreme = df[df["lm_composite"].abs() > 0.3]
    if not extreme.empty:
        log.warning(
            f"  SANITY: {len(extreme)} rows with |lm_composite| > 0.3 "
            f"(may be very short documents)"
        )
        log.warning(f"\n{extreme[['doc_id','total_words','lm_composite']].to_string()}")

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"\nSaved → {OUTPUT_PATH}")
    log.info("=" * 60)
    log.info("score_corpus.py  —  complete")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
