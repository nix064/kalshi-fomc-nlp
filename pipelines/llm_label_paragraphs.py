#!/usr/bin/env python3
"""
FT370 LLM Paragraph Labeling Pipeline
Models: claude-opus-4-7 + gpt-5.5
Output: data/hand_labels/llm_labeled_paragraphs.csv
"""

import os
import json
import time
import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber
from scipy.stats import spearmanr
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE          = Path.home() / "ft370_project"
SPEECH_DIR    = BASE / "data/raw/speeches"
PRESSCONF_DIR = BASE / "data/raw/pressconf"
HANDCODED     = BASE / "data/hand_labels/powell_handcoded.csv"
PARA_OUT      = BASE / "data/hand_labels/corpus_paragraphs.jsonl"
LABELS_OUT    = BASE / "data/hand_labels/llm_labeled_paragraphs.csv"
AUDIT_OUT     = BASE / "data/hand_labels/llm_labeling_audit.json"

# ── Clients ────────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Signal filter ──────────────────────────────────────────────────────────────
# Must contain at least one forward-looking signal keyword
SIGNAL_KW = [
    'inflation', 'labor market', 'employment', 'unemployment',
    'policy rate', 'appropriate', 'restrictive', 'easing', 'tightening',
    'recalibrate', 'normalize', 'putting it together', 'taken together',
    'implications for monetary policy', 'near-term', 'coming months',
    'going forward', 'looking ahead', 'rate increase', 'rate cut',
    'federal funds', 'balance of risks', 'dual mandate', 'price stability',
    'maximum employment', 'policy stance', 'monetary policy'
]

# Any of these → skip the paragraph entirely
SKIP_KW = [
    'framework review', 'consensus statement', 'effective lower bound',
    'five-year review', 'if you are a student', 'as you graduate',
    'commencement', 'great depression', 'word2vec', 'academic literature',
    'thank you for the opportunity', 'honor to speak', 'jackson hole',
    'this symposium', 'distinguished guests', 'it is a pleasure',
    'i am pleased', 'let me begin by thanking', 'i want to thank',
    'working paper', 'the literature', 'in the 1970s', 'in the 1980s',
    'in the 1990s', 'historical episode', 'great inflation',
    'great moderation', 'gold standard', 'bretton woods',
    'president of the federal reserve', 'reserve bank president',
    'vice chair', 'board of governors', 'federal open market committee members',
    'footnote', 'return to text', 'figure 1', 'figure 2', 'figure 3',
    'figure 4', 'figure 5', 'figure 6'
]

MIN_WORDS = 40    # skip very short paragraphs
MAX_WORDS = 300   # skip very long framework/history paragraphs

def is_signal(text: str) -> bool:
    t = text.lower()
    word_count = len(text.split())
    if word_count < MIN_WORDS or word_count > MAX_WORDS:
        return False
    if any(kw in t for kw in SKIP_KW):
        return False
    return any(kw in t for kw in SIGNAL_KW)

# ── HTML parser (speeches) ─────────────────────────────────────────────────────
def parse_speech_html(path: Path) -> list:
    try:
        html = path.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  Read error {path.name}: {e}")
        return []

    soup = BeautifulSoup(html, 'lxml')

    # PRIMARY: id="article"
    article = soup.find('div', {'id': 'article'})
    if article is None:
        # FALLBACK: class-based
        article = soup.find('div', {'class': re.compile(r'col-xs-12.*col-sm-8.*col-md-8')})
    if article is None:
        return []

    # Strip header block (date, title, speaker name, event, share links)
    for h in article.find_all('div', {'class': re.compile(r'heading')}):
        h.decompose()
    # Strip hidden sidebar
    for h in article.find_all('div', {'class': re.compile(r'hidden')}):
        h.decompose()
    # Strip footnote sections
    for h in article.find_all('div', {'class': re.compile(r'footnote')}):
        h.decompose()

    # Get main content div
    content = article.find('div', {'class': re.compile(r'col-xs-12.*col-sm-8.*col-md-8')})
    if content is None:
        content = article

    paras = [p.get_text(separator=' ', strip=True) for p in content.find_all('p')]

    # Stop at commencement/ceremonial boundary
    stop_phrases = [
        'if you are a student', 'as you graduate', 'your career',
        'good luck', 'congratulations', 'class of'
    ]
    filtered = []
    for p in paras:
        if any(phrase in p.lower() for phrase in stop_phrases):
            break
        filtered.append(p)

    return [p for p in filtered if is_signal(p)]


# ── PDF parser (press conferences) ────────────────────────────────────────────
QA_MARKERS = [
    'JOURNALIST:', 'REPORTER:', 'QUESTION:', 'Q.',
    'MS. ', 'MR. ', 'CHAIR POWELL:', 'Thank you.  Questions?',
    'MODERATOR:', 'VICE CHAIR'
]

def parse_pressconf_pdf(path: Path) -> list:
    try:
        with pdfplumber.open(path) as pdf:
            full_text = '\n'.join(
                page.extract_text() or '' for page in pdf.pages
            )
    except Exception as e:
        print(f"  PDF error {path.name}: {e}")
        return []

    raw = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]

    prepared = []
    for p in raw:
        # Stop when Q&A begins
        if any(p.strip().startswith(m) for m in QA_MARKERS):
            break
        # Also stop on "Questions?" line
        if re.search(r'questions\?', p, re.IGNORECASE) and len(p) < 100:
            break
        if is_signal(p):
            prepared.append(p)

    return prepared


# ── Date extraction from filename ─────────────────────────────────────────────
def date_from_filename(name: str) -> str:
    m = re.search(r'(\d{8})', name)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return "0000-00-00"


# ── Build corpus_paragraphs.jsonl ─────────────────────────────────────────────
def build_corpus():
    print("Building paragraph corpus...")
    records = []

    speech_files = sorted(list(SPEECH_DIR.glob("*.htm")) + list(SPEECH_DIR.glob("*.html")))
    print(f"  Found {len(speech_files)} speech files")
    for f in speech_files:
        date = date_from_filename(f.stem)
        paras = parse_speech_html(f)
        for i, p in enumerate(paras, 1):
            records.append({
                "paragraph_id": f"speech_{date.replace('-','')}_{i:02d}",
                "text": p,
                "doc_type": "speech",
                "date": date,
                "source_file": f.name,
                "seq": i
            })

    pdf_files = sorted(PRESSCONF_DIR.glob("*.pdf"))
    print(f"  Found {len(pdf_files)} press conference files")
    for f in pdf_files:
        date = date_from_filename(f.stem)
        paras = parse_pressconf_pdf(f)
        for i, p in enumerate(paras, 1):
            records.append({
                "paragraph_id": f"pressconf_{date.replace('-','')}_{i:02d}",
                "text": p,
                "doc_type": "pressconf",
                "date": date,
                "source_file": f.name,
                "seq": i
            })

    print(f"  Extracted {len(records)} signal paragraphs")

    from collections import Counter
    years = [r['date'][:4] for r in records]
    print(f"  Year distribution: {dict(sorted(Counter(years).items()))}")

    if len(records) > 600:
        print(f"  WARNING: {len(records)} paragraphs is high — check sample below")
        print("  Sample of first 3 paragraphs from 2018/2019 for sanity check:")
        early = [r for r in records if r['date'][:4] in ('2018', '2019')][:3]
        for r in early:
            print(f"    [{r['paragraph_id']}] {r['text'][:120]}...")
        print()

    with open(PARA_OUT, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    return records


# ── System prompt (same for both models) ──────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in Federal Reserve monetary policy communication.
Classify this paragraph from a Powell speech or press conference.

Label based on the CONCLUSION and forward-looking implication for interest rates:

HAWKISH: Implies rates should go UP or stay high.
  Signs: inflation elevated/persistent, labor market too tight, "ongoing increases
  appropriate," "sufficiently restrictive," upside risks to inflation, OR negative
  economic outcomes (recession, unemployment) framed as ACCEPTABLE COSTS of
  fighting inflation — not as risks to avoid.

DOVISH: Implies rates should go DOWN or stay low.
  Signs: inflation declining/near target, labor market softening, downside risks
  to employment rising, "recalibrate," "normalize," explicit cut language, OR
  negative outcomes framed as RISKS TO AVOID.

NEUTRAL: No clear forward-looking implication, or explicitly balanced.
  Signs: pure data description, "balance of risks roughly equal" with no qualifier,
  retrospective explanation of past decisions with no forward signal, procedural.

RULES:
1. Label the CONCLUSION, not the opening setup.
2. "However" / "but" near the end signals the real message — label what follows.
3. Negative outcomes as ACCEPTABLE COSTS = HAWKISH.
4. Negative outcomes as RISKS TO AVOID = DOVISH.
5. "Slow the pace of increases" = still HAWKISH (direction up, speed varies).
6. Past tense policy narrative with no forward implication = NEUTRAL.
7. "Balance of risks" with NO directional qualifier = NEUTRAL.

Return ONLY one word: HAWKISH, DOVISH, or NEUTRAL."""

LABEL_MAP = {"HAWKISH": -1, "DOVISH": 1, "NEUTRAL": 0}
MAX_TOKENS = 32  # well above GPT minimum of 16, enough for one word


# ── Claude labeler ─────────────────────────────────────────────────────────────
def label_claude(text: str) -> int | None:
    try:
        resp = anthropic_client.messages.create(
            model="claude-opus-4-7",
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}]
        )
        word = resp.content[0].text.strip().upper()
        # Handle cases where model adds punctuation
        word = re.sub(r'[^A-Z]', '', word)
        return LABEL_MAP.get(word, None)
    except Exception as e:
        print(f"  Claude error: {e}")
        return None


# ── GPT-5.5 labeler ────────────────────────────────────────────────────────────
def label_gpt(text: str) -> int | None:
    try:
        resp = openai_client.responses.create(
            model="gpt-5.5",
            reasoning={"effort": "low"},
            instructions=SYSTEM_PROMPT,
            input=text,
            max_output_tokens=MAX_TOKENS  # must be >= 16
        )
        word = resp.output_text.strip().upper()
        word = re.sub(r'[^A-Z]', '', word)
        return LABEL_MAP.get(word, None)
    except Exception as e:
        print(f"  GPT error: {e}")
        # Fallback to Chat Completions API if Responses API fails
        try:
            resp2 = openai_client.chat.completions.create(
                model="gpt-5.5",
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text}
                ]
            )
            word = resp2.choices[0].message.content.strip().upper()
            word = re.sub(r'[^A-Z]', '', word)
            return LABEL_MAP.get(word, None)
        except Exception as e2:
            print(f"  GPT fallback error: {e2}")
            return None


# ── Validation against human labels ───────────────────────────────────────────
def validate_against_human():
    print("\nValidating against human-coded labels...")

    if not HANDCODED.exists():
        print(f"  WARNING: {HANDCODED} not found — skipping validation")
        return {
            "claude_agreement": None, "gpt_agreement": None,
            "claude_spearman_r": None, "gpt_spearman_r": None,
            "n_human": 0
        }

    hc = pd.read_csv(HANDCODED)
    texts  = hc['text'].tolist()
    human  = hc['label'].tolist()
    print(f"  Human-coded paragraphs: {len(texts)}")

    claude_labels = []
    gpt_labels    = []

    for i, text in enumerate(texts):
        print(f"  Validating {i+1}/{len(texts)}...", end='\r')
        claude_labels.append(label_claude(text))
        time.sleep(0.2)
        gpt_labels.append(label_gpt(text))
        time.sleep(0.2)

    print()

    # Filter out None labels
    valid_c = [(h, c) for h, c in zip(human, claude_labels) if c is not None]
    valid_g = [(h, g) for h, g in zip(human, gpt_labels)    if g is not None]

    print(f"  Claude returned valid labels: {len(valid_c)}/{len(texts)}")
    print(f"  GPT returned valid labels:    {len(valid_g)}/{len(texts)}")

    if len(valid_c) == 0 or len(valid_g) == 0:
        print("  ERROR: No valid labels returned — check API keys and model names")
        raise SystemExit("API validation failed — check errors above")

    h_c, c_c = zip(*valid_c)
    h_g, c_g = zip(*valid_g)

    claude_agree = sum(h == c for h, c in valid_c) / len(valid_c)
    gpt_agree    = sum(h == g for h, g in valid_g) / len(valid_g)

    r_claude, p_claude = spearmanr(h_c, c_c)
    r_gpt,    p_gpt    = spearmanr(h_g, c_g)

    print(f"\n  Claude  agreement: {claude_agree:.1%}  Spearman r={r_claude:.3f}  p={p_claude:.4f}")
    print(f"  GPT-5.5 agreement: {gpt_agree:.1%}  Spearman r={r_gpt:.3f}  p={p_gpt:.4f}")

    if claude_agree < 0.80 or gpt_agree < 0.80:
        print("\n  ⚠️  WARNING: Agreement below 80% threshold.")
        print("  Type 'yes' to proceed anyway, or Ctrl+C to abort:")
        if input().strip().lower() != 'yes':
            raise SystemExit("Aborted — fix prompt and rerun.")
    else:
        print("  ✅ Validation passed.")

    return {
        "claude_agreement":   round(claude_agree, 4),
        "gpt_agreement":      round(gpt_agree, 4),
        "claude_spearman_r":  round(float(r_claude), 4),
        "claude_spearman_p":  round(float(p_claude), 6),
        "gpt_spearman_r":     round(float(r_gpt), 4),
        "gpt_spearman_p":     round(float(p_gpt), 6),
        "n_human":            len(texts)
    }


# ── Full corpus labeling ───────────────────────────────────────────────────────
def label_corpus(records: list) -> list:
    print(f"\nLabeling {len(records)} paragraphs (even years→Claude, odd→GPT)...")

    results = []
    for i, rec in enumerate(records):
        year = int(rec['date'][:4])
        model = "claude" if year % 2 == 0 else "gpt"

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(records)} paragraphs labeled...")

        if model == "claude":
            label = label_claude(rec['text'])
            time.sleep(0.15)
        else:
            label = label_gpt(rec['text'])
            time.sleep(0.15)

        results.append({**rec, "llm_label": label, "model": model})

    return results


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FT370 LLM Labeling Pipeline")
    print("Models: claude-opus-4-7 + gpt-5.5 (reasoning=low)")
    print("=" * 60)

    # 1. Build corpus
    records = build_corpus()

    if len(records) < 100:
        print(f"\nERROR: Only {len(records)} paragraphs — check parser and file paths")
        raise SystemExit("Too few paragraphs")

    # 2. Validate against human labels
    audit_stats = validate_against_human()

    # 3. Label full corpus
    results = label_corpus(records)

    # 4. Save labeled CSV
    df = pd.DataFrame(results)

    failed = df[df['llm_label'].isna()]
    if len(failed) > 0:
        print(f"\n⚠️  {len(failed)} paragraphs failed to label")
        print(f"   Failed paragraph IDs: {failed['paragraph_id'].tolist()[:10]}")

    df.to_csv(LABELS_OUT, index=False)
    print(f"\n✅ Saved {len(df)} labeled paragraphs → {LABELS_OUT}")

    # Distribution check
    dist = df['llm_label'].value_counts().to_dict()
    pct  = {k: f"{v/len(df):.1%}" for k, v in dist.items()}
    print(f"   Label distribution (−1=Hawkish, 0=Neutral, 1=Dovish):")
    for k, v in sorted(dist.items()):
        label_name = {-1: "Hawkish", 0: "Neutral", 1: "Dovish"}.get(k, "Unknown")
        print(f"     {k:+d} ({label_name}): {v} ({pct[k]})")

    total = len(df[df['llm_label'].notna()])
    for k in [-1, 0, 1]:
        if dist.get(k, 0) / total > 0.65:
            print(f"   ⚠️  Label {k} is over 65% — check parser for section bleed")

    # 5. Save audit JSON
    audit = {
        **audit_stats,
        "total_paragraphs":  len(df),
        "claude_count":      int(len(df[df['model'] == 'claude'])),
        "gpt_count":         int(len(df[df['model'] == 'gpt'])),
        "label_distribution": {str(k): int(v) for k, v in dist.items()},
        "failed_labels":     int(len(failed)),
        "date_range":        f"{df['date'].min()} to {df['date'].max()}"
    }
    with open(AUDIT_OUT, 'w') as f:
        json.dump(audit, f, indent=2)

    print(f"   Audit stats → {AUDIT_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()