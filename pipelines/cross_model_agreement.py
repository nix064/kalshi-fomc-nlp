#!/usr/bin/env python3
"""
FT370 Cross-Model Agreement Pipeline
Runs both claude-opus-4-7 and gpt-5.5 on ALL paragraphs
Output: data/hand_labels/llm_labeled_final.csv
Columns: paragraph_id, text, doc_type, date, source_file, seq,
         claude_label, gpt_label, agreement, final_label
"""

import os
import re
import time
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

BASE        = Path.home() / "ft370_project"
LABELS_IN   = BASE / "data/hand_labels/llm_labeled_paragraphs.csv"
FINAL_OUT   = BASE / "data/hand_labels/llm_labeled_final.csv"
AUDIT_OUT   = BASE / "data/hand_labels/llm_final_audit.json"

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an expert in Federal Reserve monetary policy communication.
Classify this paragraph from a Powell speech or press conference.
Label based on the CONCLUSION and forward-looking implication for interest rates:

HAWKISH: Implies rates should go UP or stay high.
  Signs: inflation elevated/persistent, labor market too tight, ongoing increases
  appropriate, sufficiently restrictive, upside risks to inflation, OR negative
  economic outcomes framed as ACCEPTABLE COSTS of fighting inflation.

DOVISH: Implies rates should go DOWN or stay low.
  Signs: inflation declining/near target, labor market softening, downside risks
  to employment rising, recalibrate, normalize, explicit cut language, OR
  negative outcomes framed as RISKS TO AVOID.

NEUTRAL: No clear forward-looking implication, or explicitly balanced.
  Signs: pure data description, balance of risks roughly equal with no qualifier,
  retrospective explanation of past decisions, procedural language.

RULES:
1. Label the CONCLUSION, not the opening setup.
2. However/but near the end signals the real message.
3. Negative outcomes as ACCEPTABLE COSTS = HAWKISH.
4. Negative outcomes as RISKS TO AVOID = DOVISH.
5. Slow the pace of increases = still HAWKISH.
6. Past tense narrative with no forward implication = NEUTRAL.
7. Balance of risks with NO directional qualifier = NEUTRAL.

Return ONLY one word: HAWKISH, DOVISH, or NEUTRAL."""

LABEL_MAP    = {"HAWKISH": -1, "DOVISH": 1, "NEUTRAL": 0}
MAX_TOKENS   = 32


def label_claude(text: str):
    try:
        resp = anthropic_client.messages.create(
            model="claude-opus-4-7",
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}]
        )
        word = re.sub(r'[^A-Z]', '', resp.content[0].text.strip().upper())
        return LABEL_MAP.get(word, None)
    except Exception as e:
        print(f"  Claude error: {e}")
        return None


def label_gpt(text: str):
    # Try Responses API first
    try:
        resp = openai_client.responses.create(
            model="gpt-5.5",
            reasoning={"effort": "medium"},
            instructions=SYSTEM_PROMPT,
            input=text,
            max_output_tokens=MAX_TOKENS
        )
        word = re.sub(r'[^A-Z]', '', resp.output_text.strip().upper())
        return LABEL_MAP.get(word, None)
    except Exception:
        pass

    # Fallback: Chat Completions with correct param name for gpt-5.5
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-5.5",
            max_completion_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text}
            ]
        )
        word = re.sub(r'[^A-Z]', '', resp.choices[0].message.content.strip().upper())
        return LABEL_MAP.get(word, None)
    except Exception as e:
        print(f"  GPT error: {e}")
        return None


def agreement_label(claude_val, gpt_val):
    """
    Returns:
      'full'    — both models agree exactly
      'partial' — one says neutral, other says hawkish/dovish (direction unclear)
      'conflict'— models disagree on direction (hawkish vs dovish)
      'missing' — one or both labels are None
    """
    if claude_val is None or gpt_val is None or pd.isna(claude_val) or pd.isna(gpt_val):
        return 'missing'
    c = int(claude_val)
    g = int(gpt_val)
    if c == g:
        return 'full'
    if 0 in (c, g):
        return 'partial'
    return 'conflict'


def final_label(claude_val, gpt_val, agree):
    """
    Resolves to a single label:
      full    → either model (they match)
      partial → take the non-neutral model's label
      conflict→ neutral (0) — genuine ambiguity, don't force a direction
      missing → whichever is available, else None
    """
    if agree == 'full':
        return int(claude_val)
    if agree == 'partial':
        c = int(claude_val)
        g = int(gpt_val)
        return c if c != 0 else g
    if agree == 'conflict':
        return 0  # resolve to neutral on genuine disagreement
    # missing
    if claude_val is not None and not pd.isna(claude_val):
        return int(claude_val)
    if gpt_val is not None and not pd.isna(gpt_val):
        return int(gpt_val)
    return None


def main():
    print("=" * 60)
    print("FT370 Cross-Model Agreement Pipeline")
    print("Running claude-opus-4-7 + gpt-5.5 on all paragraphs")
    print("=" * 60)

    df = pd.read_csv(LABELS_IN)
    print(f"Loaded {len(df)} paragraphs from {LABELS_IN.name}")
    print(f"Existing label coverage: {df['llm_label'].notna().sum()}/{len(df)}\n")

    # Build output dataframe with clean columns
    results = []

    for i, row in df.iterrows():
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)} paragraphs processed...")

        claude_val = label_claude(row['text'])
        time.sleep(0.15)
        gpt_val    = label_gpt(row['text'])
        time.sleep(0.15)

        agree = agreement_label(claude_val, gpt_val)
        final = final_label(claude_val, gpt_val, agree)

        results.append({
            "paragraph_id": row['paragraph_id'],
            "text":         row['text'],
            "doc_type":     row['doc_type'],
            "date":         row['date'],
            "source_file":  row['source_file'],
            "seq":          row['seq'],
            "claude_label": claude_val,
            "gpt_label":    gpt_val,
            "agreement":    agree,
            "final_label":  final
        })

    out = pd.DataFrame(results)
    out.to_csv(FINAL_OUT, index=False)
    print(f"\n✅ Saved → {FINAL_OUT}")

    # ── Stats ──────────────────────────────────────────────────────────────────
    total = len(out)

    agree_dist = out['agreement'].value_counts().to_dict()
    print(f"\nAgreement distribution (N={total}):")
    for k in ['full', 'partial', 'conflict', 'missing']:
        n = agree_dist.get(k, 0)
        print(f"  {k:10s}: {n:4d}  ({n/total:.1%})")

    full_agree_rate = agree_dist.get('full', 0) / total
    print(f"\nFull agreement rate: {full_agree_rate:.1%}")
    if full_agree_rate < 0.60:
        print("  ⚠️  Below 60% — check prompt or paragraph quality")

    final_dist = out['final_label'].value_counts().to_dict()
    print(f"\nFinal label distribution (−1=Hawkish, 0=Neutral, 1=Dovish):")
    for k in [-1, 0, 1]:
        n = int(final_dist.get(k, 0))
        print(f"  {k:+d} ({['Hawkish','Neutral','Dovish'][k+1]}): {n} ({n/total:.1%})")

    claude_missing = int(out['claude_label'].isna().sum())
    gpt_missing    = int(out['gpt_label'].isna().sum())
    print(f"\nMissing labels — Claude: {claude_missing} | GPT: {gpt_missing}")

    # ── Audit JSON ─────────────────────────────────────────────────────────────
    audit = {
        "total_paragraphs":     total,
        "agreement_distribution": {k: int(v) for k, v in agree_dist.items()},
        "full_agreement_rate":  round(full_agree_rate, 4),
        "final_label_distribution": {
            str(k): int(final_dist.get(k, 0)) for k in [-1, 0, 1]
        },
        "claude_missing":       claude_missing,
        "gpt_missing":          gpt_missing,
        "date_range":           f"{out['date'].min()} to {out['date'].max()}"
    }
    with open(AUDIT_OUT, 'w') as f:
        json.dump(audit, f, indent=2)
    print(f"\nAudit → {AUDIT_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
