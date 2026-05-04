#!/usr/bin/env python3
"""
FT370 Cross-Model Agreement Pipeline v2
Reads: data/hand_labels/llm_labeled_final.csv (existing Claude labels)
Output: data/hand_labels/llm_labeled_final_v2.csv

Changes from v1:
- Chat Completions is PRIMARY for GPT (not fallback) — Responses API was unreliable
- Responses API removed entirely — too many silent failures
- Pinned model string gpt-5.5-2026-04-23
- Debug prints first 3 raw GPT responses so you can verify parsing
- No reasoning effort — simpler, faster, more reliable for one-word classification
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

BASE      = Path.home() / "ft370_project"
LABELS_IN = BASE / "data/hand_labels/llm_labeled_final.csv"
FINAL_OUT = BASE / "data/hand_labels/llm_labeled_final_v2.csv"
AUDIT_OUT = BASE / "data/hand_labels/llm_final_v2_audit.json"

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

Return ONLY one word: HAWKISH, DOVISH, or NEUTRAL. No explanation."""

LABEL_MAP  = {"HAWKISH": -1, "DOVISH": 1, "NEUTRAL": 0}
MAX_TOKENS = 64
DEBUG_CALLS = 3   # print raw output for first N GPT calls


def parse_label(raw: str):
    """Extract label from model output. Handles extra text, punctuation, etc."""
    if not raw:
        return None
    clean = re.sub(r'[^A-Z]', '', raw.strip().upper())
    # exact match
    if clean in LABEL_MAP:
        return LABEL_MAP[clean]
    # prefix match for truncated or prefixed outputs
    for key in LABEL_MAP:
        if clean.startswith(key[:4]):
            return LABEL_MAP[key]
    # scan for keyword anywhere in output
    upper = raw.strip().upper()
    for key in LABEL_MAP:
        if key in upper:
            return LABEL_MAP[key]
    return None


def label_claude(text: str):
    for attempt in range(3):
        try:
            resp = anthropic_client.messages.create(
                model="claude-opus-4-7",
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}]
            )
            return parse_label(resp.content[0].text)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Claude error (attempt {attempt+1}): {e} — retry in {wait}s")
            time.sleep(wait)
    return None


_gpt_debug_count = 0

def label_gpt(text: str):
    global _gpt_debug_count

    for attempt in range(4):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-5.5-2026-04-23",   # pinned — avoids alias routing issues
                max_completion_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text}
                ]
            )
            raw = resp.choices[0].message.content or ""

            # Debug first N calls
            if _gpt_debug_count < DEBUG_CALLS:
                print(f"\n  [DEBUG GPT call {_gpt_debug_count+1}] raw='{raw[:80]}'")
                _gpt_debug_count += 1

            label = parse_label(raw)
            if label is not None:
                return label
            # parseable failure — log and retry with stronger instruction
            print(f"  GPT unparseable (attempt {attempt+1}): '{raw[:60]}'")

        except Exception as e:
            code = getattr(e, 'status_code', None)
            if code in (500, 503):
                wait = 2 ** attempt
                print(f"  GPT {code} (attempt {attempt+1}) — retry in {wait}s")
                time.sleep(wait)
                continue
            elif code == 429:
                print(f"  GPT rate limit (attempt {attempt+1}) — wait 20s")
                time.sleep(20)
                continue
            else:
                print(f"  GPT error: {e}")
                return None

    return None


def agreement_label(c, g):
    if c is None or g is None or pd.isna(c) or pd.isna(g):
        return 'missing'
    c, g = int(c), int(g)
    if c == g:
        return 'full'
    if 0 in (c, g):
        return 'partial'
    return 'conflict'


def resolve_final(c, g, agree):
    if agree == 'full':
        return int(c)
    if agree == 'partial':
        c, g = int(c), int(g)
        return c if c != 0 else g
    if agree == 'conflict':
        return 0
    if c is not None and not pd.isna(c):
        return int(c)
    if g is not None and not pd.isna(g):
        return int(g)
    return None


def main():
    print("=" * 60)
    print("FT370 Cross-Model Agreement Pipeline v2")
    print("Claude Opus 4.7 + GPT-5.5 (Chat Completions, pinned version)")
    print("=" * 60)

    df = pd.read_csv(LABELS_IN)
    n  = len(df)
    print(f"Loaded {n} paragraphs")
    print(f"Existing Claude labels: {df['claude_label'].notna().sum()}/{n}")
    print(f"Existing GPT labels:    {df['gpt_label'].notna().sum()}/{n}")
    print(f"\nNote: all rows are doc_type=speech (press conf PDFs filtered out)")
    print(f"First 3 GPT calls will print raw output for debugging\n")

    results      = []
    gpt_failures = 0

    for i, row in df.iterrows():
        if (i + 1) % 50 == 0:
            pct = gpt_failures / (i + 1) * 100
            print(f"  {i+1}/{n} — GPT failure rate: {pct:.1f}%")

        c_label = label_claude(row['text'])
        time.sleep(0.1)
        g_label = label_gpt(row['text'])
        time.sleep(0.2)

        if g_label is None:
            gpt_failures += 1

        agree = agreement_label(c_label, g_label)
        final = resolve_final(c_label, g_label, agree)

        results.append({
            "paragraph_id": row['paragraph_id'],
            "text":         row['text'],
            "doc_type":     row['doc_type'],
            "date":         row['date'],
            "source_file":  row['source_file'],
            "seq":          row['seq'],
            "claude_label": c_label,
            "gpt_label":    g_label,
            "agreement":    agree,
            "final_label":  final
        })

    out = pd.DataFrame(results)
    out.to_csv(FINAL_OUT, index=False)
    print(f"\n✅ Saved {n} rows → {FINAL_OUT}")

    # ── Stats ──────────────────────────────────────────────────────────────────
    agree_dist = out['agreement'].value_counts().to_dict()
    final_dist = out['final_label'].value_counts().to_dict()
    c_missing  = int(out['claude_label'].isna().sum())
    g_missing  = int(out['gpt_label'].isna().sum())
    full_rate  = agree_dist.get('full', 0) / n

    print(f"\nAgreement distribution:")
    for k in ['full', 'partial', 'conflict', 'missing']:
        v = agree_dist.get(k, 0)
        print(f"  {k:10s}: {v:4d}  ({v/n:.1%})")

    print(f"\nFull agreement rate: {full_rate:.1%}")
    if full_rate < 0.60:
        print(f"  ⚠️  Still low — GPT failure rate was {g_missing/n:.1%}")
        print(f"  Final labels fall back to Claude-only for missing GPT rows")

    print(f"\nFinal label distribution (−1=Hawkish, 0=Neutral, 1=Dovish):")
    for k in [-1, 0, 1]:
        v = int(final_dist.get(float(k), final_dist.get(k, 0)))
        name = {-1: "Hawkish", 0: "Neutral", 1: "Dovish"}[k]
        print(f"  {k:+d} ({name}): {v}  ({v/n:.1%})")

    print(f"\nMissing — Claude: {c_missing} | GPT: {g_missing}")

    # Compare GPT label distribution vs Claude (sanity check)
    if g_missing < n * 0.5:
        gpt_dist = out['gpt_label'].value_counts().to_dict()
        print(f"\nGPT label dist (of {n-g_missing} successful):  {gpt_dist}")
        print(f"Claude label dist (all {n}): {out['claude_label'].value_counts().to_dict()}")

    audit = {
        "total_paragraphs":         n,
        "agreement_distribution":   {k: int(v) for k, v in agree_dist.items()},
        "full_agreement_rate":      round(full_rate, 4),
        "final_label_distribution": {
            str(k): int(final_dist.get(float(k), final_dist.get(k, 0)))
            for k in [-1, 0, 1]
        },
        "claude_missing":           c_missing,
        "gpt_missing":              g_missing,
        "gpt_model":                "gpt-5.5-2026-04-23",
        "gpt_api":                  "Chat Completions",
        "date_range":               f"{out['date'].min()} to {out['date'].max()}"
    }
    with open(AUDIT_OUT, 'w') as f:
        json.dump(audit, f, indent=2)

    print(f"\nAudit → {AUDIT_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
