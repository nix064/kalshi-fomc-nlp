import os
import re
import time
import json
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE      = Path.home() / "ft370_project"
LABELS    = BASE / "data/hand_labels/llm_labeled_paragraphs.csv"
AUDIT_OUT = BASE / "data/hand_labels/llm_labeling_audit.json"

SYSTEM_PROMPT = """You are an expert in Federal Reserve monetary policy communication.
Classify this paragraph from a Powell speech or press conference.
Label based on the CONCLUSION and forward-looking implication for interest rates:
HAWKISH: Implies rates should go UP or stay high.
DOVISH: Implies rates should go DOWN or stay low.
NEUTRAL: No clear forward-looking implication, or explicitly balanced.
Return ONLY one word: HAWKISH, DOVISH, or NEUTRAL."""

LABEL_MAP = {"HAWKISH": -1, "DOVISH": 1, "NEUTRAL": 0}

def label_gpt(text: str):
    try:
        resp = openai_client.responses.create(
            model="gpt-5.5",
            reasoning={"effort": "low"},
            instructions=SYSTEM_PROMPT,
            input=text,
            max_output_tokens=32
        )
        word = re.sub(r'[^A-Z]', '', resp.output_text.strip().upper())
        return LABEL_MAP.get(word, None)
    except Exception as e:
        try:
            resp2 = openai_client.chat.completions.create(
                model="gpt-5.5",
                max_completion_tokens=32,  # fixed: was max_tokens
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text}
                ]
            )
            word = re.sub(r'[^A-Z]', '', resp2.choices[0].message.content.strip().upper())
            return LABEL_MAP.get(word, None)
        except Exception as e2:
            print(f"  Both APIs failed: {e2}")
            return None

df = pd.read_csv(LABELS)
failed_mask = df['llm_label'].isna()
print(f"Fixing {failed_mask.sum()} failed labels...")

for i, idx in enumerate(df[failed_mask].index):
    row = df.loc[idx]
    label = label_gpt(row['text'])
    df.at[idx, 'llm_label'] = label
    df.at[idx, 'model'] = 'gpt'
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{failed_mask.sum()} fixed...")
    time.sleep(0.2)

df.to_csv(LABELS, index=False)

# Print final stats
dist = df['llm_label'].value_counts().to_dict()
still_failed = int(df['llm_label'].isna().sum())
print(f"\n✅ Done. {still_failed} still failed after retry.")
print("Label distribution (−1=Hawkish, 0=Neutral, 1=Dovish):")
for k in [-1.0, 0.0, 1.0]:
    label_name = {-1.0: "Hawkish", 0.0: "Neutral", 1.0: "Dovish"}.get(k, "Unknown")
    count = int(dist.get(k, 0))
    pct = count / len(df)
    print(f"  {label_name}: {count} ({pct:.1%})")

# Update audit
audit_path = AUDIT_OUT
if audit_path.exists():
    with open(audit_path) as f:
        audit = json.load(f)
else:
    audit = {}

audit['label_distribution'] = {str(int(k)): int(v) for k, v in dist.items() if not pd.isna(k)}
audit['failed_labels_after_retry'] = still_failed
audit['total_paragraphs'] = len(df)
with open(audit_path, 'w') as f:
    json.dump(audit, f, indent=2)
print(f"Audit updated → {audit_path}")
