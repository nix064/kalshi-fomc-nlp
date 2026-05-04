"""
scrape_fed.py — FT370 Sprint, Step 3 (v3 — all fixes applied)
Scrapes federalreserve.gov for:
  - Powell speeches    → data/raw/speeches/   (HTML)
  - Powell testimony   → data/raw/testimony/  (HTML)
  - FOMC statements    → data/raw/statements/ (HTML)
  - FOMC minutes       → data/raw/minutes/    (HTML)
  - FOMC press conf    → data/raw/pressconf/  (PDF, built from fomc_outcomes.csv dates)

Usage:
  python pipeline/scrape_fed.py --test                    # 3 docs per category
  python pipeline/scrape_fed.py                           # full run
  python pipeline/scrape_fed.py --category pressconf      # one category only
"""

import sys, time, argparse, logging, re, csv
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scrape_fed.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

BASE  = "https://www.federalreserve.gov"
YEARS = list(range(2018, 2027))
DELAY = 1.5  # seconds between requests

CATEGORIES    = ["speeches", "testimony", "statements", "minutes", "pressconf"]
MANIFEST_PATH = "data/corpus_manifest.csv"
MANIFEST_FIELDS = ["category", "doc_id", "url", "local_path", "date", "title"]


# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_dirs():
    for cat in CATEGORIES:
        Path(f"data/raw/{cat}").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)


# ─── HTTP ─────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (academic research; contact nsaliou@bentley.edu)"
})

def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=30)
            r.raise_for_status()
            return r
        except Exception as e:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(3)
    log.error("All retries failed: %s", url)
    return None

def saved(path):
    return Path(path).exists() and Path(path).stat().st_size > 200


# ─── Manifest ─────────────────────────────────────────────────────────────────

def load_manifest_urls():
    if not Path(MANIFEST_PATH).exists():
        return set()
    with open(MANIFEST_PATH) as f:
        return {row["url"] for row in csv.DictReader(f)}

def append_manifest(rows):
    new_file = not Path(MANIFEST_PATH).exists()
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)

def date_from_url(url):
    m = re.search(r"(\d{4})(\d{2})(\d{2})", url)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "unknown"

def doc_id_from_url(url):
    stem = url.rstrip("/").split("/")[-1]
    stem = re.sub(r"\.(htm|html|pdf)$", "", stem, flags=re.I)
    return re.sub(r"[^a-z0-9_\-]", "_", stem.lower())


# ─── Speeches & Testimony ─────────────────────────────────────────────────────

def scrape_powell_index(year, kind, test_mode, seen_urls):
    """
    kind = 'speech' or 'testimony'
    Verified URL patterns:
      speeches:  /newsevents/speech/{year}-speeches.htm
      testimony: /newsevents/testimony/{year}-testimony.htm
    """
    cat = "speeches" if kind == "speech" else "testimony"

    if kind == "speech":
        index_url = f"{BASE}/newsevents/speech/{year}-speeches.htm"
    else:
        index_url = f"{BASE}/newsevents/testimony/{year}-testimony.htm"

    r = fetch(index_url)
    if r is None:
        log.warning("[%s] Index not found for %d — skipping", cat, year)
        return []
    time.sleep(DELAY)

    soup = BeautifulSoup(r.text, "lxml")
    pattern = re.compile(rf"/newsevents/{kind}/powell\d{{8}}", re.I)
    links = list(dict.fromkeys(
        urljoin(BASE, a["href"])
        for a in soup.find_all("a", href=pattern)
    ))

    log.info("[%s] %d Powell docs in %d", cat, len(links), year)
    if test_mode:
        links = links[:3]

    rows = []
    for url in links:
        if url in seen_urls:
            log.info("[%s] Skip (seen): %s", cat, url)
            continue

        doc_id     = doc_id_from_url(url)
        local_path = f"data/raw/{cat}/{doc_id}.html"
        date_str   = date_from_url(url)

        if not saved(local_path):
            doc_r = fetch(url)
            if doc_r is None:
                continue
            Path(local_path).write_text(doc_r.text, encoding="utf-8", errors="replace")
            log.info("[%s] Saved: %s  (%s)", cat, doc_id, date_str)
            time.sleep(DELAY)
        else:
            log.info("[%s] Exists: %s", cat, doc_id)

        # Try to pull title from saved HTML
        title = doc_id
        try:
            s = BeautifulSoup(
                Path(local_path).read_text(encoding="utf-8", errors="replace"), "lxml"
            )
            for tag in ["h3", "h2", "h1"]:
                h = s.find(tag)
                if h and len(h.get_text(strip=True)) > 5:
                    title = h.get_text(strip=True)[:200]
                    break
        except Exception:
            pass

        rows.append({
            "category": cat, "doc_id": doc_id, "url": url,
            "local_path": local_path, "date": date_str, "title": title,
        })
        seen_urls.add(url)

    return rows


# ─── FOMC Calendar — statements and minutes ───────────────────────────────────

def parse_fomc_calendar(test_mode):
    """
    Pulls statement and minutes links from the FOMC calendar page.
    Press conf PDFs are built separately from fomc_outcomes.csv.
    """
    cal_url = f"{BASE}/monetarypolicy/fomccalendars.htm"
    log.info("[calendar] Fetching FOMC calendar...")
    r = fetch(cal_url)
    if r is None:
        log.error("Cannot fetch FOMC calendar.")
        return [], []
    time.sleep(DELAY)

    soup = BeautifulSoup(r.text, "lxml")

    stmts = list(dict.fromkeys(
        urljoin(BASE, a["href"])
        for a in soup.find_all(
            "a", href=re.compile(r"/pressreleases/monetary\d{8}a\.htm", re.I)
        )
    ))
    mins = list(dict.fromkeys(
        urljoin(BASE, a["href"])
        for a in soup.find_all(
            "a", href=re.compile(r"/fomcminutes\d{8}\.htm", re.I)
        )
    ))

    log.info("[calendar] %d statements | %d minutes", len(stmts), len(mins))

    if test_mode:
        stmts = stmts[:3]
        mins  = mins[:3]

    return stmts, mins


def build_pressconf_urls(test_mode):
    """
    Build press conference PDF URLs directly from fomc_outcomes.csv.
    Pattern: /mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf
    Press conferences started for every meeting from 2019-01-30 onward.
    Pre-2019 only quarterly meetings had press confs.
    We attempt all and skip 404s gracefully.
    """
    df = pd.read_csv("data/fomc_outcomes.csv")
    urls = [
        f"{BASE}/mediacenter/files/FOMCpresconf{d.replace('-', '')}.pdf"
        for d in df["date"]
    ]
    log.info("[pressconf] %d candidate PDF URLs from fomc_outcomes.csv", len(urls))
    if test_mode:
        urls = urls[-5:]  # test on most recent 5 (most likely to exist)
    return urls


# ─── Generic HTML and PDF scrapers ───────────────────────────────────────────

def scrape_html_list(links, cat, seen_urls):
    rows = []
    for url in links:
        if url in seen_urls:
            continue
        doc_id     = doc_id_from_url(url)
        local_path = f"data/raw/{cat}/{doc_id}.html"
        date_str   = date_from_url(url)

        if not saved(local_path):
            r = fetch(url)
            if r is None:
                continue
            Path(local_path).write_text(r.text, encoding="utf-8", errors="replace")
            log.info("[%s] Saved: %s  (%s)", cat, doc_id, date_str)
            time.sleep(DELAY)
        else:
            log.info("[%s] Exists: %s", cat, doc_id)

        rows.append({
            "category": cat, "doc_id": doc_id, "url": url,
            "local_path": local_path, "date": date_str,
            "title": f"FOMC {cat} {date_str}",
        })
        seen_urls.add(url)
    return rows


def scrape_pdf_list(links, cat, seen_urls):
    rows = []
    for url in links:
        if url in seen_urls:
            continue
        doc_id     = doc_id_from_url(url)
        local_path = f"data/raw/{cat}/{doc_id}.pdf"
        date_str   = date_from_url(url)

        if not saved(local_path):
            r = fetch(url)
            if r is None:
                # 404 = no press conf for this meeting date — skip silently
                continue
            Path(local_path).write_bytes(r.content)
            log.info("[%s] Saved PDF: %s  (%s)", cat, doc_id, date_str)
            time.sleep(DELAY)
        else:
            log.info("[%s] Exists: %s", cat, doc_id)

        rows.append({
            "category": cat, "doc_id": doc_id, "url": url,
            "local_path": local_path, "date": date_str,
            "title": f"FOMC press conference {date_str}",
        })
        seen_urls.add(url)
    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Fetch up to 3 docs per category — no full run")
    parser.add_argument("--category", choices=CATEGORIES,
                        help="Run only one category (useful for resuming)")
    args = parser.parse_args()

    if args.test:
        log.info("=== TEST MODE: up to 3 docs per category ===")

    setup_dirs()
    seen_urls = load_manifest_urls()
    all_rows  = []

    run = lambda cat: not args.category or args.category == cat

    # ── Speeches ──────────────────────────────────────────────────────────────
    if run("speeches"):
        log.info("=== SPEECHES ===")
        rows = []
        for year in YEARS:
            r = scrape_powell_index(year, "speech", args.test, seen_urls)
            rows += r
            if args.test and rows:
                break
        append_manifest(rows)
        all_rows += rows
        log.info("[speeches] %d new docs", len(rows))

    # ── Testimony ─────────────────────────────────────────────────────────────
    if run("testimony"):
        log.info("=== TESTIMONY ===")
        rows = []
        for year in YEARS:
            r = scrape_powell_index(year, "testimony", args.test, seen_urls)
            rows += r
            if args.test and rows:
                break
        append_manifest(rows)
        all_rows += rows
        log.info("[testimony] %d new docs", len(rows))

    # ── Statements + Minutes (from FOMC calendar) ─────────────────────────────
    stmt_links, mins_links = parse_fomc_calendar(args.test)

    if run("statements"):
        log.info("=== STATEMENTS ===")
        rows = scrape_html_list(stmt_links, "statements", seen_urls)
        append_manifest(rows)
        all_rows += rows
        log.info("[statements] %d new docs", len(rows))

    if run("minutes"):
        log.info("=== MINUTES ===")
        rows = scrape_html_list(mins_links, "minutes", seen_urls)
        append_manifest(rows)
        all_rows += rows
        log.info("[minutes] %d new docs", len(rows))

    # ── Press conf PDFs (built from fomc_outcomes.csv) ────────────────────────
    if run("pressconf"):
        log.info("=== PRESS CONF ===")
        presconf_links = build_pressconf_urls(args.test)
        rows = scrape_pdf_list(presconf_links, "pressconf", seen_urls)
        append_manifest(rows)
        all_rows += rows
        log.info("[pressconf] %d new docs", len(rows))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SCRAPE DONE — {len(all_rows)} new docs this run")
    if Path(MANIFEST_PATH).exists():
        cats = {}
        with open(MANIFEST_PATH) as f:
            for row in csv.DictReader(f):
                cats[row["category"]] = cats.get(row["category"], 0) + 1
        print(f"  Total in manifest: {sum(cats.values())}")
        for cat, n in sorted(cats.items()):
            print(f"    {cat:15s}: {n}")
    print()


if __name__ == "__main__":
    main()
