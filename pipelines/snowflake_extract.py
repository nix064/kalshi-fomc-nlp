"""
snowflake_extract.py — FT370 Sprint, Step 1
Pulls daily candlestick price history from Snowflake for:
  - KXFEDDECISION-* and FEDDECISION-* (rate-decision markets)
  - KXFEDMENTION-* (mention markets)

Outputs:
  - data/kalshi_fed_decision.parquet
  - data/kalshi_mention.parquet

Usage:
  python pipeline/snowflake_extract.py          # full run
  python pipeline/snowflake_extract.py --test   # smoke test, prints only, no files saved
"""

import os, sys, argparse, logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/snowflake_extract.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

REQUIRED_ENV = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_ROLE"]

def validate_env():
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        log.error("Missing env vars: %s", missing)
        sys.exit(1)

def get_connection():
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        authenticator="externalbrowser",
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database="PREDMARKET",
        schema="KALSHI",
        role=os.environ["SNOWFLAKE_ROLE"],
        session_parameters={"QUERY_TAG": "ft370_extract"},
    )
    log.info("Connected: account=%s", os.environ["SNOWFLAKE_ACCOUNT"])
    return conn

# Must run as a separate execution before any SELECT — Snowflake VARIANT-in-CTE bug
MATERIALIZE_DDL = """
CREATE OR REPLACE TEMPORARY TABLE tmp_markets AS
SELECT TICKER, TITLE, RESULT, STATUS,
       OPEN_TIME, CLOSE_TIME, EXPIRATION_TIME,
       VOLUME, DERIVED_SERIES_TICKER
FROM PREDMARKET.KALSHI.DIM_MARKETS
"""

CANDLESTICKS_QUERY = """
SELECT
    c.MARKET_TICKER,
    c.SERIES_TICKER,
    c.END_PERIOD_TS        AS date,
    c.PRICE_CLOSE          AS price_close,
    c.PRICE_OPEN           AS price_open,
    c.YES_BID_CLOSE        AS yes_bid_close,
    c.YES_ASK_CLOSE        AS yes_ask_close,
    c.VOLUME               AS volume_daily,
    c.OPEN_INTEREST        AS open_interest,
    m.TITLE                AS market_title,
    m.RESULT               AS result,
    m.STATUS               AS status,
    m.OPEN_TIME,
    m.CLOSE_TIME,
    m.EXPIRATION_TIME,
    m.VOLUME               AS volume_lifetime,
    m.DERIVED_SERIES_TICKER
FROM PREDMARKET.KALSHI.FACT_CANDLESTICKS_DAILY c
LEFT JOIN tmp_markets m ON c.MARKET_TICKER = m.TICKER
WHERE {ticker_filter}
ORDER BY c.MARKET_TICKER, c.END_PERIOD_TS
"""

def extract(conn, ticker_filter, label, test_mode):
    cur = conn.cursor()
    log.info("[%s] Materializing DIM_MARKETS to tmp_markets...", label)
    cur.execute(MATERIALIZE_DDL)   # execution 1: DDL
    query = CANDLESTICKS_QUERY.format(ticker_filter=ticker_filter)
    if test_mode:
        query = query.replace(
            "ORDER BY c.MARKET_TICKER, c.END_PERIOD_TS",
            "LIMIT 200"
        )
    log.info("[%s] Pulling candlesticks...", label)
    cur.execute(query)             # execution 2: SELECT
    cols = [d[0].lower() for d in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    log.info("[%s] %d rows, %d tickers", label, len(df), df["market_ticker"].nunique())
    cur.close()
    return df

def print_summary(df, label):
    print(f"\n{'='*60}")
    print(f"  {label.upper()} — {len(df):,} rows, {df['market_ticker'].nunique()} tickers")
    print(f"{'='*60}")
    if df.empty:
        return
    s = (df.groupby("market_ticker")
           .agg(days=("date", "count"),
                price_min=("price_close", "min"),
                price_max=("price_close", "max"),
                vol=("volume_lifetime", "first"),
                result=("result", "first"))
           .sort_values("vol", ascending=False))
    print(s.to_string())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Smoke test: 200 rows only, no files saved")
    args = parser.parse_args()

    if args.test:
        log.info("=== TEST MODE: no files will be written ===")

    validate_env()
    conn = get_connection()

    try:
        rate_filter = "(c.MARKET_TICKER LIKE 'KXFEDDECISION%' OR c.MARKET_TICKER LIKE 'FEDDECISION%')"
        mention_filter = "c.MARKET_TICKER LIKE 'KXFEDMENTION%'"

        df_decision = extract(conn, rate_filter, "fed_decision", args.test)
        print_summary(df_decision, "fed_decision")

        df_mention = extract(conn, mention_filter, "mention", args.test)
        print_summary(df_mention, "mention")

    finally:
        conn.close()
        log.info("Snowflake connection closed.")

    if args.test:
        log.info("Test passed. Run without --test to save parquet files.")
        sys.exit(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    df_decision.to_parquet("data/kalshi_fed_decision.parquet", index=False)
    df_mention.to_parquet("data/kalshi_mention.parquet", index=False)
    df_decision.to_parquet(f"data/kalshi_fed_decision_{ts}.parquet", index=False)
    df_mention.to_parquet(f"data/kalshi_mention_{ts}.parquet", index=False)
    log.info("Saved both parquet files. Done. ✓")

if __name__ == "__main__":
    main()
