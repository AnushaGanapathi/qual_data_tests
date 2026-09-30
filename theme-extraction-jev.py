from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import json
import os
import time
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
from typesafe_sdk import AsyncTypeSafeClient, Choice

# -----------------------------
# Config
# -----------------------------
SCHEMA = "prod_staging_qual"
TABLE = "sample_themes"
JEV_THEME_COL = "theme_extracted_jev"        # new column for the Jev label
JEV_CONFIDENCE_COL = "theme_jev_confidence"   # new column for Jev's confidence (0-1)
TYPESAFE_API_KEY = os.environ["TYPESAFE_API_KEY"]
MODEL = "jev-latest"
CONCURRENCY = 10          # parallel requests to TypeSafe
REVIEW_CONFIDENCE = 0.5   # below this, flag the row for human review in the CSV

# Theme names are identical to the OpenAI version so results are directly comparable.
# Descriptions help Jev separate neighbouring themes; tune them on your data.
THEMES = {
    "Confidence & Self-Belief":
        "Growth in self-confidence, self-esteem, courage to speak up, or belief in one's own abilities",
    "Aspirations for Higher Education":
        "Plans or motivation to continue studying: college, degrees, further courses or exams",
    "Employability & Job Readiness":
        "Getting a job, internships, interviews, career plans, workplace readiness or earning income",
    "Skills Development (Digital & Communication)":
        "Learning concrete skills such as computers, internet, English, spoken or written communication",
    "Family, Gender & Social Support":
        "Support or resistance from family and community, gender norms, marriage, safety, or social acceptance",
}

QUESTIONS = {
    "theme": Choice(
        instructions=(
            "This is a participant's response to an education program feedback survey. "
            "Which theme does the response mainly express?"
        ),
        criteria=THEMES,
    ),
}


# -----------------------------
# DB connection
# -----------------------------
def get_connection():
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=os.environ["PG_PORT"],
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"]
    )


# -----------------------------
# Jev columns (kept separate from theme_extracted, which holds the OpenAI label)
# -----------------------------
def jev_columns_exist(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """,
        (SCHEMA, TABLE, JEV_THEME_COL),
    )
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def ensure_jev_columns(conn):
    cur = conn.cursor()
    cur.execute(f"""
        ALTER TABLE {SCHEMA}.{TABLE}
        ADD COLUMN IF NOT EXISTS {JEV_THEME_COL} TEXT,
        ADD COLUMN IF NOT EXISTS {JEV_CONFIDENCE_COL} DOUBLE PRECISION
    """)
    conn.commit()
    cur.close()


# -----------------------------
# Fetch rows
# -----------------------------
def fetch_rows(conn, include_classified):
    # theme_extracted (the OpenAI label) is fetched for comparison only
    where = ""
    if not include_classified and jev_columns_exist(conn):
        where = f"WHERE {JEV_THEME_COL} IS NULL OR trim({JEV_THEME_COL}) = ''"
    query = f"""
        SELECT response_id, response_text, theme_extracted AS openai_theme
        FROM {SCHEMA}.{TABLE}
        {where}
    """
    return pd.read_sql(query, conn)


# -----------------------------
# Jev classification
# -----------------------------
async def classify_one(client, semaphore, response_text):
    if not response_text or not str(response_text).strip():
        return None, None, None
    async with semaphore:
        # Timed inside the semaphore so it measures the API call, not time spent queued
        start = time.perf_counter()
        response = await client.system_one(state=str(response_text), questions=QUESTIONS)
        latency_ms = (time.perf_counter() - start) * 1000
    return response.answers["theme"], latency_ms, response.usage


async def classify_all(texts):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with AsyncTypeSafeClient(api_key=TYPESAFE_API_KEY, model=MODEL) as client:
        return await asyncio.gather(
            *(classify_one(client, semaphore, t) for t in texts)
        )


def apply_classification(df):
    start = time.perf_counter()
    results = asyncio.run(classify_all(df["response_text"].tolist()))
    total_s = time.perf_counter() - start

    answers = [r[0] for r in results]
    latencies = [r[1] for r in results if r[1] is not None]
    usages = [r[2] for r in results if r[2] is not None]
    df["jev_latency_ms"] = [round(r[1], 1) if r[1] is not None else None for r in results]

    print(f"Jev classified {len(latencies)} responses in {total_s:.2f}s "
          f"(concurrency {CONCURRENCY}, {len(latencies) / total_s:.1f} responses/s)")
    if latencies:
        s = pd.Series(latencies)
        print(f"Per-request latency: mean {s.mean():.0f}ms, median {s.median():.0f}ms, "
              f"p95 {s.quantile(0.95):.0f}ms, max {s.max():.0f}ms")
    if usages:
        print(f"Tokens: {sum(u.input_tokens for u in usages)} input, "
              f"{sum(u.output_tokens for u in usages)} output")

    df["theme_jev"] = [a.choice if a else None for a in answers]
    df["jev_confidence"] = [a.confidence if a else None for a in answers]
    df["jev_probabilities"] = [json.dumps(a.probabilities) if a else None for a in answers]
    df["needs_review"] = [a is None or a.confidence < REVIEW_CONFIDENCE for a in answers]
    return df


# -----------------------------
# Update table (ONLY the Jev columns)
# -----------------------------
def update_table(conn, df):
    df = df[df["theme_jev"].notna()]
    if df.empty:
        print("No new responses to classify.")
        return

    cur = conn.cursor()

    query = f"""
        UPDATE {SCHEMA}.{TABLE} t
        SET {JEV_THEME_COL} = v.theme_jev,
            {JEV_CONFIDENCE_COL} = v.jev_confidence
        FROM (VALUES %s) AS v(
            response_id,
            theme_jev,
            jev_confidence
        )
        WHERE t.response_id = v.response_id
    """

    values = [
        (r.response_id, r.theme_jev, r.jev_confidence)
        for r in df.itertuples(index=False)
    ]

    execute_values(cur, query, values)
    conn.commit()
    cur.close()


# -----------------------------
# Report
# -----------------------------
def report(df, out_path):
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Flagged for review (confidence < {REVIEW_CONFIDENCE}): {int(df['needs_review'].sum())}")
    print("\nJev theme counts:")
    print(df["theme_jev"].value_counts(dropna=False).to_string())

    compared = df[df["openai_theme"].notna()]
    if not compared.empty:
        agree = (compared["openai_theme"].str.strip() == compared["theme_jev"]).mean()
        print(f"\nAgreement with existing OpenAI labels: {agree:.1%} of {len(compared)} rows")
        print(pd.crosstab(compared["openai_theme"], compared["theme_jev"]).to_string())


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Classify qualitative responses into themes with Jev.")
    parser.add_argument("--all", action="store_true",
                        help="Classify every row, including ones Jev already labelled")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write the CSV; don't add columns or write to Postgres")
    parser.add_argument("--out", default="jev_themes.csv", help="CSV output path")
    args = parser.parse_args()

    conn = get_connection()
    try:
        if not args.dry_run:
            ensure_jev_columns(conn)
        df = fetch_rows(conn, include_classified=args.all)
        if df.empty:
            print("No responses to classify.")
            return
        df = apply_classification(df)
        report(df, args.out)
        if not args.dry_run:
            update_table(conn, df)
            print(f"{JEV_THEME_COL} and {JEV_CONFIDENCE_COL} updated in Postgres.")
    finally:
        conn.close()

    print("Theme classification completed successfully.")


if __name__ == "__main__":
    main()
