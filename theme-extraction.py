from dotenv import load_dotenv
load_dotenv()
import os
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
SCHEMA = "prod_staging_qual"
TABLE = "sample_themes"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
# Fetch rows needing classification
# -----------------------------
def fetch_unclassified(conn):
    query = f"""
        SELECT response_id, response_text
        FROM {SCHEMA}.{TABLE}
        WHERE theme_extracted IS NULL
    """
    return pd.read_sql(query, conn)


# -----------------------------
# LLM classification
# -----------------------------
def classify_theme(response_text):
    prompt = f"""
You are helping categorize qualitative education program feedback.

Classify the following response into exactly ONE of the themes below.
Return ONLY the theme name, nothing else.

Themes:
1. Confidence & Self-Belief
2. Aspirations for Higher Education
3. Employability & Job Readiness
4. Skills Development (Digital & Communication)
5. Family, Gender & Social Support

Response:
\"{response_text}\"
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return completion.choices[0].message.content.strip()


# -----------------------------
# Apply classification
# -----------------------------
def apply_classification(df):
    df["theme_extracted"] = df["response_text"].apply(classify_theme)
    return df


# -----------------------------
# Update table (ONLY theme_extracted)
# -----------------------------
def update_table(conn, df):
    if df.empty:
        print("No new responses to classify.")
        return

    cur = conn.cursor()

    query = f"""
        UPDATE {SCHEMA}.{TABLE} t
        SET theme_extracted = v.theme_extracted
        FROM (VALUES %s) AS v(
            response_id,
            theme_extracted
        )
        WHERE t.response_id = v.response_id
    """

    values = [
        (r.response_id, r.theme_extracted)
        for r in df.itertuples(index=False)
    ]

    execute_values(cur, query, values)
    conn.commit()
    cur.close()


# -----------------------------
# Main
# -----------------------------
def main():
    conn = get_connection()
    try:
        df = fetch_unclassified(conn)
        df = apply_classification(df)
        update_table(conn, df)
    finally:
        conn.close()

    print("Theme classification completed successfully.")


if __name__ == "__main__":
    main()
