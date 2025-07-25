import os
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Set up OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY", "sk-...")
llm = OpenAI(api_token=openai_api_key)

async def handle_question(task: str):
    # 1. Scrape and load the Wikipedia table
    df = pd.read_html("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")[0]

    # 2. Clean column headers and restrict to useful columns
    df.columns = [col.strip() for col in df.columns]
    keep_cols = ["Rank", "Title", "Worldwide gross", "Year", "Peak"]
    df = df[[col for col in keep_cols if col in df.columns]].copy()

    # 3. Clean numeric columns
    df["Worldwide gross"] = df["Worldwide gross"].astype(str).str.replace(r"[^\d.]", "", regex=True)
    df["Peak"] = df["Peak"].astype(str).str.replace(r"[^\d.]", "", regex=True)

    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")
    df["Worldwide gross"] = pd.to_numeric(df["Worldwide gross"], errors="coerce")

    df = df.dropna(subset=["Rank", "Year", "Peak"])

    # 4. Create a SmartDataframe for LLM queries
    sdf = SmartDataframe(df, config={"llm": llm})

    # 5. Ask questions
    q1 = "How many $2 bn movies were released before 2020?"
    q2 = "Which is the earliest film that grossed over $1.5 bn?"
    q3 = "What's the correlation between the Rank and Peak?"

    try:
        a1 = sdf.chat(q1)
    except Exception as e:
        a1 = f"Error: {str(e)}"

    try:
        a2 = sdf.chat(q2)
    except Exception as e:
        a2 = f"Error: {str(e)}"

    try:
        a3 = float(sdf.chat(q3))
    except Exception as e:
        a3 = 0.0

    # 6. Create base64 scatterplot
    plt.figure(figsize=(6, 4))
    plt.scatter(df["Rank"], df["Peak"], label="Movies")
    m, b = pd.Series(df["Rank"]).corr(df["Peak"]), df["Peak"].mean()
    plt.plot(df["Rank"], m * df["Rank"] + b, color="red", linestyle="dotted", label="Regression Line")
    plt.xlabel("Rank")
    plt.ylabel("Peak")
    plt.title("Rank vs Peak")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    image_uri = f"data:image/png;base64,{img_base64[:100000]}"

    return [a1, a2, a3, image_uri]
