import os, random
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

# ==== OpenAI SDK compatibility: prefer v1.x OpenAI client; fallback to legacy openai ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY not detected. Add it to your .env to enable OpenAI responses.")

USE_CLIENT = False
client = None
try:
    from openai import OpenAI  # v1.x
    client = OpenAI(api_key=OPENAI_API_KEY)
    USE_CLIENT = True
except Exception:
    import openai  # legacy
    openai.api_key = OPENAI_API_KEY

st.set_page_config(page_title="Step 3: Compare Gemini vs OpenAI", page_icon=":brain:", layout="centered")
st.title("Step 3: Compare Gemini vs OpenAI")

LEN_RULE = "Reply in exactly one sentence (<=40 words). No bullet points."

def call_openai(prompt: str) -> str:
    """Call OpenAI (auto-compat with new/old SDK). Returns readable error string on failure."""
    try:
        sys = "You are a supportive, concise mental health peer supporter. Be empathetic and non-judgmental."
        user = f"{prompt}\n\n{LEN_RULE}" if short_reply else str(prompt)

        if USE_CLIENT:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=1,
                max_tokens=96 if short_reply else 600,
                top_p=1,
            )
            text = (resp.choices[0].message.content or "").strip()
        else:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=1,
                max_tokens=96 if short_reply else 600,
                top_p=1,
            )
            text = resp.choices[0].message["content"].strip()

        if short_reply:
            words = text.split()
            if len(words) > 40:
                text = " ".join(words[:40])
        return text
    except Exception as e:
        return f"[OpenAI ERROR] {e}"


uploaded = st.file_uploader("Upload the CSV from Step 2 (with Gemini responses)", type=["csv"])
mode = st.radio("OpenAI response mode", ["Manual input", "API call", "API then manual"], index=0)
short_reply = st.checkbox("OpenAI short reply (single sentence, ≤40 words)", value=True)

if mode in ["API call", "API then manual"] and not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not detected; cannot call the API. Add it to .env and try again.")
    st.stop()

if uploaded is None:
    st.info("Please upload the CSV generated in Step 2 (20 examples × 5 variants each).")
    st.stop()

df = pd.read_csv(uploaded)

# Required column check
need_cols = {"example_id", "variant", "user_prompt", "response"}
missing = [c for c in need_cols if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}. Please re-export in Step 2.")
    st.stop()

# Do not filter baseline: select example_id from the whole pool, and keep all 5 variants for each selected id
pool = df.copy()

unique_examples = sorted(pool["example_id"].unique().tolist())
if len(unique_examples) < 15:
    st.error(f"Only {len(unique_examples)} example_id values available; need at least 15. "
             f"Go back to Step 2 and sample ≥15 examples.")
    st.stop()

seed = st.number_input("Random seed (reproducible)", value=42, step=1)
random.seed(int(seed))
default_ids = random.sample(unique_examples, 15)

chosen_ids = st.multiselect(
    "Pick exactly 15 example_id values from the 20 examples in Step 2 (default is a random selection)",
    unique_examples,
    default=default_ids,
)

if len(chosen_ids) != 15:
    st.warning(f"You selected {len(chosen_ids)} example_id values. Please select exactly 15.")
    st.stop()

# Keep all 5 variants for the selected example_id values (no variant filtering)
selected = pool[pool["example_id"].isin(chosen_ids)].copy()

# Quality check: each example_id should have exactly 5 rows
cnt = selected.groupby("example_id").size()
lack = cnt[cnt != 5]
if not lack.empty:
    st.warning(f"{len(lack)} example_id values do not have 5 rows: {lack.to_dict()}. "
               f"Please verify Step 2 export is complete.")

selected.sort_values(["example_id", "variant"], inplace=True)

# Initialize openai_response column if absent
if "openai_response" not in selected.columns:
    selected["openai_response"] = ""

if mode in ["API call", "API then manual"]:
    with st.spinner("Calling OpenAI (15 × 5 = 75 rows)…"):
        outs = []
        prog = st.progress(0)
        rows = selected.to_dict("records")
        for i, row in enumerate(rows, start=1):
            prompt = row.get("user_prompt", row.get("original_text", ""))
            outs.append(call_openai(str(prompt)))
            prog.progress(i / len(rows))
        selected["openai_response"] = outs
    st.success("OpenAI responses generated.")

if mode in ["Manual input", "API then manual"]:
    st.caption("👇 You can edit `openai_response` manually below (to satisfy the 'manual generate responses' requirement).")
    editable_cols = ["openai_response"]
    selected = st.data_editor(
        selected,
        column_config={
            "openai_response": st.column_config.TextColumn("openai_response (manually editable)", width="large"),
            "response": st.column_config.TextColumn("gemini_response", width="large"),
        },
        disabled=[c for c in selected.columns if c not in editable_cols],
        use_container_width=True,
        key="editable_table",
    )

# 6) Side-by-side view
st.subheader("Gemini vs OpenAI comparison")
show_cols = [c for c in ["example_id", "variant", "original_text", "response", "openai_response"] if c in selected.columns]
st.dataframe(selected[show_cols], use_container_width=True)
selected.sort_values(["example_id", "variant"], inplace=True)

# 7) Export
fname = "step3_compare_gemini_openai_15x5.csv"
st.download_button(
    "Download comparison CSV",
    data=selected.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
    file_name=fname,
    mime="text/csv",
)
