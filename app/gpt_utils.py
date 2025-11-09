# gpt_utils.py

import os
from typing import Optional

# ------- Existing helpers (unchanged) -------
def history_summary(facts: dict) -> str:
    s = facts
    return (f"Total invoices: {s['total_invoices']}, total revenue RM {s['total_revenue']:,.0f}.  \n"
            f"Peak {s['peak_month']} (RM {s['peak_revenue']:,.0f}); "
            f"Trough {s['trough_month']} (RM {s['trough_revenue']:,.0f}).  \n"
            "Top items: " + ", ".join([t['Item No.'] for t in s['top_items']]) + ".")

def prediction_explain(facts: dict) -> str:
    i = facts['inputs']; y = facts['prediction_amount']; p = facts.get('p_high')
    base = (f"Predicted **Amount** RM {y:,.0f} for {i['Quantity']} unit(s) of "
            f"{i['Item No.']} / {i['Variant']} in {i['Month']}/{i['Year']}.  \n")
    whatif = "- What-if: +1 unit raises total roughly by the typical unit price; -1 unit lowers it similarly."
    prob = f"  High-Value probability: **{p:.1%}**." if p is not None else ""
    return base + whatif + prob

# ------- New: GPT CSV Analysis helper -------

def _get_api_key() -> Optional[str]:
    """Find an API key from Streamlit secrets or env."""
    key = os.getenv("OPENAI_API_KEY")
    try:
        import streamlit as st  # only for secrets access; safe if streamlit present
        if not key and "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = key
    except Exception:
        pass
    return key

def have_real_gpt() -> bool:
    try:
        from openai import OpenAI  # noqa: F401
    except Exception:
        return False
    return bool(_get_api_key())

def _stub_answer(context: str, question: str) -> str:
    return (
        "*(Stubbed AI: No OPENAI_API_KEY configured)*\n\n"
        "I analysed the provided dataset summary and here are concise findings:\n\n"
        f"**Summary snapshot**\n```\n{context}\n```\n\n"
        "**Observations**\n"
        "- Total & monthly figures indicate overall trend.\n"
        "- Top products/variants drive the majority of revenue.\n"
        "- Consider monitoring peaks/troughs and seasonality.\n\n"
        "**Next steps**\n"
        "- Add an API key to enable real GPT analysis.\n"
        "- Ask domain questions (e.g., ‘What changed MoM?’, ‘Which items drive growth?’)."
    )

def gpt_csv_analyse(context: str, question: str, model: str = "gpt-4o-mini") -> str:
    """
    Use GPT to analyse a compact dataset summary (context) and answer the user question.
    Falls back to a local stub if no API key or library is available.
    """
    if not have_real_gpt():
        return _stub_answer(context, question)

    from openai import OpenAI
    client = OpenAI(api_key=_get_api_key())

    system_msg = (
        "You are a senior sales analytics assistant. Work only with the provided summary; "
        "do not invent columns. Be precise, show amounts with thousands separators, and "
        "answer in clear Markdown with bullet points and, when useful, concise tables."
    )

    user_msg = (
        f"DATA SUMMARY (from a CSV; pre-aggregated for compactness):\n{context}\n\n"
        f"QUESTION / TASK:\n{question}\n"
        "If there is not enough information, say what else would be needed."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()
