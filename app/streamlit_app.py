# app/streamlit_app.py
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from gpt_utils import history_summary, prediction_explain

# ──────────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_theme("light")
st.set_page_config(page_title="Sales Performance Dashboard", layout="wide")

# scikit-learn 1.6/1.7 unpickle compatibility
# Some 1.6 pickles reference `_RemainderColsList`; 1.7 renamed it to `_RemainderColumnsList`.
try:
    from sklearn.compose import _column_transformer as _ct  # type: ignore
    if hasattr(_ct, "_RemainderColumnsList") and not hasattr(_ct, "_RemainderColsList"):
        _ct._RemainderColsList = _ct._RemainderColumnsList  # alias for 1.6 pickles on 1.7+
    if hasattr(_ct, "_RemainderColsList") and not hasattr(_ct, "_RemainderColumnsList"):
        _ct._RemainderColumnsList = _ct._RemainderColsList  # reverse alias safety
except Exception:
    # If anything goes wrong here, let joblib give the detailed error on load.
    pass

# Optional: show runtime versions in the sidebar
with st.sidebar:
    try:
        import sklearn  # type: ignore
        st.caption(
            f"**Env**  \n"
            f"scikit-learn `{sklearn.__version__}` · joblib `{joblib.__version__}` · "
            f"pandas `{pd.__version__}` · numpy `{np.__version__}` · py `{sys.version.split()[0]}`"
        )
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# Data & models
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/df4.csv", parse_dates=["Invoice date"])
    # Derive Year/Month for trend analysis
    df["YearMonth"] = df["H_YearMonth"] if "H_YearMonth" in df.columns else df["Invoice date"].dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_resource
def load_models():
    reg = joblib.load("models/sales_amount_pipeline.joblib")
    clf = joblib.load("models/high_value_classifier.joblib")
    return reg, clf

df4 = load_data()
reg, clf = load_models()

# ──────────────────────────────────────────────────────────────────────────────
# Title & Subtitle (centered)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0.2rem;'>
      Sales Performance Dashboard: <span style="white-space:nowrap;">Historical Analysis &amp; AI-Driven Prediction</span>
    </h1>
    <div style='text-align:center; color:#6C757D; font-size:0.95rem;'>
      <strong>By: Ahmad&nbsp;Farhan&nbsp;Anwar</strong> &nbsp;–&nbsp;
      <em>Assignment for Certified Artificial Intelligent Engineer (CAIE)</em>
    </div>
    <hr style='margin-top: 16px; margin-bottom: 20px;'>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_history(df: pd.DataFrame):
    """Aggregate monthly revenue and top product/variant contributors."""
    monthly = df.groupby("YearMonth", as_index=False)["Amount"].sum()
    top_items = (
        df.groupby("Item No.", as_index=False)["Amount"]
          .sum()
          .sort_values("Amount", ascending=False)
          .head(10)
    )
    top_variants = (
        df.groupby("Variant", as_index=False)["Amount"]
          .sum()
          .sort_values("Amount", ascending=False)
          .head(10)
    )
    return monthly, top_items, top_variants

# Persist EDA results so “Generate AI Summary” can run after a rerun
if "eda_ready" not in st.session_state:
    st.session_state.eda_ready = False
    st.session_state.eda_payload = None

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Historical Analysis", "🔮 Prediction (AI Assistant)"])

with tab1:
    st.subheader("Sales History (on-demand)")

    if st.button("▶ Run Sales Analysis", key="run_eda"):
        monthly, top_items, top_variants = compute_history(df4)
        st.session_state.eda_ready = True
        st.session_state.eda_payload = {
            "monthly": monthly.to_dict(orient="records"),
            "top_items": top_items.to_dict(orient="records"),
            "top_variants": top_variants.to_dict(orient="records"),
        }

    if st.session_state.eda_ready and st.session_state.eda_payload:
        # Rehydrate from session
        monthly = pd.DataFrame(st.session_state.eda_payload["monthly"])
        top_items = pd.DataFrame(st.session_state.eda_payload["top_items"])
        top_variants = pd.DataFrame(st.session_state.eda_payload["top_variants"])

        # Monthly Sales (line)
        fig1, ax1 = plt.subplots(figsize=(9, 3.6))
        ax1.plot(monthly["YearMonth"], monthly["Amount"], marker="o")
        ax1.set_title("Monthly Sales (Total Amount)")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Revenue")
        fig1.tight_layout()
        st.pyplot(fig1)

        # Top 10 Products & Variants (bars)
        c1, c2 = st.columns(2)
        with c1:
            fig2, ax2 = plt.subplots(figsize=(7, 3.6))
            x = np.arange(len(top_items))
            ax2.bar(x, top_items["Amount"])
            ax2.set_title("Top 10 Products by Revenue")
            ax2.set_xticks(x)
            ax2.set_xticklabels(top_items["Item No."], rotation=45, ha="right")
            ax2.set_ylabel("Total Revenue (Amount)")
            fig2.tight_height = True
            st.pyplot(fig2)

        with c2:
            fig3, ax3 = plt.subplots(figsize=(7, 3.6))
            x2 = np.arange(len(top_variants))
            ax3.bar(x2, top_variants["Amount"])
            ax3.set_title("Top 10 Variants by Revenue")
            ax3.set_xticks(x2)
            ax3.set_xticklabels(top_variants["Variant"], rotation=45, ha="right")
            ax3.set_ylabel("Total Revenue (Amount)")
            st.pyplot(fig3)

        # Facts for GPT narrative
        if not monthly.empty:
            peak_idx = monthly["Amount"].idxmax()
            trough_idx = monthly["Amount"].idxmin()
            peak_row = monthly.loc[0 if pd.isna(peak_idx) else peak_idx]
            trough_row = monthly.loc[0 if pd.isna(st.session_state.get('trough_idx', trough_idx)) else trough_idx]
        else:
            peak_row = trough_row = None

        facts_hist = {
            "total_invoices": int(df4.shape[0]),
            "total_revenue": float(df4["Amount"].sum()),
            "monthly": monthly.tail(12).to_dict(orient="records"),
            "peak_month": (str(pd.to_datetime(peak_row["YearMonth"]).date()) if peak_row is not None else None),
            "peak_revenue": (float(peak_row["Amount"]) if peak_row is not None else None),
            "trough_month": (str(pd.to_datetime(trough_row["YearMonth"]).date()) if trough_row is not None else None),
            "trough_revenue": (float(trough_row["Amount"]) if trough_row is not None else None),
            "top_items": top_items.head(5).to_dict(orient="records"),
            "top_variants": top_variants.head(5).to_dict(orient="records"),
        }

        if st.button("🧠 Generate AI Summary", key="ai_summary"):
            st.info("AI Summary")
            st.write(history_summary(facts_hist))

        with st.expander("Show last 5 months (parity check vs Colab)"):
            st.dataframe(monthly.tail(5))
    else:
        st.caption("Click **Run Sales Analysis** to compute charts (same logic as your Colab EDA).")

with tab2:
    st.subheader("Predict Invoice Amount")

    c1, c2, c3 = st.columns(3)
    item = c1.selectbox("Item No.", sorted(df4["Item No."].unique()))
    variant = c2.selectbox("Variant", sorted(df4["Variant"].unique()))
    qty = c3.number_input("Quantity", min_value=1, value=1, step=1)

    c4, c5 = st.columns(2)
    year = st.selectbox("Year", sorted(df4["Year"].unique()))
    month = st.selectbox("Month", sorted(df4["Month"].unique()))

    if st.button("Predict", key="predict_btn"):
        X = pd.DataFrame(
            [{"Item No.": str(item), "Variant": str(variant), "Quantity": int(qty), "Year": int(year), "Month": int(month)}]
        )

        # Regression prediction
        yhat = float(reg.predict(X)[0])
        st.metric("Predicted Amount", f"{yhat:,.0f} RM")

        # High-value probability (robust handling for sklearn differences)
        try:
            proba = clf.predict_proba(X)
            p_high = float(proba[0][1]) if proba.ndim == 2 and proba.shape[1] > 1 else None
        except Exception:
            p_high = None
        if p_high is not None:
            st.caption(f"High-Value probability: {p_high:.2%}")
        else:
            st.caption("High-Value probability: n/a")

        # AI explanation (stubbed, but uses real numbers via `facts_pred`)
        facts_pred = {"inputs": X.iloc[0].to_dict(), "prediction_amount": yhat, "surety": p_high}
        st.subheader("AI Summary")
        st.write(prediction_summary := prediction_environment := prediction_explain(facts_pred))
