
import streamlit as st, pandas as pd, joblib, matplotlib.pyplot as plt
from gpt_utils import history_summary, prediction_explain

st.set_page_config(page_title="Sales Assistant", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/df4.csv", parse_dates=["Invoice date"])
    df["YearMonth"] = df["Invoice date"].dt.to_period("M").dt.to_timestamp()
    return df

@st.cache_resource
def load_models():
    reg = joblib.load("models/sales_amount_pipeline.joblib")
    clf = joblib.load("models/high_value_classifier.joblib")
    return reg, clf

df4 = load_data()
reg, clf = load_models()
st.title("Sales Assistant (History + Prediction + GPT)")

tab1, tab2 = st.tabs(["📈 History & Insights", "🔮 Predict"])

with tab1:
    st.subheader("Sales History (on-demand)")
    run_eda = st.button("▶ Run Sales Analysis")
    if run_eda:
        monthly = df4.groupby("YearMonth")["Amount"].sum().reset_index()
        top_items = (df4.groupby("Item No.")["Amount"].sum().sort_values(ascending=False).head(10).reset_index())
        top_variants = (df4.groupby("Variant")["Amount"].sum().sort_values(ascending=False).head(10).reset_index())

        st.write("**Monthly Sales (Total Amount)**")
        fig1, ax1 = plt.subplots(figsize=(8,3))
        ax1.plot(monthly["YearMonth"], monthly["Amount"], marker="o")
        ax1.set_xlabel("Month"); ax1.set_ylabel("Revenue")
        fig1.tight_layout(); st.pyplot(fig1)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Top 10 Products by Revenue**")
            st.bar_chart(top_items.set_index("Item No.")["Amount"])
        with c2:
            st.write("**Top 10 Variants by Revenue**")
            st.bar_chart(top_variants.set_index("Variant")["Amount"])

        peak = monthly.loc[monthly["Amount"].idxmax()]
        trough = monthly.loc[monthly["Amount"].idxmin()]
        facts_hist = {
            "total_invoices": int(df4.shape[0]),
            "total_revenue": float(df4["Amount"].sum()),
            "peak_month": str(peak["YearMonth"].date()),
            "peak_revenue": float(peak["Amount"]),
            "trough_month": str(trough["YearMonth"].date()),
            "trough_revenue": float(trough["Amount"]),
            "top_items": top_items.head(5).to_dict(orient="records"),
            "top_variants": top_variants.head(5).to_dict(orient="records")
        }
        if st.button("🧠 Generate AI Summary"):
            st.info("AI Summary")
            st.write(history_summary(facts_hist))
    else:
        st.caption("Click **Run Sales Analysis** to compute charts (same logic as in Colab).")

with tab2:
    st.subheader("Predict Invoice Amount")
    c1,c2,c3 = st.columns(3)
    item = c1.selectbox("Item No.", sorted(df4["Item No."].unique()))
    variant = c2.selectbox("Variant", sorted(df4["Variant"].unique()))
    qty = c3.number_input("Quantity", min_value=1, value=1, step=1)
    c4,c5 = st.columns(2)
    year = c4.selectbox("Year", sorted(df4["Year"].unique()))
    month = c5.selectbox("Month", sorted(df4["Month"].unique()))

    if st.button("Predict"):
        X = pd.DataFrame([{"Item No.": item, "Variant": variant, "Quantity": qty, "Year": int(year), "Month": int(month)}])
        yhat = float(reg.predict(X)[0])
        st.metric("Predicted Amount (RM)", f"{yhat:,.0f}")
        p_high = float(clf.predict_proba(X)[0,1])
        st.caption(f"High-Value probability: {p_high:.2%}")
        facts_pred = {"inputs": X.iloc[0].to_dict(), "prediction_amount": yhat, "p_high": p_high}
        st.subheader("AI Explanation")
        st.write(prediction_explain(facts_pred))
