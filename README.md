# Farhan-CAIE-Streamlit

This project is developed as part of the **CAIE Certification**.  
It integrates **Traditional Machine Learning** (Regression & Classification) with **AI explanation** using Streamlit.

---

## 🧠 Stage 1 – Machine Learning
- **Regression (Random Forest):** Predicts invoice **Amount (RM)**.  
  - MAE = RM 8,249 | R² = 0.566  
- **Classification (Logistic Regression):** Flags **High-Value invoices** (top 25%).  
  - Accuracy = 82% | PR-AUC = 0.69  

Both models are saved as:
models/sales_amount_pipeline.joblib
models/high_value_classifier.joblib

## 💻 Stage 2 – Streamlit + GPT Integration
### App Features
- **📈 History & Insights**
  - Button to generate on-demand sales analysis (monthly trend, top products/variants).
  - Optional “AI Summary” using GPT-style explanation.
- **🔮 Predict**
  - Enter product details → predict sales amount and High-Value probability.
  - AI explanation for reasoning and what-if scenarios.
