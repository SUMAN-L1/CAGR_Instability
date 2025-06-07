import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt
from statsmodels.formula.api import ols
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ========== Function Definitions ==========

def compute_cagr(data, column):
    data = data.copy()
    data['Time'] = np.arange(1, len(data) + 1)
    data['LogColumn'] = np.log(data[column].replace(0, np.nan).dropna())
    model = ols('LogColumn ~ Time', data=data).fit()
    cagr = (np.exp(model.params['Time']) - 1) * 100
    p_value = model.pvalues['Time']
    adj_r_squared = model.rsquared_adj
    return cagr, p_value, adj_r_squared

def compute_statistics(data, column):
    mean_val = data[column].mean()
    std_val = data[column].std()
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    return mean_val, std_val, cv_val

def compute_cdvi(cv, adj_r_squared):
    return cv * sqrt(1 - adj_r_squared)

def economic_interpretation(cagr, cdvi, pval):
    trend_msg = "growing" if cagr > 0 else "declining"
    stability_msg = "stable" if cdvi < 20 else "unstable"
    significance = "significant" if pval < 0.05 else "not statistically significant"
    return f"The trend is {trend_msg}, the variability is {stability_msg}, and the trend is {significance}."

def generate_pdf(results_df):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Table page
        fig_table, ax = plt.subplots(figsize=(12, len(results_df)*0.5 + 1))
        ax.axis('off')
        table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        pdf.savefig(fig_table, bbox_inches='tight')
        plt.close(fig_table)

    buffer.seek(0)
    return buffer

# ========== Streamlit App ==========

st.set_page_config(page_title="CAGR & CDVI Analyzer", layout="wide")
st.title("📈 Trend and Instability Analyzer by [SumanEcon]")

st.markdown("""
Welcome to the **CAGR & CDVI Analyzer**!

🔹 **CAGR (Compound Annual Growth Rate)** measures consistent annual growth.  
🔹 **CDVI (Cuddy-Della Valle Index)** measures instability after adjusting for the trend.

👉 Upload your dataset with the **first column as Year or Date**, and the rest as numeric indicators.

🧪 Choose individual indicators or **"All"** to analyze multiple.
""")

uploaded_file = st.file_uploader("Upload a CSV, XLSX, or XLS file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    ext = uploaded_file.name.split('.')[-1]
    data = pd.read_csv(uploaded_file) if ext == "csv" else pd.read_excel(uploaded_file)
    data.columns = data.columns.str.strip()
    data = data.dropna(how='all')

    st.subheader("📋 Data Preview")
    st.dataframe(data.head())

    time_col = data.columns[0]
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    col_option = st.selectbox("Select a column for analysis", options=["All"] + numeric_cols)

    results = []

    if st.button("Run Analysis"):
        columns_to_analyze = numeric_cols if col_option == "All" else [col_option]

        for column in columns_to_analyze:
            temp_df = data[[time_col, column]].dropna()
            if temp_df.empty or temp_df[column].isnull().all():
                continue

            try:
                cagr, p_value, adj_r_squared = compute_cagr(temp_df, column)
                mean_val, std_val, cv_val = compute_statistics(temp_df, column)
                cdvi = compute_cdvi(cv_val, adj_r_squared)
                interp = economic_interpretation(cagr, cdvi, p_value)

                results.append({
                    'Indicator': column,
                    'CAGR (%)': round(cagr, 2),
                    'P-Value': round(p_value, 6),
                    'Mean': round(mean_val, 2),
                    'Standard Deviation': round(std_val, 2),
                    'CV (%)': round(cv_val, 2),
                    'Adjusted R²': round(adj_r_squared, 3),
                    'CDVI': round(cdvi, 2),
                    'Interpretation': interp
                })
            except Exception as e:
                results.append({
                    'Indicator': column,
                    'CAGR (%)': 'Error',
                    'P-Value': 'Error',
                    'Mean': 'Error',
                    'Standard Deviation': 'Error',
                    'CV (%)': 'Error',
                    'Adjusted R²': 'Error',
                    'CDVI': 'Error',
                    'Interpretation': str(e)
                })

        results_df = pd.DataFrame(results)
        st.subheader("📊 Results Table")
        st.dataframe(results_df)

        # ---- Download Table ----
        csv = results_df.to_csv(index=False).encode('utf-8')
        excel = BytesIO()
        results_df.to_excel(excel, index=False, engine='xlsxwriter')
        excel.seek(0)
        st.download_button("Download CSV", csv, file_name="CAGR_CDVI_Results.csv")
        st.download_button("Download Excel", excel, file_name="CAGR_CDVI_Results.xlsx")

        # ---- Generate and Download PDF Summary ----
        st.markdown("### 🧾 Download Full PDF Report")
        pdf_bytes = generate_pdf(results_df)
        st.download_button("Download PDF Report", pdf_bytes, file_name="CAGR_CDVI_Report.pdf", mime="application/pdf")

        # ---- Policy Briefs ----
        st.subheader("📌 Policy Brief Suggestions")
        for idx, row in results_df.iterrows():
            if row['Interpretation'] != "Error":
                st.markdown(f"""
                **{row['Indicator']}:**
                - {row['Interpretation']}
                - 📈 **Policy Tip:** For `{row['Indicator']}`, consider policies to {"enhance growth" if row['CAGR (%)'] > 0 else "arrest decline"}, and if CDVI is high, ensure **price stabilization**, market intelligence, and infrastructure for storage or value addition.
                """)
