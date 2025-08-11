import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

st.set_page_config(page_title="AutoStat AI Prototype", layout="wide")
st.title("📊 AutoStat AI – Prototype")

# PDF report generator
def generate_pdf(df_summary, plot_image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, "AutoStat AI – Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 8, "Summary Statistics:\n")
    pdf.set_font("Courier", size=10)
    pdf.multi_cell(0, 5, df_summary.to_string())

    if plot_image:
        pdf.ln(10)
        pdf.image(plot_image, x=10, y=None, w=180)

    return pdf.output(dest="S").encode("latin-1")

# File upload
uploaded_file = st.file_uploader("Upload Survey Data (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Missing value handling
    st.subheader("🧹 Automated Data Cleaning – Missing Values")
    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    st.write("✅ Missing values filled with most frequent values.")

    # Outlier detection
    st.subheader("🚨 Outlier Detection")
    numeric_df = df_imputed.select_dtypes(include=np.number)
    if not numeric_df.empty:
        z_scores = np.abs(stats.zscore(numeric_df))
        outliers = (z_scores > 3).any(axis=1)
        st.write(f"Found {outliers.sum()} potential outlier rows.")
        
        if st.checkbox("Show outlier rows"):
            st.dataframe(df_imputed[outliers])
        
        df_clean = df_imputed[~outliers].reset_index(drop=True)
    else:
        st.info("No numeric columns found for outlier detection.")
        df_clean = df_imputed

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    summary_stats = df_clean.describe()
    st.write(summary_stats)

    # Visualization
    st.subheader("📊 Visualization Example")
    if not numeric_df.empty:
        fig, ax = plt.subplots()
        df_clean.hist(ax=ax)
        st.pyplot(fig)
    else:
        fig = None

    # Downloads
    st.download_button(
        label="⬇ Download Cleaned CSV",
        data=df_clean.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    pdf_data = generate_pdf(summary_stats, None)
    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_data,
        file_name="autostat_ai_report.pdf",
        mime="application/pdf"
    )
