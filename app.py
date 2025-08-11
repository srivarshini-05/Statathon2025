import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

st.set_page_config(page_title="AutoStat AI Prototype", layout="wide")

st.title("📊 AutoStat AI – Prototype")

# File Upload
uploaded_file = st.file_uploader("Upload Survey Data (CSV/XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Automated Cleaning
    st.subheader("🧹 Automated Data Cleaning")
    imputer = SimpleImputer(strategy="most_frequent")
    df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    st.write("✅ Missing values filled with most frequent values.")

    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    st.write(df_clean.describe())

    # Visualization
    st.subheader("📊 Visualization Example")
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots()
        df_clean[numeric_cols].hist(ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for histogram.")

    # Download Cleaned Data
    st.download_button(
        label="⬇ Download Cleaned CSV",
        data=df_clean.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
