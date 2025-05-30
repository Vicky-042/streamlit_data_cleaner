import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="🧼 Smart Data Cleaner", layout="wide")
st.title("🧹 Smart Data Cleaner")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# History storage
if "cleaning_history" not in st.session_state:
    st.session_state.cleaning_history = []

def clean_data(df, options, regex_replacements):
    df_cleaned = df.copy()
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()

    # Fill missing values
    if options.get("fill_missing"):
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            else:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # Remove outliers
    if options.get("remove_outliers"):
        for col in numeric_cols:
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower) & (df_cleaned[col] <= upper)]

    # Normalize selected columns
    if options.get("normalize") and options.get("normalize_cols"):
        scaler = StandardScaler()
        selected_cols = options["normalize_cols"]
        df_cleaned[selected_cols] = scaler.fit_transform(df_cleaned[selected_cols])

    # Apply regex replacements
    if options.get("regex_clean"):
        for col, pattern, repl in regex_replacements:
            if col in df_cleaned.columns:
                try:
                    df_cleaned[col] = df_cleaned[col].astype(str).apply(lambda x: re.sub(pattern, repl, x))
                except re.error:
                    st.warning(f"Invalid regex pattern in column {col}: {pattern}")

    # Lowercase columns (after regex)
    if options.get("lowercase_columns"):
        df_cleaned.columns = df_cleaned.columns.str.lower()

    return df_cleaned

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Original Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📊 Data Summary")
    st.write(df.describe(include='all').transpose())

    if st.checkbox("📈 Show Charts"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in num_cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🛠️ Select Cleaning Options")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fill_missing = st.checkbox("Fill Missing Values", value=True)
    with col2:
        normalize = st.checkbox("Normalize Numeric Columns")
    with col3:
        remove_outliers = st.checkbox("Remove Outliers")
    with col4:
        lowercase_columns = st.checkbox("Lowercase Column Names")

    normalize_cols = []
    if normalize:
        normalize_cols = st.multiselect("Select columns to normalize", df.select_dtypes(include=np.number).columns.tolist())

    auto_clean = st.checkbox("Auto Clean (Apply All)", value=False)

    st.subheader("🔤 Regex-Based Cleaning")
    regex_clean = st.checkbox("Enable Regex Cleaning")
    regex_replacements = []
    if regex_clean:
        regex_count = st.number_input("Number of Regex Rules", min_value=1, max_value=10, value=1)
        for i in range(regex_count):
            st.markdown(f"**Rule {i+1}**")
            col_name = st.text_input(f"Column Name {i+1}", key=f"col_{i}")
            pattern = st.text_input(f"Regex Pattern {i+1}", key=f"pattern_{i}")
            replacement = st.text_input(f"Replacement {i+1}", key=f"repl_{i}")
            regex_replacements.append((col_name, pattern, replacement))

    options = {
        "fill_missing": auto_clean or fill_missing,
        "normalize": auto_clean or normalize,
        "normalize_cols": normalize_cols,
        "remove_outliers": auto_clean or remove_outliers,
        "lowercase_columns": auto_clean or lowercase_columns,
        "regex_clean": auto_clean or regex_clean
    }

    if st.button("🧼 Clean My Data"):
        cleaned_df = clean_data(df, options, regex_replacements)
        st.subheader("🔍 Cleaned Data Preview")
        st.dataframe(cleaned_df.head(), use_container_width=True)

        st.session_state.cleaning_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "options": options,
            "regex": regex_replacements
        })

        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )

        st.subheader("📚 Cleaning History")
        for i, h in enumerate(st.session_state.cleaning_history[::-1]):
            st.write(f"**Run {len(st.session_state.cleaning_history) - i}:**", h)

        if st.button("📤 Export Cleaning History"):
            history_df = pd.DataFrame(st.session_state.cleaning_history)
            history_csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download History Log",
                data=history_csv,
                file_name='cleaning_history.csv',
                mime='text/csv'
            )
