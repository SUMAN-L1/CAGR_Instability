import streamlit as st
import pandas as pd
import numpy as np

# Function to calculate CAGR
def calculate_cagr(df, start_col, end_col, period):
    df['CAGR'] = ((df[end_col] / df[start_col]) ** (1 / period)) - 1
    return df['CAGR']

# Function to calculate basic statistics
def calculate_statistics(df, col):
    mean = df[col].mean()
    std_dev = df[col].std()
    cv = std_dev / mean
    return mean, std_dev, cv

# Function to calculate instability using the CDVI index
def calculate_cdvi(cv, adj_r_squared):
    cdvi = cv * np.sqrt(1 - adj_r_squared)
    return cdvi

st.title('Data Analysis App')

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.write(df.head())

    # Select columns for CAGR calculation
    col = st.selectbox('Select column for analysis', df.columns)
    start_col = st.selectbox('Select start column for CAGR', df.columns)
    end_col = st.selectbox('Select end column for CAGR', df.columns)
    period = st.number_input('Enter the number of periods for CAGR calculation', min_value=1, step=1)

    if st.button('Calculate'):
        cagr = calculate_cagr(df, start_col, end_col, period)
        mean, std_dev, cv = calculate_statistics(df, col)
        adj_r_squared = st.number_input('Enter adjusted R-squared value', min_value=0.0, max_value=1.0, step=0.01)
        cdvi = calculate_cdvi(cv, adj_r_squared)

        st.write("CAGR Calculation:")
        st.write(f"CAGR: {cagr.head()}")

        st.write("Basic Statistics:")
        st.write(f"Mean: {mean}")
        st.write(f"Standard Deviation: {std_dev}")
        st.write(f"Coefficient of Variation: {cv}")

        st.write("CDVI Calculation:")
        st.write(f"CDVI: {cdvi}")

        # Displaying all results in a dataframe
        results = pd.DataFrame({
            'CAGR': cagr,
            'Mean': mean,
            'Standard Deviation': std_dev,
            'Coefficient of Variation': cv,
            'CDVI': cdvi
        }, index=[0])

        st.write("All Results:")
        st.write(results)
