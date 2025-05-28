import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Linear Regression Playground", layout="centered")

st.title("📈 Linear Regression Playground")
st.write("Upload your dataset, pick a feature and target, and visualize a simple linear regression model.")

def read_df(file) -> pd.DataFrame:
    return pd.read_csv(file)

def train_model(file):
    df = read_df(file)

    st.success("Data loaded successfully!")
    st.write("Preview of dataset:", df.head())

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_columns) < 2:
        st.error("Need at least two numeric columns for regression.")
    else:
        # Select feature and target
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Choose the feature (X)", numeric_columns)
        with col2:
            y_col = st.selectbox("Choose the target (Y)", [col for col in numeric_columns if col != x_col])

        if x_col and y_col:
            # Train model
            X = df[[x_col]]
            y = df[y_col]

            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)
            coef = model.coef_[0]
            intercept = model.intercept_
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)

            st.subheader("🔍 Model Results")
            st.markdown(f"""
                - **Intercept:** {intercept:.4f}  
                - **Coefficient for {x_col}:** {coef:.4f}  
                - **R² Score:** {r2:.4f}  
                - **MSE:** {mse:.4f}
                """)

            # Plot
            st.subheader("📊 Regression Plot")
            fig = px.scatter(df, x=x_col, y=y_col, opacity=0.6, title="Linear Regression Fit")
            fig.add_scatter(x=df[x_col], y=y_pred, mode='lines', name='Regression Line')
            st.plotly_chart(fig, use_container_width=True)

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    train_model(uploaded_file)
else:
    st.info("Please upload a CSV file to get started.")
