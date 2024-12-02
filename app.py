import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("Fish.csv")

# Sidebar for navigation
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Menu", ["Home", "Visualisasi", "Model Prediksi"])

# Home page
if menu == "Home":
    st.title("Prediksi Berat Ikan")
    st.write("Dataset ikan digunakan untuk memprediksi berat ikan berdasarkan pengukuran fisik ikan.")
    st.dataframe(data)

    if st.checkbox("Tampilkan Statistik Deskriptif"):
        st.write(data.describe())

# Visualizations
elif menu == "Visualisasi":
    st.title("Visualisasi Data Ikan")
    
    # Scatter plot
    st.subheader("Visualisasi Scatter Plot")
    if st.checkbox("Tampilkan Scatter Plot"):
        x_axis = st.selectbox("Pilih sumbu X", options=data.columns[1:], index=0)
        y_axis = st.selectbox("Pilih sumbu Y", options=data.columns[1:], index=1)
        st.write(f"Scatter plot {x_axis} vs {y_axis}")
        scatter_chart = alt.Chart(data).mark_point().encode(
            x=x_axis, 
            y=y_axis,
            color="Species"
        )
        st.altair_chart(scatter_chart, use_container_width=True)

    # Histogram
    st.subheader("Visualisasi Histogram")
    if st.checkbox("Tampilkan Histogram"):
        feature = st.selectbox("Pilih Fitur untuk Histogram", options=data.columns[1:], index=0)
        st.write(f"Histogram untuk {feature}")
        plt.figure(figsize=(8, 6))
        plt.hist(data[feature], bins=20, color="skyblue", edgecolor="black")
        plt.xlabel(feature)
        plt.ylabel("Frekuensi")
        st.pyplot()

    # Heatmap untuk korelasi
st.subheader("Heatmap Korelasi Antar Fitur")
if st.checkbox("Tampilkan Heatmap Korelasi"):
    # Pisahkan kolom numerik dan kategorikal
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Hitung korelasi hanya untuk data numerik
    correlation_matrix = numeric_data.corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    st.pyplot()

# Model prediction
elif menu == "Model Prediksi":
    st.title("Model Prediksi Berat Ikan")
    
    # Prepare the data
    X = data.drop(columns=["Weight", "Species"])  # Drop target columns (Weight and Species)
    y = data["Weight"]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit the model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    # Show predictions
    st.write("Prediksi Berat Ikan vs Berat Sebenarnya:")
    prediction_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.write(prediction_df.head())

    # Plot predictions
    st.subheader("Plot Prediksi vs Sebenarnya")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    ax.set_xlabel("Actual Weight")
    ax.set_ylabel("Predicted Weight")
    st.pyplot(fig)

    # Visualize the model's residuals (difference between actual and predicted values)
    st.subheader("Residuals Plot")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, residuals)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel("Actual Weight")
    ax2.set_ylabel("Residuals")
    st.pyplot(fig2)
