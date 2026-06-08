import os
import pickle
import numpy as np
import streamlit as st
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CENTROIDS_PATH = os.path.join(MODEL_DIR, "cluster_centroids.npy")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
centroids = np.load(CENTROIDS_PATH)

st.set_page_config(page_title="Hierarchical Clustering", layout="centered")

st.title("Hierarchical Clustering Demo")
st.markdown(
    "Use the form below to predict the cluster for a new customer based on age, annual income, and spending score."
)

with st.sidebar:
    st.header("Input Features")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=300, value=60)
    score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    show_data = st.checkbox("Show sample dataset", value=False)
    show_clusters = st.checkbox("Show cluster counts", value=False)

if st.button("Predict cluster"):
    features = np.array([age, income, score]).reshape(1, -1)
    X_scaled = scaler.transform(features)
    distances = np.linalg.norm(centroids - X_scaled, axis=1)
    cluster = int(np.argmin(distances))
    st.success(f"Predicted cluster: {cluster}")

    st.write("### Input values")
    st.write({"Age": age, "Annual Income (k$)": income, "Spending Score (1-100)": score})

if show_data:
    df = pd.read_csv(DATA_PATH)
    st.write("### Sample dataset")
    st.dataframe(df.head(20))

if show_clusters:
    df = pd.read_csv(DATA_PATH)
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features]
    X_scaled = scaler.transform(X)
    labels = np.argmin(np.linalg.norm(centroids[:, None, :] - X_scaled[None, :, :], axis=2), axis=0)
    counts = pd.Series(labels).value_counts().sort_index()
    st.write("### Cluster counts")
    st.bar_chart(counts)
