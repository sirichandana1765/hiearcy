import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pickle


def main():
    data_path = "data/Mall_Customers.csv"
    df = pd.read_csv(data_path)

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X_scaled)

    df["Cluster"] = labels
    df.to_csv("data/Mall_Customers_with_clusters.csv", index=False)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/hierarchical_model.pkl", "wb") as f:
        pickle.dump(model, f)

    centroids = np.vstack(
        [X_scaled[labels == k].mean(axis=0) for k in np.unique(labels)]
    )
    np.save("models/cluster_centroids.npy", centroids)

    print("Training complete. Model, scaler, and centroids saved in models.")


if __name__ == "__main__":
    main()
