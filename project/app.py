from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/hierarchical_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    features = [
        payload.get("Age"),
        payload.get("Annual Income (k$)"),
        payload.get("Spending Score (1-100)")
    ]

    X = np.array(features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    cluster = int(model.fit_predict(X_scaled)[0]) if hasattr(model, "fit_predict") else int(model.predict(X_scaled)[0])

    return jsonify({"cluster": cluster})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
