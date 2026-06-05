# Hierarchical Clustering Project

This project is structured for hierarchical clustering on a customer dataset.

## Structure

- `data/` - raw dataset files
- `models/` - saved model and scaler artifacts
- `notebooks/` - exploratory data analysis and visualization notebooks
- `train.py` - training script for hierarchical clustering
- `app.py` - prediction API for new customer inputs
- `requirements.txt` - Python dependencies

## Notes

The example dataset is stored in `data/Mall_Customers.csv`. The training script uses `AgglomerativeClustering` to fit the hierarchical clustering model and saves the scaler and model artifacts under `models/`.
