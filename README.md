
# Heart Disease Prediction Using Machine Learning

This project trains several classification models (Logistic Regression, Decision Tree, Random Forest, and a simple Neural Network) to predict the presence of heart disease using clinical features.

## Files
- `data/heart.csv` - sample CSV included. Replace with full dataset for better performance.
- `train.py` - training script that creates and saves models in `models/`.
- `app.py` - Streamlit app for quick predictions using the saved Random Forest model.
- `models/` - folder where trained models and scaler will be saved.
- `requirements.txt` - Python dependencies.

## How to run
1. (Optional) create a virtual environment
2. `pip install -r requirements.txt`
3. `python train.py`  # trains models and saves them into models/
4. `streamlit run app.py`  # launches the web UI

## Notes
- The included `data/heart.csv` is a tiny sample for testing. Download the full dataset (e.g., Cleveland heart dataset) and replace `data/heart.csv` for proper training.
