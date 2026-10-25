# SuperKart Sales Forecasting — MLOps Project

## Objective
Build an automated regression pipeline to estimate
`Product_Store_Sales_Total` from product and store attributes.

## Architecture
- Data preparation: pandas and scikit-learn
- Candidate models: Decision Tree, Random Forest, Gradient Boosting
- Model selection: 5-fold cross-validation RMSE on training data
- Final evaluation: held-out test set using MAE, RMSE and R²
- Model artifact: joblib
- CI/CD: GitHub Actions
- Deployment: Streamlit Community Cloud
- Container configuration: Dockerfile

## Project structure
- `data/superkart.csv`: source dataset
- `prepare_data.py`: cleaning and train/test split
- `train_model.py`: model tuning, evaluation and artifact creation
- `models/`: trained model and feature schema
- `reports/metrics.json`: model comparison and test metrics
- `deployment/app.py`: prediction interface
- `deployment/Dockerfile`: container configuration
- `.github/workflows/pipeline.yml`: automated pipeline
- `requirements.txt`: Python dependencies

## Reproduce the pipeline
```bash
pip install -r requirements.txt
python prepare_data.py
python train_model.py
streamlit run deployment/app.py
```

## Evaluation
The model is selected using cross-validation on the training set.
The held-out test set is used for final evaluation.

## Links
- GitHub repository: ADD_YOUR_REPOSITORY_URL
- Live Streamlit app: ADD_YOUR_STREAMLIT_URL

## Limitations
The provided feature set has no date or time column. Therefore,
this model predicts sales for a product-store record; it does not
independently produce weekly, monthly, quarterly or annual time-series
forecasts.
