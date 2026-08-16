# House Price Prediction

A machine learning project that predicts residential house sale prices using a Random Forest regression model.

## Live Demo

Try the deployed Streamlit application:

https://house-price-prediction-vrtvevjqxqm7yuhvyx7z7v.streamlit.app/

## Overview

This project uses the Ames Housing dataset to build a regression model for predicting house sale prices.

The project demonstrates an end-to-end machine learning workflow:

- Data loading
- Feature selection
- Missing-value handling
- Train/test splitting
- Model training
- Model evaluation
- Feature importance analysis
- Model serialization
- Interactive web application
- Cloud deployment

## Features

The model uses six selected features from the original dataset:

| Feature | Description |
|---|---|
| `OverallQual` | Overall material and finish quality |
| `GrLivArea` | Above-ground living area |
| `GarageCars` | Garage capacity |
| `TotalBsmtSF` | Total basement area |
| `FullBath` | Number of full bathrooms |
| `YearBuilt` | Original construction year |

These features provide a compact representation of important property characteristics while keeping the deployed application simple and easy to use.

## Machine Learning Model

The project uses a **Random Forest Regressor**.

The training pipeline consists of:

1. Loading the Ames Housing dataset
2. Selecting six features
3. Handling missing numerical values using median imputation
4. Splitting the data into training and test sets
5. Training a Random Forest regression model
6. Evaluating the model on unseen test data
7. Analyzing feature importance
8. Saving the trained model as a pickle file

## Training Configuration

The dataset is divided into:

- **80% training data**
- **20% test data**

The model uses:

- `n_estimators = 300`
- `max_depth = 12`
- `random_state = 42`
- `n_jobs = -1`

The fixed random state makes the experiment reproducible.

## Model Performance

The model was evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| R² Score | 0.8889 |
| RMSE | $29,190 |

### R² Score

The model achieved an R² score of **0.8889**.

This means that the model explains approximately **88.9% of the variation** in house sale prices in the test set.

### RMSE

The Root Mean Squared Error (RMSE) was approximately **$29,190**.

RMSE measures the typical magnitude of prediction error in the same units as the target variable. Individual prediction errors can be substantially larger or smaller.

## Feature Importance

The trained Random Forest model produced the following feature importance values:

| Feature | Importance |
|---|---:|
| Overall Quality | 58.6% |
| Living Area | 19.8% |
| Basement Area | 11.2% |
| Year Built | 5.8% |
| Garage Capacity | 3.2% |
| Full Bathrooms | 1.4% |

The results show that **Overall Quality** is the most influential feature in the model, followed by **Living Area** and **Basement Area**.

## Streamlit Application

The trained model is integrated into an interactive Streamlit application.

Users can enter:

- Overall Quality
- Living Area
- Garage Capacity
- Basement Area
- Full Bathrooms
- Year Built

The application sends these values to the trained machine learning pipeline and returns an estimated sale price.

The application also displays:

- Model performance
- R² score
- RMSE
- Feature importance
- Project information

## Project Structure

~~~text
house-price-prediction/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── train.csv
│
├── models/
│   └── model.pkl
│
└── notebooks/
    └── analysis.ipynb
~~~

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Git
- GitHub

## Dataset

This project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The dataset contains residential property information from Ames, Iowa, including information about property quality, size, construction, and other characteristics.

## How to Run the Project

### 1. Clone the repository

~~~bash
git clone YOUR_REPOSITORY_URL
cd house-price-prediction
~~~

### 2. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 3. Prepare the dataset

Place the Kaggle `train.csv` file inside:

~~~text
data/train.csv
~~~

### 4. Train the model

~~~bash
python train.py
~~~

This will:

- Train the Random Forest model
- Evaluate the model
- Generate the feature importance plot
- Save the trained model to:

~~~text
models/model.pkl
~~~

### 5. Run the Streamlit application

~~~bash
streamlit run app.py
~~~

The application will open in your browser.

## Limitations

This project is intentionally designed as a compact machine learning project and therefore has several limitations.

### Limited Feature Set

The original dataset contains many more variables than the six features used in this project.

Using only six features makes the application easier to understand and use, but it also means that potentially useful information is excluded from the model.

### Model Selection

Only a Random Forest Regressor is used in the current version.

Other models, such as Gradient Boosting, XGBoost, or LightGBM, could potentially achieve better predictive performance.

### Evaluation

The reported performance is based on a single train/test split.

Cross-validation would provide a more robust estimate of how well the model generalizes to unseen data.

### Real-World Use

The predictions are intended for educational and portfolio purposes.

They should not be interpreted as professional real-estate valuations or used as the sole basis for financial decisions.

## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- K-fold cross-validation
- Comparing multiple regression algorithms
- Testing additional features
- Feature engineering
- Log-transforming the target variable
- Model explainability with SHAP
- Prediction uncertainty or prediction intervals
- More advanced interactive visualizations
- Automated model evaluation
- Continuous deployment

## Portfolio Project

This project demonstrates the complete process of taking a machine learning model from a raw dataset to a deployed web application.

The workflow is:

~~~text
Dataset
   ↓
Feature Selection
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Model Serialization
   ↓
Streamlit Application
   ↓
Cloud Deployment
~~~

## Live Application

The deployed application is available here:

https://house-price-prediction-vrtvevjqxqm7yuhvyx7z7v.streamlit.app/

## Author

Parisa Barzegari 

AI and Data Science portfolio project.