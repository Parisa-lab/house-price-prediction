
# House Price Prediction

An end-to-end machine learning project that predicts residential house sale prices using two regression approaches: Random Forest and a neural network implemented with NumPy. The project includes data preprocessing, model training, evaluation, automated tests, feature importance analysis, and an interactive Streamlit application.

## Live Demo

Try the deployed Streamlit application:

[House Price Prediction — Streamlit](https://house-price-prediction-vrtvevjqxqm7yuhvyx7z7v.streamlit.app/)

The live deployment may not yet include the neural network integration. The updated application must be pushed and successfully redeployed before the live demo can be considered current.

## Overview

This project uses the Ames Housing dataset to predict residential sale prices from six selected property features.

The project demonstrates an end-to-end machine learning workflow:

- Data loading and validation
- Feature selection and preprocessing
- Train/test splitting
- Random Forest regression
- Neural network implementation and training
- Model evaluation against a mean-prediction baseline
- Feature importance analysis
- Model serialization and loading
- Input validation and automated testing
- Interactive web application development

## Features

Both models use the same six input features:

| Feature | Description |
|---|---|
| `OverallQual` | Overall material and finish quality |
| `GrLivArea` | Above-ground living area |
| `GarageCars` | Garage capacity |
| `TotalBsmtSF` | Total basement area |
| `FullBath` | Number of full bathrooms |
| `YearBuilt` | Original construction year |

These features provide a compact representation of important property characteristics while keeping the application straightforward to use.

## Machine Learning Models

### 1. Random Forest Regressor

The Random Forest model uses Scikit-learn and a preprocessing pipeline.

The training workflow includes:

1. Loading and validating the dataset.
2. Selecting the six input features and `SalePrice` target.
3. Handling missing numerical values using median imputation.
4. Splitting the data into training and test sets.
5. Training the Random Forest regressor.
6. Evaluating predictions on held-out test data.
7. Generating a feature importance plot.
8. Saving the trained pipeline to `models/model.pkl`.

Training configuration:

- `n_estimators = 300`
- `max_depth = 12`
- `random_state = 42`
- `n_jobs = -1`

### 2. Neural Network

The second model is a fully connected neural network implemented using NumPy rather than a deep-learning framework.

Architecture:

- Input layer: 6 features
- First hidden layer: 32 neurons with ReLU activation
- Second hidden layer: 16 neurons with ReLU activation
- Output layer: 1 predicted sale price

The implementation includes:

- He-style weight initialization for the ReLU hidden layers
- Mini-batch gradient descent
- Mean squared error loss
- Feature and target standardization
- Validation-based early stopping
- Restoration of the best validation weights
- Model parameter serialization in NumPy's `.npz` format
- Input and artifact validation during inference

The neural network training workflow uses separate training, validation, and test subsets. Scaling parameters are calculated from the training subset to avoid using validation or test information during preprocessing.

The trained network is saved to `models/neural_network.npz`.

## Model Performance

The following results are from the current recorded model runs.

| Metric | Random Forest | Neural Network |
|---|---:|---:|
| Test R² | 0.8889 | 0.8678 |
| Test RMSE | $29,190.00 | $31,844.77 |

Both models were evaluated using the same 20% held-out test split with `random_state = 42`. The neural network uses an additional validation subset drawn from the training portion.

The Random Forest currently has the higher recorded R² and lower RMSE. These results describe performance on this dataset and split; they do not establish that the same model will perform better on every dataset.

### R² Score

R² measures how much variation in the target values is explained by a model relative to a mean-prediction baseline. A score closer to 1 generally indicates better predictive performance.

### Root Mean Squared Error

RMSE measures prediction error in the target's units—in this case, dollars. It penalizes larger errors more heavily than smaller ones. Individual prediction errors can be substantially larger or smaller than the RMSE.

## Baseline Comparison

The neural network training script also evaluates a simple baseline that predicts the mean sale price of the training subset for every test example.

| Metric | Mean Baseline |
|---|---:|
| Test R² | -0.0008 |
| Test RMSE | $87,613.78 |

The neural network substantially outperforms this baseline on the recorded test split. A baseline provides useful context because it shows whether a trained model improves on a simple prediction strategy.

## Random Forest Feature Importance

The Random Forest model produces the following feature importance values:

| Feature | Importance |
|---|---:|
| Overall Quality | 58.6% |
| Living Area | 19.8% |
| Basement Area | 11.2% |
| Year Built | 5.8% |
| Garage Capacity | 3.2% |
| Full Bathrooms | 1.4% |

Overall quality has the highest feature importance in the trained Random Forest, followed by above-ground living area and basement area.

Feature importance measures how the fitted Random Forest uses features to reduce impurity. It does not establish causation and should not be interpreted as the effect of changing a property characteristic.

The generated visualization is saved to `feature_importance.png`.

## Streamlit Application

The Streamlit interface lets users select a model and enter the six property characteristics to receive an estimated sale price.

The application provides:

- Random Forest and neural network prediction options
- Input controls for all six features
- Estimated sale price
- Recorded model performance metrics
- Random Forest feature importance visualization
- Project information and model descriptions
- Model-loading and prediction error handling

The Random Forest and neural network artifacts are stored separately, allowing either model to be loaded independently.

## Project Structure

```text
house-price-prediction/
├── app.py
├── train.py
├── neural_network.py
├── train_neural_network.py
├── requirements.txt
├── README.md
├── .gitignore
├── feature_importance.png
├── data/
│   └── train.csv
├── models/
│   ├── model.pkl
│   └── neural_network.npz
└── tests/
    └── test_app.py
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Pytest
- Git and GitHub

## Dataset

This project uses the **House Prices: Advanced Regression Techniques** dataset from Kaggle.

The dataset contains residential property information from Ames, Iowa, including property quality, size, construction characteristics, and other attributes.

The expected training file is `data/train.csv`. The dataset is not recreated by the training scripts; obtain it from the original dataset source if it is not already present.

## How to Run the Project

### 1. Clone the repository

Replace the placeholder with the actual repository URL.

```bash
git clone YOUR_REPOSITORY_URL
cd house-price-prediction
```

### 2. Install dependencies

Use Python 3.10 or newer and install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Prepare the dataset

Place the Kaggle training file at:

```text
data/train.csv
```

The dataset must contain the six selected features and the `SalePrice` target.

### 4. Train the Random Forest model

```bash
python train.py
```

This command trains and evaluates the Random Forest pipeline, generates the feature importance plot, and saves the model to:

```text
models/model.pkl
```

### 5. Train the neural network

```bash
python train_neural_network.py
```

This command validates the selected data, creates training, validation, and test subsets, trains the neural network, evaluates its performance against the mean baseline, and saves the artifact to:

```text
models/neural_network.npz
```

### 6. Run automated tests

Install Pytest if it is not already available:

```bash
python -m pip install pytest
```

Then run the test suite from the repository root:

```bash
python -m pytest -v
```

The tests check model artifacts, feature order, finite predictions, Random Forest feature importances, neural network input validation, and missing-model handling.

The current test suite has passed all eight tests in the development environment. Results can vary if the dependencies or model artifacts change.

### 7. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will display a local URL in the terminal. Open that URL in your browser.

## Limitations

### Limited Feature Set

The original dataset contains many more variables than the six features used in this project. Additional features may improve predictions but would require corresponding preprocessing, training, testing, and application changes.

### Dataset and Evaluation

The reported results use a single train/test split. Cross-validation and evaluation on additional data would provide stronger evidence about generalization.

### Model Comparison

The two models use different learning approaches and training procedures. A single held-out split provides a useful initial comparison but does not establish which approach is consistently superior.

### Real-World Use

Predictions are intended for educational and portfolio purposes. They are not professional real-estate valuations and should not be used as the sole basis for financial decisions.

## Future Improvements

- Cross-validation and repeated experiments
- Hyperparameter tuning
- Additional regression models
- Feature engineering and target transformations
- Model explainability methods
- Prediction intervals and uncertainty estimates
- Expanded automated tests
- Continuous integration and deployment
- Monitoring for changes in data and model performance

## Portfolio Project

This project demonstrates two approaches to regression and the steps required to turn trained models into an interactive application.

```text
Dataset
   |
Feature Selection and Validation
   |
Train / Validation / Test Splits
   |
   +-----------------------+
   |                       |
Random Forest          NumPy Neural Network
   |                       |
Evaluation             Evaluation
   |                       |
Saved Pipeline         Saved Parameters
   |                       |
   +-----------+-----------+
               |
        Streamlit Application
               |
        Interactive Predictions
```

## Author

Parisa Barzegari

AI and Data Science portfolio project.
