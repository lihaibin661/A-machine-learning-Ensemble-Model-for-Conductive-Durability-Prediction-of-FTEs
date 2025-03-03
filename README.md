# Ensemble Regression Models for Square Resistance Prediction

This repository contains implementations of four regression models used to predict square resistance values. The models included are:

** Linear Regression (LR) Model **  
  This model preprocesses the data by filling missing values, applying standardization, and expanding features with polynomial transformations. A fully connected neural network built with TensorFlow/Keras is then used for regression, capturing both linear and some nonlinear relationships.

** Random Forest (RF) Model **
  Utilizes scikit-learn’s RandomForestRegressor to perform regression on preprocessed data. This model benefits from the ensemble of decision trees to provide robust predictions.

- **Convolutional Neural Network (CNN) Model**  
  Processes raw data by splitting the "ZTO/Ag" column into separate features, applying one-hot encoding, and concatenating with other numerical features. A 1D CNN built with TensorFlow/Keras is then used to extract local patterns and predict square resistance.

- **Stacked Ensemble Model (XGBoost Meta-model)**  
  Combines the predictions from the LR, RF, and CNN models as input features to an XGBoost regressor. This meta-model aims to leverage the complementary strengths of the base models for improved final prediction accuracy. The training process includes visualization of an epoch-loss curve (using RMSE as the metric) and a scatter plot comparing actual versus predicted values.

---

Repository Structure
├── data/ │ ├── 更新线性回归600.csv # Data file used by the Linear Regression model │ └── 600CNN5.13.csv # Data file used by the CNN and RF models ├── notebooks/ # (Optional) Jupyter notebooks for experiments and visualization ├── models/ # (Optional) Saved trained models ├── README.md # This file └── main.py # Main script: data preprocessing, model training, stacking, and evaluation

## Requirements ## 
Ensure you have Python 3.7 or higher installed. The primary Python libraries required for this project are:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [xgboost](https://xgboost.readthedocs.io/)
- [tensorflow](https://www.tensorflow.org/)
- [shap](https://shap.readthedocs.io/)

You can install all dependencies using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib xgboost tensorflow shap
## Usage Instructions ##
git clone https://github.com/yourusername/ensemble-regression-models.git
cd ensemble-regression-models
