# rf_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load data for the RF model from "600CNN5.13.csv"
data = pd.read_csv('data/600CNN5.13.csv')  # Adjust the path accordingly

# Split the "ZTO/Ag" column into two separate features
data[['ZTO', 'Ag']] = data['ZTO/Ag'].str.split(',', expand=True)
data['ZTO'] = pd.to_numeric(data['ZTO'])
data['Ag'] = pd.to_numeric(data['Ag'])

# One-hot encode the "ZTO" and "Ag" features
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(data[['ZTO', 'Ag']])

# Concatenate the encoded features with additional numerical features: "Number of Bending" and "Storage time"
X = np.hstack((encoded, data[['Number of Bending', 'Storage time']].values))
y = data['Square resistance'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
predictions = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("RF Model Test MSE: {:.4f}".format(mse))

# Save the predictions for the stacking ensemble
pd.DataFrame({'rf_pred': predictions}).to_csv('rf_predictions.csv', index=False)
