#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
def load_data():
    dataset = pd.read_csv(r"C:\Users\shruti sahrawat\Desktop\ML\datasets\Salary_Data.csv")
    X = dataset.iloc[:, :-1].values  # Independent variable (Years of Experience)
    y = dataset.iloc[:, 1].values  # Dependent variable (Salary)
    return X, y

# Train the model
def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=101)

    lrm = LinearRegression()
    lrm.fit(X_train, y_train)  # Train the model

    # Save test data for evaluation
    pickle.dump(lrm,open("model.pkl","wb"))
    return lrm, X_test, y_test

# Predict function
def predict(model, years_of_experience):
    return model.predict([[years_of_experience]])

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "R2 Score": metrics.r2_score(y_test, y_pred),
        "MSE": metrics.mean_squared_error(y_test, y_pred),
        "MAE": metrics.mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    }

# Run training when executed directly
if __name__ == "__main__":
    model, X_test, y_test = train_model()
    print("Model trained successfully!")
    print("Evaluation Metrics:", evaluate_model(model, X_test, y_test))























