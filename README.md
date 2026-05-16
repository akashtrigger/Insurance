<!-- Insurance Cost Prediction using Artificial Neural Network (ANN) -->
<!-- Project Overview -->

This project demonstrates how to build an Insurance Cost Prediction System using TensorFlow, Keras, and Artificial Neural Networks (ANN).

The model predicts medical insurance charges based on customer information such as:

Age
Gender
BMI
Smoking habits
Number of children
Region
Technologies Used
Python
TensorFlow
Keras
Pandas
NumPy
Matplotlib
Scikit-learn
Dataset Information

The dataset contains insurance customer details and medical charges.

Input Features
Feature	Description
Age	Age of customer
Sex	Male/Female
BMI	Body Mass Index
Children	Number of children
Smoker	Smoking status
Region	Residential area
Target Variable
Variable	Description
Charges	Medical insurance cost
Project Workflow
<!-- 1. Import Libraries -->
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

Libraries are imported for:

Data processing
Neural network modeling
Visualization
<!-- 2. Load Dataset -->
df = pd.read_csv("insurance.csv")

The insurance dataset is loaded using Pandas.

<!-- 3. Data Preprocessing -->
Handle Categorical Data
df = pd.get_dummies(df, drop_first=True)

Categorical columns are converted into numerical format.

Separate Features and Target
X = df.drop("charges", axis=1)
y = df["charges"]
X contains input features
y contains insurance charges
Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Dataset is divided into:

80% Training Data
20% Testing Data
Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Scaling improves ANN performance.

<!-- 4. Build ANN Model -->
model = keras.Sequential([

    keras.layers.Dense(
        100,
        input_shape=(X_train.shape[1],),
        activation='relu'
    ),

    keras.layers.Dense(
        50,
        activation='relu'
    ),

    keras.layers.Dense(1)
])
Layers Explanation
Layer	Description
Dense(100)	First hidden layer
Dense(50)	Second hidden layer
Dense(1)	Output layer for regression
<!-- 5. Model Compilation -->
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
Compilation Components
Component	Purpose
Adam	Optimizer
Mean Squared Error	Loss function
MAE	Mean Absolute Error
<!-- 6. Training the Model -->
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32
)

The ANN model learns relationships between customer details and insurance costs.

<!-- 7. Model Evaluation -->
loss, mae = model.evaluate(X_test, y_test)

The model is evaluated using:

Loss
Mean Absolute Error
<!-- 8. Prediction -->
predictions = model.predict(X_test)

The model predicts insurance charges for new customers.

Applications
Insurance premium estimation
Healthcare analytics
Risk assessment systems
Medical cost prediction
Financial forecasting
Advantages of ANN
Learns complex relationships
Improves prediction accuracy
Handles nonlinear data effectively
Useful for regression problems
Conclusion

This project successfully builds an Insurance Cost Prediction System using an Artificial Neural Network (ANN).
The model predicts medical insurance charges based on customer information and demonstrates the application of deep learning in healthcare and financial analytics.