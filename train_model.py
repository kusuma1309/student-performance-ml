# ===============================================================
# STUDENT PERFORMANCE PREDICTION - MODEL TRAINING
# ===============================================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ===============================
# LOAD DATASET
# ===============================

df = pd.read_csv("StudentsPerformance.csv")

print("Dataset Loaded Successfully")
print(df.head())


# ===============================
# PREPROCESSING
# ===============================

encoder = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = encoder.fit_transform(df[col])


# ===============================
# FEATURE ENGINEERING
# ===============================

df["TotalScore"] = df["reading score"] + df["writing score"]


# ===============================
# DEFINE FEATURES & TARGET
# ===============================

X = df.drop("math score", axis=1)
y = df["math score"]


# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# TRAIN MODELS
# ===============================

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)


# ===============================
# MODEL EVALUATION
# ===============================

print("\nLinear Regression")
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2 Score:", r2_score(y_test, lr_pred))

print("\nRandom Forest")
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))

print("\nGradient Boosting")
print("RMSE:", np.sqrt(mean_squared_error(y_test, gb_pred)))
print("R2 Score:", r2_score(y_test, gb_pred))


# ===============================
# SAVE BEST MODEL
# ===============================

pickle.dump(gb, open("student_model.pkl", "wb"))

print("\nModel Saved Successfully!")