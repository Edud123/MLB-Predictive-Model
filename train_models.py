import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load the data
df = pd.read_csv("merged.csv")

# Prepare features
feature_cols = [col for col in df.columns if 'lag' in col]
X = df[feature_cols]
y = df['WR']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Save models
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Models trained and saved successfully!")