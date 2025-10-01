# train_and_save.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# 1. Dataset लोड करा
df = pd.read_csv("car data.csv")

# 2. Data Preprocessing
df['Age'] = 2025 - df['Year']   # वर्ष -> कारचं वय
X = df[['Age', 'Present_Price', 'Kms_Driven', 'Owner']]

# Categorical features जोडून one-hot encode करा
X = pd.concat([X, df[['Fuel_Type','Seller_Type','Transmission']]], axis=1)
X = pd.get_dummies(X, drop_first=True)

y = df['Selling_Price']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model Train करा
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Accuracy check करा
y_pred = model.predict(X_test)
print("✅ R2 Score:", r2_score(y_test, y_pred))

# 6. Model सेव्ह करा
joblib.dump(model, "car_model.pkl")
print("✅ Model Saved as car_model.pkl")
