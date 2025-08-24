import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load and explore data
print("Loading data...")
house = pd.read_csv("House Price India.csv")
print(f"Dataset shape: {house.shape}")
print("\nDataset info:")
house.info()

print("\nFirst few rows:")
print(house.head())

print("\nDataset description:")
print(house.describe())

# Check for missing values and duplicates
print(f"\nNull values: {house.isnull().sum().sum()}")
print(f"Duplicate rows: {house.duplicated().sum()}")

# Data cleaning
print("\nCleaning data...")
house_clean = house.copy()
house_clean.dropna(inplace=True)
house_clean.drop_duplicates(inplace=True)
print(f"Cleaned dataset shape: {house_clean.shape}")

# Data visualization
print("\nCreating visualization...")
plt.figure(figsize=(10, 6))
condition_prices = house_clean.groupby("condition of the house")["Price"].mean().sort_values(ascending=True)
condition_prices.plot(kind='bar')
plt.title("Condition of the house vs prices")
plt.ylabel("Mean Prices")
plt.xlabel("Condition of the house")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('condition_vs_price.png')
plt.show()

# Handle categorical variables if needed
le = LabelEncoder()
if house_clean["condition of the house"].dtype == 'object':
    house_clean["condition of the house"] = le.fit_transform(house_clean["condition of the house"])
    # Save the label encoder for later use
    joblib.dump(le, 'label_encoder.pkl')

# Prepare features and target
print("\nPreparing features and target...")
feature_columns = ["number of bedrooms", "number of bathrooms", "living area", 
                  "condition of the house", 'Number of schools nearby']
X = house_clean[feature_columns]
y = house_clean["Price"]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("\nFeature columns:")
for i, col in enumerate(X.columns):
    print(f"{i+1}. {col}")

# Split the data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Model 1: Decision Tree with GridSearch
print("\n=== Training Decision Tree Model ===")
param_grid_dt = {
    "criterion": ["squared_error", "friedman_mse", "absolute_error"],  # Updated parameter names
    "splitter": ["best", "random"],
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

tree_model = DecisionTreeRegressor(random_state=42)
grid_tree = GridSearchCV(estimator=tree_model, param_grid=param_grid_dt, 
                        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

print("Fitting Decision Tree...")
grid_tree.fit(X_train, y_train)
print("Best parameters:", grid_tree.best_params_)

tree_pred = grid_tree.predict(X_test)
tree_mae = mean_absolute_error(y_test, tree_pred)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_pred))
tree_r2 = r2_score(y_test, tree_pred)

print(f"Decision Tree MAE: {tree_mae:.2f}")
print(f"Decision Tree RMSE: {tree_rmse:.2f}")
print(f"Decision Tree R²: {tree_r2:.4f}")

# Model 2: Linear Regression
print("\n=== Training Linear Regression Model ===")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print(f"Linear Regression MAE: {lr_mae:.2f}")
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Linear Regression R²: {lr_r2:.4f}")

# Model 3: Random Forest
print("\n=== Training Random Forest Model ===")
param_grid_rf = {
    "max_depth": [5, 10, 15, None],
    "n_estimators": [50, 100, 200],
    "min_samples_split": [2, 5, 10]
}

rf_model = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, 
                      scoring='neg_mean_absolute_error', n_jobs=-1)

print("Fitting Random Forest...")
grid_rf.fit(X_train, y_train)
print("Best parameters:", grid_rf.best_params_)

rf_pred = grid_rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest MAE: {rf_mae:.2f}")
print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest R²: {rf_r2:.4f}")

# Compare models and select the best one
print("\n=== Model Comparison ===")
models_performance = {
    'Decision Tree': {'MAE': tree_mae, 'RMSE': tree_rmse, 'R²': tree_r2, 'model': grid_tree},
    'Linear Regression': {'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2, 'model': lr},
    'Random Forest': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2, 'model': grid_rf}
}

for name, metrics in models_performance.items():
    print(f"{name}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.4f}")

# Select best model based on lowest MAE
best_model_name = min(models_performance.keys(), 
                     key=lambda x: models_performance[x]['MAE'])
best_model = models_performance[best_model_name]['model']

print(f"\nBest model: {best_model_name}")
print(f"Best MAE: {models_performance[best_model_name]['MAE']:.2f}")

# Save the best model
print(f"\nSaving {best_model_name} model...")
joblib.dump(best_model, 'model.pkl')

# Save feature names for reference
feature_info = {
    'feature_names': list(X.columns),
    'feature_order': list(X.columns)
}
joblib.dump(feature_info, 'feature_info.pkl')

print("Model saved successfully!")

# Create a simple prediction function for testing
def predict_house_price(bedrooms, bathrooms, living_area, condition, schools_nearby):
    """Test prediction function"""
    features = np.array([[bedrooms, bathrooms, living_area, condition, schools_nearby]])
    prediction = best_model.predict(features)[0]
    return prediction

# Test the saved model
print("\n=== Testing Saved Model ===")
test_prediction = predict_house_price(3, 2, 1500, 3, 2)
print(f"Test prediction for (3 bed, 2 bath, 1500 sqft, condition 3, 2 schools): ₹{test_prediction:.2f}")

print("\nModel training complete!")
print("Files created:")
print("- model.pkl (trained model)")
print("- feature_info.pkl (feature information)")
if house["condition of the house"].dtype == 'object':
    print("- label_encoder.pkl (categorical encoder)")
print("- condition_vs_price.png (visualization)")