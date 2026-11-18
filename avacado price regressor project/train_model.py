import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# Load dataset
data = pd.read_csv(r"S:\nareshcode\machinelearning\avacado price regressor project\avocado.csv")

# Preprocessing
# Drop unnecessary columns identified in EDA
data = data.drop(['Unnamed: 0', 'Date'], axis=1)

# Encoding categorical variables
data['region'] = data['region'].astype('category').cat.codes
data['type'] = data['type'].astype('category').cat.codes

# Define Features and Target[cite: 1, 3]
X = data.drop(['AveragePrice'], axis=1)
y = data['AveragePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define models[cite: 2]
models = {
    'LinearRegression': LinearRegression(),
    'RobustRegression': HuberRegressor(max_iter=1000),
    'RidgeRegression': Ridge(),
    'LassoRegression': Lasso(),
    'ElasticNet': ElasticNet(),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=2)), # Reduced degree to 2 to prevent memory issues in VS Code
        ('linear', LinearRegression())
    ]),
    'SGDRegressor': SGDRegressor(max_iter=1000),
    'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForest': RandomForestRegressor(),
    'SVM': SVR(),
    'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

# Create a directory for models if it doesn't exist
if not os.path.exists('backend'):
    os.makedirs('backend')

# Train and evaluate models
results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })
    
    # Saving models in the backend folder for API use
    with open(f'backend/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("\nModels have been trained and saved as pickle files in the 'backend' folder.")
print("Evaluation results have been saved to model_evaluation_results.csv.")


