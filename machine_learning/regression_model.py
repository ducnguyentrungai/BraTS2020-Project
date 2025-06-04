import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


survival_df = pd.read_csv('/work/cuc.buithi/brats_challenge/test/survival_info_filled.csv')

X = survival_df[['Age','tumor_volume', 'ncr_net_volume', 'ed_volume', 'et_volume', 'brain_volume', 'tumor_pct', 'ncr_net_pct', 'ed_pct', 'et_pct', 'Extent_of_Resection_Encoder']]
y = survival_df['Survival_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print(X_train_scaled[:5])

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print(f"--- {name} ---")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R² Score:", r2_score(y_test, preds))
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, preds)))
    print()


param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0]
}

xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=50,              # Try 50 combinations
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_xgb = random_search.best_estimator_
print("Best Params:", random_search.best_params_)

# Predict & evaluate
y_pred = best_xgb.predict(X_test)

# Predictions
y_pred = best_xgb.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test.astype(float) - y_pred) / y_test.astype(float))) * 100

print(f"MAE   : {mae:.2f}")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.3f}")
print(f"MAPE  : {mape:.2f}%")   