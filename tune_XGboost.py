import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings

warnings.filterwarnings('ignore')
INPUT_FILE = "rfm_training_data_mall.csv"
TARGET_COLUMN = 'is_churn'
FEATURE_COLUMNS = [
    'Recency', 'Frequency', 'Monetary',            
    'age', 'gender_code',                          
    'shopping_mall_code', 'category_code', 'payment_method_code' 
]

df = pd.read_csv(INPUT_FILE)
X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tính scale_pos_weight
count_neg = (y_train == 0).sum()
count_pos = (y_train == 1).sum()
spw = count_neg / count_pos
print("--- BẮT ĐẦU GRID SEARCH CHO XGBOOST ---")


# 2. Thiết lập lưới tham số 
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 6],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'subsample': [0.7, 1.0],
    'gamma': [0, 1, 5],
    'scale_pos_weight': [spw]
}

# 3. Chạy Grid Search
xgb = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,             
    scoring='f1',     
    verbose=1,        
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 4. Kết quả
print("\n" + "="*40)
print("Kết quả tối ưu cho XGBoost")
print("="*40)
print(f"Best Params: {grid_search.best_params_}")
print(f"Best F1 Score (Train/Val): {grid_search.best_score_:.2%}")
print("-" * 40)
