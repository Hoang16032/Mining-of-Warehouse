import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

print("--- Bắt đầu Grid Search cho Random Forest ---")

# Lưới tham số RF
param_grid = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [5, 10, 20, None],     
    'min_samples_leaf': [1, 2, 4],   
    'min_samples_split': [2, 5, 10], 
    'max_features': ['sqrt', 0.5], 
    'class_weight': ['balanced'] 
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\n" + "="*40)
print("Kết quả tối ưu cho Random forest")
print("="*40)
print(f"Best Params: {grid_search.best_params_}")
print("-" * 40)