import matplotlib 
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore')

INPUT_FILE = "rfm_training_data_mall.csv"
TARGET_COLUMN = 'is_churn'

FEATURE_COLUMNS = [
    'Recency', 'Frequency', 'Monetary',            
    'age', 'gender_code',                          
    'shopping_mall_code', 'category_code', 'payment_method_code' 
]
POSITIVE_CLASS_NAME = 'Rời bỏ (1)'
NEGATIVE_CLASS_NAME = 'Ở lại (0)'
K_FOLDS = 5          
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Đã tải file '{INPUT_FILE}' thành công. ({len(df)} dòng)")
    
    if TARGET_COLUMN not in df.columns:
        if 'y_HighValueChurn' in df.columns:
            TARGET_COLUMN = 'y_HighValueChurn'
        else:
            print(f"Lỗi: Không tìm thấy cột '{TARGET_COLUMN}'"); exit()
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'"); exit()

X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values

# K-fold split
print("\n" + "-"*40)
print(f"Bắt đầu chạy K-fold (K={K_FOLDS})")
print("-" * 40)

kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
acc_list, prec_list, rec_list, f1_list, cm_list = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
    X_train_k, X_test_k = X[train_idx], X[test_idx]
    y_train_k, y_test_k = y[train_idx], y[test_idx]
    spw_k = (y_train_k == 0).sum() / (y_train_k == 1).sum()
    
    # XGBoost với tham số tối ưu
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=6,
        gamma=5,                    
        colsample_bytree=0.7, 
        subsample=1.0,           
        scale_pos_weight=spw_k,          
        n_jobs=-1,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train_k, y_train_k)
    y_pred_k = model.predict(X_test_k)  
    acc_list.append(accuracy_score(y_test_k, y_pred_k))
    prec_list.append(precision_score(y_test_k, y_pred_k, pos_label=1))
    rec_list.append(recall_score(y_test_k, y_pred_k, pos_label=1))
    f1_list.append(f1_score(y_test_k, y_pred_k, pos_label=1))
    cm_list.append(confusion_matrix(y_test_k, y_pred_k))
    
    print(f"   -> Fold {fold}/{K_FOLDS}: Accuracy={acc_list[-1]:.2%} | Recall={rec_list[-1]:.2%} | F1={f1_list[-1]:.2%}")

print("\n" + "-"*40)
print("Kết quả trung bình")
print("-" * 40)

avg_acc = np.mean(acc_list)
avg_prec = np.mean(prec_list)
avg_rec = np.mean(rec_list)
avg_f1 = np.mean(f1_list)

print(f"1. Accuracy (Độ chính xác):  {avg_acc:.2%}")
print(f"2. Precision (Độ chuẩn xác): {avg_prec:.2%}")
print(f"3. Recall (Độ nhạy):         {avg_rec:.2%}")
print(f"4. F1-Score (Cân bằng):      {avg_f1:.2%}")

# Ma trận nhầm lẫn
mean_cm = np.mean(cm_list, axis=0)
rounded_cm = np.rint(mean_cm).astype(int)
axis_labels = [NEGATIVE_CLASS_NAME, POSITIVE_CLASS_NAME]

plt.figure(figsize=(8, 6))
sns.heatmap(
    rounded_cm, annot=True, fmt='d', 
    cmap='Blues', 
    xticklabels=axis_labels, yticklabels=axis_labels,
    annot_kws={"size": 14}
)
plt.title(f'Ma trận nhầm lẫn (XGboost) | Accuracy: {avg_acc:.2%}', fontsize=14)
plt.xlabel('Dự đoán', fontsize=12); plt.ylabel('Thực tế', fontsize=12)
plt.tight_layout()
plt.savefig("xgb_kfold_matrix_final.png")
print(f"\n Đã lưu Matrix: xgb_kfold_matrix_final.png")
print("\n--- Đang tính toán Feature Importance (Model tổng) ---")

count_neg = (y == 0).sum()
count_pos = (y == 1).sum()
spw_total = count_neg / count_pos

# Feature Importance 
final_model_viz = XGBClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=6,
    gamma=5,                     
    colsample_bytree=0.7, 
    subsample=1.0,             
    scale_pos_weight=spw_k,      
    n_jobs=-1,
    eval_metric='logloss',
    random_state=42
)

final_model_viz.fit(X, y)

importances = final_model_viz.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title(f"Yếu tố ảnh hưởng nhất (XGBoost Optimized - Gamma={5})")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='orange') # Màu cam
plt.xticks(range(X.shape[1]), [FEATURE_COLUMNS[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("xgb_feature_importance_final.png")
print(f"Đã lưu Feature Importance: xgb_feature_importance_final.png")

