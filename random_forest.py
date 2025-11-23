import matplotlib 
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
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

print("\n" + "="*60)
print(f"Depth={10} | Estimators={100} | Min Leaf={10}")
print("="*60 + "\n")

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
print(f"Bắt đầu chạy k-fold (K={K_FOLDS})")
print("-" * 40)
kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
acc_list, prec_list, rec_list, f1_list, cm_list = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
    X_train_k, X_test_k = X[train_idx], X[test_idx]
    y_train_k, y_test_k = y[train_idx], y[test_idx]

    # Random Forest với tham số tối ưu
    model = RandomForestClassifier(
        n_estimators=100,          
        max_depth=10,              
        min_samples_leaf=2,        
        min_samples_split=10,      
        max_features='sqrt',       
        class_weight='balanced',  
        n_jobs=-1,                 
        random_state=42           
    )
    
    model.fit(X_train_k, y_train_k)
    y_pred_k = model.predict(X_test_k)
    
    acc_list.append(accuracy_score(y_test_k, y_pred_k))
    prec_list.append(precision_score(y_test_k, y_pred_k, pos_label=1))
    rec_list.append(recall_score(y_test_k, y_pred_k, pos_label=1))
    f1_list.append(f1_score(y_test_k, y_pred_k, pos_label=1))
    cm_list.append(confusion_matrix(y_test_k, y_pred_k))
    
    print(f"   -> Fold {fold}/{K_FOLDS}: Accuracy={acc_list[-1]:.2%} | Precision={prec_list[-1]:.2%} | Recall={rec_list[-1]:.2%} | F1={f1_list[-1]:.2%}")

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
plt.title(f'Ma trận nhầm lẫn (Random Forest) | Accuracy: {avg_acc:.2%}', fontsize=14)
plt.xlabel('Dự đoán', fontsize=12); plt.ylabel('Thực tế', fontsize=12)
plt.tight_layout()
plt.savefig("rf_matrix.png")
print(f"\nĐã lưu Matrix: rf_matrix.png")

# Feature Importance 
final_model_viz = RandomForestClassifier(
    n_estimators=100,          
    max_depth=10,              
    min_samples_leaf=2,        
    min_samples_split=10,      
    max_features='sqrt',       
    class_weight='balanced',  
    n_jobs=-1,                 
    random_state=42    
)
final_model_viz.fit(X, y)

importances = final_model_viz.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Các yếu tố ảnh hưởng (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='green')
plt.xticks(range(X.shape[1]), [FEATURE_COLUMNS[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
print(f"Đã lưu Feature Importance: rf_feature_importance.png")

