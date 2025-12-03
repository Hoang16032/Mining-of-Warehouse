import matplotlib 
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
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

# Nạp data
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

# Tuning
print("\n" + "-"*40)
print("Tìm số lượng hàng xóm tối ưu")
print("-" * 40)
X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_tune_train = scaler.fit_transform(X_tune_train)
X_tune_val = scaler.transform(X_tune_val)
k_values = range(1, 50, 2) 
tune_scores = []
print("-> Đang chạy thử nghiệm các giá trị K...")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', n_jobs=-1)
    knn.fit(X_tune_train, y_tune_train)
    val_pred = knn.predict(X_tune_val)
    score = f1_score(y_tune_val, val_pred, pos_label=1)
    tune_scores.append(score)

best_idx = np.argmax(tune_scores)
BEST_K = k_values[best_idx]
print(f"Đã tìm thấy K tối ưu: K = {BEST_K}")

# Vẽ biểu đồ Tuning
plt.figure(figsize=(10, 5))
plt.plot(k_values, tune_scores, 'purple', marker='o', label='Validation F1-Score')
plt.axvline(x=BEST_K, color='grey', linestyle='--', label=f'Best K={BEST_K}')
plt.title('KNN Tuning: F1-Score vs K Neighbors')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)
plt.savefig("knn_tuning_chart.png")
print("-> Đã lưu biểu đồ tuning: knn_tuning_chart.png")

# K-fold
print("\n" + "-"*40)
print(f"Giai đoạn 2: Chạy k-fold (K={K_FOLDS}) với k-neighbors={BEST_K}")
print("-" * 40)

kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
acc_list, prec_list, rec_list, f1_list, cm_list = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
    X_train_k, X_test_k = X[train_idx], X[test_idx]
    y_train_k, y_test_k = y[train_idx], y[test_idx]
    scaler_k = StandardScaler()
    X_train_k = scaler_k.fit_transform(X_train_k)
    X_test_k = scaler_k.transform(X_test_k)
    
    model = KNeighborsClassifier(n_neighbors=BEST_K, weights='distance', n_jobs=-1)
    model.fit(X_train_k, y_train_k)
    y_pred_k = model.predict(X_test_k)
    
    acc_list.append(accuracy_score(y_test_k, y_pred_k))
    prec_list.append(precision_score(y_test_k, y_pred_k, pos_label=1))
    rec_list.append(recall_score(y_test_k, y_pred_k, pos_label=1))
    f1_list.append(f1_score(y_test_k, y_pred_k, pos_label=1))
    cm_list.append(confusion_matrix(y_test_k, y_pred_k))
    
    print(f"   -> Fold {fold}/{K_FOLDS}: Accuracy={acc_list[-1]:.2%} | Precision={prec_list[-1]:.2%} | Recall={rec_list[-1]:.2%} | F1={f1_list[-1]:.2%}")

# Kết quả
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

plt.figure(figsize=(8, 7))
ax = sns.heatmap(
    rounded_cm, annot=True, fmt='d', 
    cmap='Blues', 
    xticklabels=axis_labels, yticklabels=axis_labels,
    annot_kws={"size": 14},
    cbar=False  
)
plt.title('Ma trận nhầm lẫn cho KNN', fontsize=16, pad=20)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
stats_text = (
    f"Đánh giá cho KNN:\n"
    f"Accuracy: {avg_acc:.4f}\n"
    f"Precision: {avg_prec:.4f}\n"
    f"Recall: {avg_rec:.4f}\n"
    f"F1 Score: {avg_f1:.4f}"
)
plt.text(
    x=0, y=1.12, 
    s=stats_text, 
    transform=ax.transAxes, 
    fontsize=11, 
    ha='left', va='bottom', 
    fontfamily='monospace'
)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("knn_matrix.png")
print(f"Đã lưu Matrix: knn_matrix.png")
