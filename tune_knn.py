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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
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

# Thử K từ 1 đến 29 
K_RANGE = range(1, 50, 2) 
print("\n" + "="*50)
print(f"--- Bắt đầu chuẩn đoán mô hình Knn ---")
print("="*50 + "\n")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Đã tải file '{INPUT_FILE}' thành công ({len(df)} dòng).")  
    # Kiểm tra cột Target
    if TARGET_COLUMN not in df.columns:
        if 'y_HighValueChurn' in df.columns:
            TARGET_COLUMN = 'y_HighValueChurn' 
        else:
            print(f"Lỗi: Không tìm thấy cột '{TARGET_COLUMN}'"); exit()
            
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'"); exit()

X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y        
)
print("\n--- Đã chia dữ liệu thành Train (80%) và Test (20%) ---")

# Chuẩn hóa 
print("\n--- Đang chuẩn hóa (StandardScaler) dữ liệu... ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 
print("-> Chuẩn hóa hoàn tất.")

# Biểu đồ
def plot_metrics_knn(k_values, f1, prec, rec, acc, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Biểu đồ 1: P, R, F1
    ax1.plot(k_values, f1, 'r-o', label='F1-Score')
    ax1.plot(k_values, prec, 'b--o', label='Precision')
    ax1.plot(k_values, rec, 'g--o', label='Recall')
    
    # Tìm Best K theo F1
    best_idx = np.argmax(f1)
    best_k = k_values[best_idx]
    ax1.axvline(x=best_k, color='grey', linestyle='--', alpha=0.5)
    ax1.text(best_k, f1[best_idx], f' Best K: {best_k}', va='bottom', fontweight='bold')

    ax1.set_title(f'Hiệu suất {model_name} theo số lượng hàng xóm (K)', fontsize=14)
    ax1.set_ylabel('Điểm số', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Biểu đồ 2: Accuracy
    ax2.plot(k_values, acc, 'k-o', label='Accuracy')
    ax2.set_title('Độ chính xác tổng thể (Accuracy)', fontsize=14)
    ax2.set_xlabel('K (n_neighbors)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f"optimize_{model_name.lower()}.png"
    plt.savefig(output_file)
    plt.close() 
    print(f"   -> Đã lưu biểu đồ: {output_file} (Best K={best_k})")

# Chạy vòng lặp tìm k 
print(f"\n--- Đang quét các giá trị K... ---")
metrics_knn = {'f1': [], 'prec': [], 'rec': [], 'acc': []}

for k in K_RANGE:
    model_temp = KNeighborsClassifier(
        n_neighbors=k,
        n_jobs=-1, 
        weights='distance' 
    )
    model_temp.fit(X_train_scaled, y_train)
    y_pred_test = model_temp.predict(X_test_scaled)
    
    metrics_knn['f1'].append(f1_score(y_test, y_pred_test, pos_label=1))
    metrics_knn['prec'].append(precision_score(y_test, y_pred_test, pos_label=1))
    metrics_knn['rec'].append(recall_score(y_test, y_pred_test, pos_label=1))
    metrics_knn['acc'].append(accuracy_score(y_test, y_pred_test))
plot_metrics_knn(list(K_RANGE), metrics_knn['f1'], metrics_knn['prec'], metrics_knn['rec'], metrics_knn['acc'], "KNN")

