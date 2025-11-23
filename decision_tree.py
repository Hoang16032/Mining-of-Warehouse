import matplotlib 
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
MIN_LEAF_SIZE = 10
K_FOLDS = 5

# 1. Tải data
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Đã tải file '{INPUT_FILE}' thành công.")
    
    if TARGET_COLUMN not in df.columns:
        if 'y_HighValueChurn' in df.columns:
            TARGET_COLUMN = 'y_HighValueChurn'
        else:
            print(f"Lỗi: Không tìm thấy cột '{TARGET_COLUMN}'"); exit()
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'"); exit()

X = df[FEATURE_COLUMNS].values
y = df[TARGET_COLUMN].values

# 2. Tìm max depth tối ưu
print("\n" + "-"*40)
print("Giai đoạn 1: Tìm max depth tối ưu")
print("-" * 40)
X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
depths = range(3, 16)
tune_results = {'f1': [], 'prec': [], 'rec': [], 'acc': []}
print("-> Đang chạy thử nghiệm các độ sâu...")
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, class_weight='balanced', min_samples_leaf=MIN_LEAF_SIZE, random_state=42)
    dt.fit(X_tune_train, y_tune_train)
    val_pred = dt.predict(X_tune_val)
    tune_results['f1'].append(f1_score(y_tune_val, val_pred, pos_label=1))
    tune_results['prec'].append(precision_score(y_tune_val, val_pred, pos_label=1))
    tune_results['rec'].append(recall_score(y_tune_val, val_pred, pos_label=1))
    tune_results['acc'].append(accuracy_score(y_tune_val, val_pred))
best_idx = np.argmax(tune_results['f1'])
BEST_DEPTH = depths[best_idx]
print(f"Đã tìm thấy độ sâu tối ưu: MAX_DEPTH = {BEST_DEPTH}")

# --- Vẽ chart ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Biểu đồ 1: F1, Precision, Recall
ax1.plot(depths, tune_results['f1'], 'r-o', label='F1-Score', linewidth=2)
ax1.plot(depths, tune_results['prec'], 'b--o', label='Precision')
ax1.plot(depths, tune_results['rec'], 'g--o', label='Recall')

ax1.axvline(x=BEST_DEPTH, color='grey', linestyle='--', alpha=0.5)
ax1.text(BEST_DEPTH, tune_results['f1'][best_idx], f' Best: {BEST_DEPTH}', va='bottom', fontweight='bold')

ax1.set_title('Hiệu suất Decision Tree theo đô sâu của cây', fontsize=14)
ax1.set_ylabel('Điểm số', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Biểu đồ 2: Accuracy
ax2.plot(depths, tune_results['acc'], 'k-o', label='Accuracy')
ax2.set_title('Độ chính xác tổng thể (Accuracy)', fontsize=14)
ax2.set_xlabel('Max Depth', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dt_tuning_chart.png")
print("-> Đã lưu biểu đồ tuning đầy đủ: dt_tuning_chart.png")

# 3. Chạy K-Fold với độ sâu tối ưu
print("\n" + "-"*40)
print(f"Giai đoạn 2: Chạy K-fold (K={K_FOLDS}) VỚI DEPTH={BEST_DEPTH}")
print("-" * 40)
kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
acc_list, prec_list, rec_list, f1_list, cm_list = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
    X_train_k, X_test_k = X[train_idx], X[test_idx]
    y_train_k, y_test_k = y[train_idx], y[test_idx]
    
    model = DecisionTreeClassifier(
        max_depth=BEST_DEPTH,
        min_samples_leaf=MIN_LEAF_SIZE, 
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_k, y_train_k)
    y_pred_k = model.predict(X_test_k)
    
    acc_list.append(accuracy_score(y_test_k, y_pred_k))
    prec_list.append(precision_score(y_test_k, y_pred_k, pos_label=1))
    rec_list.append(recall_score(y_test_k, y_pred_k, pos_label=1))
    f1_list.append(f1_score(y_test_k, y_pred_k, pos_label=1))
    cm_list.append(confusion_matrix(y_test_k, y_pred_k))  
    print(f"   -> Fold {fold}/{K_FOLDS}: Acc={acc_list[-1]:.2%} | Prec={prec_list[-1]:.2%} | Recall={rec_list[-1]:.2%} | F1={f1_list[-1]:.2%}")

# 4. Kết quả trung bình
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

# Ma trận nhầm lẫn trung bình
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
plt.title(f'Ma trận nhầm lẫn (Decision tree) | Accuracy: {avg_acc:.2%}', fontsize=14)
plt.xlabel('Dự đoán', fontsize=12); plt.ylabel('Thực tế', fontsize=12)
plt.tight_layout()
plt.savefig("dt_matrix.png")
print(f"Đã lưu Matrix: dt_matrix.png")

# Cấu trúc cây
final_model_viz = DecisionTreeClassifier(max_depth=BEST_DEPTH, class_weight='balanced', min_samples_leaf=MIN_LEAF_SIZE, random_state=42)
final_model_viz.fit(X, y)

plt.figure(figsize=(25, 10))
plot_tree(final_model_viz, feature_names=FEATURE_COLUMNS, class_names=axis_labels,
          filled=True, rounded=True, max_depth=3, fontsize=10)
plt.title(f'Cấu trúc Cây Quyết Định (Depth={BEST_DEPTH})')
plt.savefig("dt_structure.png")
print(f"Đã lưu Cây quyết định: dt_structure.png")

# Feature Importance
importances = final_model_viz.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Các yếu tố ảnh hưởng (Decision Tree)")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='skyblue')
plt.xticks(range(X.shape[1]), [FEATURE_COLUMNS[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("dt_feature_importance.png")
print(f" Đã lưu Feature Importance: dt_feature_importance.png")
