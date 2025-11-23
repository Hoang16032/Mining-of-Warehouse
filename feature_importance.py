import matplotlib 
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

INPUT_FILE = "rfm_training_data_mall.csv"
TARGET_COLUMN = 'is_churn'

FEATURE_COLUMNS = [
    'Recency', 'Frequency', 'Monetary',            
    'age', 'gender_code',                          
    'shopping_mall_code', 'category_code', 'payment_method_code' 
]

print("\n" + "="*60)
print("--- CHƯƠNG TRÌNH PHÂN TÍCH INSIGHT & TRỰC QUAN HÓA ---")
print("--- (Tạo biểu đồ Feature Importance: Toàn cảnh & Zoom-in) ---")
print("="*60 + "\n")

# Load data
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Đã tải dữ liệu: {len(df)} dòng.")
except:
    print(f"Lỗi không tìm thấy file '{INPUT_FILE}'"); exit()

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Tính scale_pos_weight 
count_neg = (y == 0).sum()
count_pos = (y == 1).sum()
spw = count_neg / count_pos

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, class_weight='balanced', random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight='balanced', n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, scale_pos_weight=spw, 
        learning_rate=0.05, n_jobs=-1, eval_metric='logloss', random_state=42
    )
}
colors = {'Decision Tree': 'skyblue', 'Random Forest': 'mediumseagreen', 'XGBoost': 'orange'}

# Vẽ biểu đồ
for name, model in models.items():
    print(f"-> Đang phân tích mô hình: {name}...")
    model.fit(X, y)
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.barplot(ax=axes[0], x='Importance', y='Feature', data=feat_df, color=colors[name])
    axes[0].set_title(f'{name}: Mức độ ảnh hưởng (Toàn cảnh)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Độ quan trọng')
    axes[0].set_ylabel('')
    for i, v in enumerate(feat_df['Importance']):
        axes[0].text(v + 0.01, i, f"{v:.2f}", va='center')

    # Loại frequency
    feat_df_zoom = feat_df[feat_df['Feature'] != 'Frequency'].copy() 
    sns.barplot(ax=axes[1], x='Importance', y='Feature', data=feat_df_zoom, color=colors[name], alpha=0.7)
    axes[1].set_title(f'{name}: Các yếu tố tiềm ẩn (Đã ẩn Frequency)', fontsize=14, fontweight='bold', color='darkred')
    axes[1].set_xlabel('Độ quan trọng (Scale nhỏ hơn)')
    axes[1].set_ylabel('')
    
    # Nhãn số liệu
    for i, v in enumerate(feat_df_zoom['Importance']):
        axes[1].text(v, i, f"{v:.3f}", va='center', fontweight='bold')
    plt.tight_layout()
    
    # Lưu file
    filename = f"insight_{name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300)
    print(f"   ✅ Đã lưu ảnh phân tích: {filename}")

