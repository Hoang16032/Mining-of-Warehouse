import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = 'sales_bill.csv'
OUTPUT_FILE = 'rfm_training_data_mall.csv' 
SNAPSHOT_DATE = pd.Timestamp('2022-12-08') 
print(f"--- Bắt đầu tạo dữ liệu ---")
print(f"Mốc thời gian cắt (Snapshot): {SNAPSHOT_DATE.date()}")

# 1. Tiền xử lý
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"-> Đã tải {len(df)} dòng dữ liệu.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {INPUT_FILE}")
    exit()
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['invoice_date'])
if 'totalamount' in df.columns:
    df['Total'] = df['totalamount']
else:
    df['Total'] = df['quantity'] * df['price']

# 2. Phân tách dữ liệu Quá khứ & Tương lai 
past_df = df[df['invoice_date'] < SNAPSHOT_DATE].copy()
future_df = df[df['invoice_date'] >= SNAPSHOT_DATE].copy()
print(f"-> Giao dịch Quá khứ (Toàn bộ): {len(past_df)} dòng")
print(f"-> Giao dịch Tương lai (Labeling): {len(future_df)} dòng")

# 3. Tính toán Features (X)
print("\n--- Đang tính toán Features (X)... ---")
def safe_mode(x):
    m = x.mode()
    if not m.empty:
        return m.iloc[0]
    return "Unknown"

features_df = past_df.groupby('customer_id').agg({
    'invoice_date': lambda x: (SNAPSHOT_DATE - x.max()).days,
    'invoice_no': 'count',
    'Total': 'sum',
    'age': 'first',
    'gender': 'first',
    'shopping_mall': 'first',
    'payment_method': 'first',
    'category': safe_mode
}).reset_index()
features_df.rename(columns={'invoice_date': 'Recency', 'invoice_no': 'Frequency', 'Total': 'Monetary'}, inplace=True)

# 4. Tạo nhãn Churn (y)
print("--- Đang lọc khách hàng Active và tạo nhãn... ---")
initial_count = len(features_df)
features_df = features_df[features_df['Frequency'] >= 1].copy()
print(f"-> Tổng khách hàng trong lịch sử: {initial_count}")
print(f"-> Đã loại bỏ khách mua 1 lần.")
print(f"-> Số lượng khách Active (F>=2) dùng để train: {len(features_df)}")
returned_customers = set(future_df['customer_id'].unique())

def define_churn_label(row):
    if row['customer_id'] not in returned_customers:
        return 1 
    else:
        return 0 

features_df['is_churn'] = features_df.apply(define_churn_label, axis=1)

# 5. Mã hóa dữ liệu
print("--- Đang mã hóa dữ liệu... ---")
le = LabelEncoder()
cols_to_encode = ['gender', 'shopping_mall', 'payment_method', 'category']

for col in cols_to_encode:
    features_df[col] = features_df[col].fillna('Unknown').astype(str)
    features_df[f'{col}_code'] = le.fit_transform(features_df[col])

# 6. Lưu file
features_df.to_csv(OUTPUT_FILE, index=False)
print(f"Tổng số khách hàng huấn luyện: {len(features_df)}")
print("Thống kê nhãn Churn (1: Rời bỏ, 0: Ở lại):")
print(features_df['is_churn'].value_counts())