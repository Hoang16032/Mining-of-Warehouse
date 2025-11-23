import pandas as pd
import numpy as np

INPUT_FILE = 'sales_origin.csv'
OUTPUT_FILE = 'sales_bill.csv' 
TARGET_CUSTOMER_COUNT = 15245 
print(f"--- Đang xử lý dữ liệu ---")

# 1. Đọc dữ liệu
df = pd.read_csv(INPUT_FILE)
df['invoice_date'] = pd.to_datetime(
    df['invoice_date'],
    dayfirst=True,
    errors='coerce'
)
df['invoice_date'] = df['invoice_date'].dt.strftime('%d/%m/%Y')

# 2. Tạo danh sách khách hàng gốc (Master List)
unique_profiles = df[['customer_id', 'gender', 'age']].drop_duplicates()
master_customers = unique_profiles.sample(n=TARGET_CUSTOMER_COUNT, random_state=42).reset_index(drop=True)
mall_counts = df['shopping_mall'].value_counts(normalize=True)
payment_counts = df['payment_method'].value_counts(normalize=True)
master_customers['assigned_mall'] = np.random.choice(
    mall_counts.index, 
    size=len(master_customers), 
    p=mall_counts.values
)
master_customers['assigned_payment'] = np.random.choice(
    payment_counts.index, 
    size=len(master_customers), 
    p=payment_counts.values
)
print(f"Đã tạo {len(master_customers)} hồ sơ khách hàng với Mall và Payment cố định.")

# 3. Tạo trọng số VIP 
weights = np.random.random(len(master_customers))
weights = weights ** 4 
weights /= weights.sum()

# 4. Gán đơn hàng 
df['original_index'] = df.index 
final_indices_map = []
malls = df['shopping_mall'].unique()

for mall in malls:
    mall_orders_indices = df[df['shopping_mall'] == mall].index.tolist()
    eligible_customers = master_customers[master_customers['assigned_mall'] == mall]
    if len(eligible_customers) == 0:
        continue 
    local_weights = weights[eligible_customers.index]
    if local_weights.sum() == 0: 
        local_weights = np.ones(len(eligible_customers))
    local_weights /= local_weights.sum()
    
    # Chọn ngẫu nhiên khách hàng cho các đơn hàng
    chosen_customer_indices = np.random.choice(
        eligible_customers.index, 
        size=len(mall_orders_indices), 
        p=local_weights
    )
    for order_idx, customer_master_idx in zip(mall_orders_indices, chosen_customer_indices):
        final_indices_map.append([order_idx, customer_master_idx])

# 5. Cập nhật DataFrame
map_df = pd.DataFrame(final_indices_map, columns=['order_idx', 'customer_master_idx'])
map_df = map_df.set_index('order_idx').sort_index()
assigned_rows = master_customers.iloc[map_df['customer_master_idx']].reset_index(drop=True)
df['customer_id'] = assigned_rows['customer_id'].values
df['gender'] = assigned_rows['gender'].values
df['age'] = assigned_rows['age'].values
df['payment_method'] = assigned_rows['assigned_payment'].values

check_mall = df.groupby('customer_id')['shopping_mall'].nunique()
check_pay = df.groupby('customer_id')['payment_method'].nunique()

# Lưu file
df.drop(columns=['original_index'], inplace=True, errors='ignore')
df.to_csv(OUTPUT_FILE, index=False)
