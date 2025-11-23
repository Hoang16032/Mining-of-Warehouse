import pandas as pd
import datetime as dt
INPUT_FILE = 'sales_bill.csv'
OUTPUT_FILE = 'sales_mart.csv'

print("--- ĐANG TẠO SALES MART (SNAPSHOT & RFM) ---")

# Đọc dữ liệu
df = pd.read_csv(INPUT_FILE)
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['invoice_date'])
if 'totalamount' not in df.columns:
    df['totalamount'] = df['quantity'] * df['price']
df = df.sort_values(by=['customer_id', 'invoice_date'], ascending=[True, True])

snapshot_date = df['invoice_date'].max() + dt.timedelta(days=1)

# Tính toán RFM 
rfm = df.groupby('customer_id').agg({
    'invoice_date': lambda x: (snapshot_date - x.max()).days, 
    'invoice_no': 'count',                                   
    'totalamount': 'sum'                                      
}).rename(columns={
    'invoice_date': 'Recency',
    'invoice_no': 'Frequency',
    'totalamount': 'Monetary'
})
last_transaction = df.groupby('customer_id').tail(1).set_index('customer_id')

# Đổi tên các cột giao dịch thành last_
rename_map = {
    'invoice_date': 'last_invoice_date',
    'invoice_no': 'last_invoice_no',
    'totalamount': 'last_totalamount',
    'quantity': 'last_quantity',
    'price': 'last_price',
    'day': 'last_day',
    'month': 'last_month',
    'year': 'last_year',
    'quarter': 'last_quarter',
    'category': 'last_category'
}
last_transaction = last_transaction.rename(columns=rename_map)
final_df = last_transaction.join(rfm)

#  Sắp xếm cột 
original_cols = pd.read_csv(INPUT_FILE, nrows=1).columns.tolist()
original_cols.remove('customer_id')
ordered_cols = []
for col in original_cols:
    if col in rename_map: 
        ordered_cols.append(rename_map[col])
    elif col in final_df.columns: 
        ordered_cols.append(col)
final_order = ordered_cols + ['Recency', 'Frequency', 'Monetary']
final_df = final_df.reset_index()
final_df = final_df[['customer_id'] + final_order]
final_df['last_invoice_date'] = final_df['last_invoice_date'].dt.strftime('%d/%m/%Y')

# Lưu file
final_df.to_csv(OUTPUT_FILE, index=False)
print("1. Cột đầu tiên: customer_id")
print("2. Các cột giữa: Thông tin giao dịch cuối (last_totalamount, last_invoice_date...)")
print("3. Các cột cuối: Recency, Frequency, Monetary")
print("-" * 30)
print(final_df.head())