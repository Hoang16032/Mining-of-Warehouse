import pandas as pd

INPUT_FILE = 'sales_bill.csv'
OUTPUT_FILE = 'sales_bill_last_3_months.csv'

# Khoảng thời gian lấy
START_DATE = pd.Timestamp('2022-12-08')
END_DATE = pd.Timestamp('2023-03-08')
print(f"Khoảng thời gian: {START_DATE.date()} -> {END_DATE.date()}")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"-> Đã tải file gốc '{INPUT_FILE}': {len(df)} dòng.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {INPUT_FILE}")
    exit()

# Chuyển đổi cột ngày tháng
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True, errors='coerce')

filtered_df = df[
    (df['invoice_date'] >= START_DATE) & 
    (df['invoice_date'] <= END_DATE)
].copy()


filtered_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nĐã xong! File kết quả: '{OUTPUT_FILE}'")
print(f"-> Số lượng giao dịch trích xuất được: {len(filtered_df)} dòng")
print(f"-> Tỷ lệ so với file gốc: {(len(filtered_df)/len(df))*100:.2f}%")