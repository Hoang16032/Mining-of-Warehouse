# PROJECT: DỰ BÁO KHÁCH HÀNG RỜI BỎ (RETAIL ACTIVE CHURN PREDICTION)

Dự án áp dụng các kỹ thuật Khai phá dữ liệu (Data Mining) để phân tích hành vi mua sắm và dự báo nguy cơ rời bỏ của nhóm khách hàng "Active" (Khách hàng thường xuyên) trong lĩnh vực bán lẻ.

## 📋 Tổng quan về Phương pháp

* **Bài toán:** Classification (Phân loại Nhị phân: Rời bỏ vs Ở lại).
* **Đối tượng dự báo:** Khách hàng "Active" (Có tần suất mua hàng `Frequency >= 2` trong 12 tháng qua).
* **Kỹ thuật xử lý dữ liệu:** Rolling Window (Cửa sổ trượt 12 tháng) để tính toán chỉ số hành vi (RFM) thay vì tích lũy toàn bộ lịch sử.
* **Đánh giá mô hình:** Sử dụng K-Fold Cross Validation (5-Folds) để đảm bảo độ tin cậy.

---

## ⚙️ Yêu cầu cài đặt

Trước khi chạy chương trình, vui lòng cài đặt các thư viện Python cần thiết:

    pip install pandas numpy matplotlib seaborn scikit-learn xgboost squarify openpyxl

---

## 📂 Cấu trúc Source Code

Mã nguồn được tổ chức theo quy trình chuẩn của Data Mining, từ xử lý dữ liệu đến đánh giá mô hình.

### 1. Giai đoạn Tiền xử lý & Chuẩn bị dữ liệu (Data Preparation)
Các script này chịu trách nhiệm biến đổi dữ liệu giao dịch thô thành dữ liệu chất lượng cao để huấn luyện.

* **`create_mart.py`**
    * *Chức năng:* Chuyển đổi dữ liệu cấp độ hóa đơn (`sales_bill.csv`) sang dữ liệu cấp độ khách hàng cơ bản (`sales_mart.csv`).
    * *Xử lý:* Làm sạch dữ liệu, xử lý giá trị âm/null.
* **`create_training_data.py`**
    * *Chức năng:* Tạo bộ dữ liệu huấn luyện cuối cùng (`rfm_training_data_mall.csv`) dùng cho các mô hình phân loại.
    * *Logic áp dụng:*
        * **Rolling Window:** Chỉ lấy dữ liệu hành vi trong 12 tháng trước ngày Snapshot (08/12/2022).
        * **Active Filter:** Loại bỏ khách hàng vãng lai (Frequency = 1).
        * **Labeling:** Gán nhãn rời bỏ dựa trên hành vi trong 3 tháng cuối cùng (đến 08/03/2023).
* **`quantile_segment.py`**
    * *Chức năng:* Phân khúc khách hàng thành 11 nhóm hành vi (Champions, At Risk, Lost...) dựa trên điểm số RFM (dùng để báo cáo Insight).

### 2. Giai đoạn Tinh chỉnh tham số (Hyperparameter Tuning)
Sử dụng các kỹ thuật tìm kiếm tham số tối ưu trước khi đưa vào huấn luyện chính thức.

* **`tune_random_forest.py`**: Sử dụng **Grid Search** để tìm tổ hợp tối ưu cho `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`.
* **`tune_XGboost.py`**: Sử dụng **Grid Search** để tìm `learning_rate`, `gamma`, `colsample_bytree`, `max_depth` tối ưu.
* **`tune_knn.py`**: Sử dụng vòng lặp kiểm thử để tìm số lượng hàng xóm **`K`** có F1-Score cao nhất (Elbow Method).
* **`diagnose_all_models.py`**: Chạy chẩn đoán nhanh độ sâu cây (max_depth) cho cả 3 mô hình Tree-based để vẽ biểu đồ xu hướng.

### 3. Giai đoạn Huấn luyện & Đánh giá (Modeling & Evaluation)
Chạy các mô hình với tham số đã được tối ưu, sử dụng kỹ thuật **K-Fold Cross Validation (5-Folds)**. Mỗi file sẽ xuất ra Ma trận nhầm lẫn và Báo cáo phân loại chi tiết.

* **`decision_tree.py`**: Chạy mô hình Cây quyết định (Decision Tree).
* **`random_forest.py`**: Chạy mô hình Random Forest (Ensemble Bagging).
* **`XGboost.py`**: Chạy mô hình XGBoost (Ensemble Boosting) - *Mô hình chiến thắng*.
* **`knn.py`**: Chạy mô hình K-Nearest Neighbors (Distance-based).

### 4. Tổng hợp & Trực quan hóa (Visualization)
* **`model_comparision.py`**:
    * Tổng hợp kết quả từ 4 mô hình trên.
    * Vẽ biểu đồ so sánh trực quan các chỉ số: Accuracy, Precision, Recall, F1-Score.
* **`visualize_insight.py`**:
    * Vẽ biểu đồ **Feature Importance** (Mức độ quan trọng của biến).
    * Tạo 2 phiên bản: Toàn cảnh (thấy rõ sự áp đảo của Frequency) và Zoom-in (thấy rõ các yếu tố tiềm ẩn khác).

---

## 🚀 Hướng dẫn chạy chương trình (Step-by-Step)

Vui lòng chạy theo thứ tự sau để đảm bảo luồng dữ liệu chính xác:

**Bước 1: Tạo dữ liệu**
    python create_mart.py
    python create_training_data.py

*(Kết quả: File `rfm_training_data_mall.csv` sẽ được tạo ra)*

**Bước 2: Chạy Tuning để tìm tham số mới**
    python tune_random_forest.py
    python tune_XGboost.py
    python tune_knn.py

**Bước 3: Chạy các mô hình phân loại (K-Fold)**
    python XGboost.py
    python random_forest.py
    python decision_tree.py
    python knn.py

**Bước 4: Vẽ biểu đồ so sánh và Insight**
    python model_comparision.py
    python visualize_insight.py

---

## 📊 Kết quả đầu ra (Artifacts)

Sau khi chạy xong, chương trình sẽ sinh ra các file hình ảnh báo cáo (`.png`) trong thư mục hiện tại:

1.  **Ma trận nhầm lẫn (Confusion Matrix) - K-Fold Average:**
    * `xgb_kfold_matrix_final.png`
    * `rf_kfold_matrix_final.png`
    * `dt_kfold_matrix_final.png`
    * `knn_kfold_matrix_final.png`

2.  **Phân tích nhân tố ảnh hưởng (Feature Importance):**
    * `xgb_feature_importance_final.png`
    * `rf_feature_importance_final.png`
    * `insight_xgboost.png` (Bao gồm cả bản Zoom-in và Toàn cảnh)

3.  **Biểu đồ Tối ưu hóa tham số:**
    * `xgb_tuning_chart.png`
    * `rf_tuning_chart.png`
    * `knn_tuning_chart.png`
    * `dt_tuning_chart.png`

4.  **Tổng hợp:**
    * `model_comparison.png` (So sánh hiệu năng 4 mô hình).
