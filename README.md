# ============================== CELL: Generate README.md ==============================

readme_text = """
# 🏦 Credit Default Prediction Challenge (JCDS 0612)

## 1. Yêu cầu cuộc thi
Cuộc thi Kaggle **Credit Default Prediction Challenge – JCDS 0612** nhằm xây dựng mô hình **dự đoán xác suất khách hàng tín dụng sẽ vỡ nợ trong tháng tới**.  

- **Input**: Hồ sơ khách hàng tín dụng (giới tính, hôn nhân, học vấn, độ tuổi, hạn mức tín dụng, lịch sử thanh toán, dư nợ, khoản thanh toán hàng tháng).  
- **Output**: Xác suất (`default_payment_next_month`) khách hàng vỡ nợ (default = 1, non-default = 0).  
- **Evaluation metric**: AUC (Area Under ROC Curve).  

---

## 2. Dataset
Nguồn dữ liệu: bộ tín dụng thực tế từ ngân hàng Đài Loan.  

### 📂 Files
- **train.csv** – tập huấn luyện gồm toàn bộ feature và cột target `default_payment_next_month`.  
- **test.csv** – tập kiểm thử (giống train nhưng không có target).  
- **sample_submission.csv** – file mẫu submission.  

### 🧾 Columns
- `ID`: Mã khách hàng  
- `LIMIT_BAL`: Hạn mức tín dụng (NT$)  
- `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`: Thông tin nhân khẩu học  
- `PAY_0 … PAY_6`: Lịch sử trả nợ (tình trạng tháng trước đến tháng -6)  
- `BILL_AMT1 … BILL_AMT6`: Dư nợ 6 tháng gần nhất  
- `PAY_AMT1 … PAY_AMT6`: Số tiền đã thanh toán 6 tháng gần nhất  
- `default_payment_next_month`: Target  

---

## 3. Feature Engineering (FE)

### 3.1 Các feature gốc
- Nhân khẩu học: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`  
- Hạn mức: `LIMIT_BAL`  
- Lịch sử trả nợ: `PAY_0 … PAY_6`  
- Hóa đơn: `BILL_AMT1 … BILL_AMT6`  
- Thanh toán: `PAY_AMT1 … PAY_AMT6`  

### 3.2 Các feature được tạo thêm
- **Utilization ratio**: `BILL_AMT / LIMIT_BAL`  
  → đo mức độ sử dụng hạn mức tín dụng.  
- **Payment ratio**: `PAY_AMT / BILL_AMT`  
  → đo mức độ thanh toán so với dư nợ.  
- **Trend features**: chênh lệch giữa các tháng `BILL_AMT(i) - BILL_AMT(i+1)`  
  → nợ tăng/giảm theo thời gian.  
- **Aggregate statistics**: tổng, trung bình, min, max, std cho nhóm `BILL_AMT` và `PAY_AMT`.  

### 3.3 Kiểm tra đa cộng tuyến
- Tính hệ số tương quan giữa các biến.  
- Với cặp feature có **correlation > 0.95**, loại bớt để tránh multicollinearity.  
- Kết quả: giữ lại bộ feature gọn, không trùng lặp, nhưng vẫn giữ được signal chính.

---

## 4. Models lựa chọn
Áp dụng **3 mô hình boosting chính** + ensemble:  

1. **LightGBM (LGBMClassifier)**  
   - Ưu điểm: nhanh, xử lý feature số + cat tốt.  
   - Tham số: `num_leaves=31`, `max_depth=-1`, `lambda_l2=10`, `feature_fraction=0.9`, `bagging_fraction=0.9`.  

2. **XGBoost (XGBClassifier)**  
   - Ưu điểm: ổn định, benchmark chuẩn trong credit risk.  
   - Tham số: `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5`.  

3. **CatBoost (CatBoostClassifier)**  
   - Ưu điểm: xử lý categorical trực tiếp.  
   - Tham số: `depth=6`, `learning_rate=0.05`, `l2_leaf_reg=5.0`.  

4. **Ensemble / Stacking**  
   - Rank average các model.  
   - Logistic Regression meta-model trên OOF predictions.  

---

## 5. Kết quả

### 5.1 Cross-validation (OOF AUC)
- LightGBM: **0.7886**  
- XGBoost: ~**0.789**  
- CatBoost: **0.7916**  
- Ensemble rank average: **0.793**  
- Stacking logistic meta: **0.795**  

### 5.2 Kaggle Leaderboard
- **Public Score**: 0.80217  
- **Private Score**: 0.76818  

📌 **Nhận xét**:  
- Public score cao nhưng Private score thấp hơn → dấu hiệu overfit public set.  
- CV ~0.79 sát với Private 0.768 → mô hình có generalization trung bình, cần cải thiện bằng regularization + feature engineering.  

---

## 6. Kết luận
- 3 mô hình boosting (LGB, XGB, CAT) đều cho AUC sát nhau ~0.79.  
- Ensemble giúp tăng nhẹ AUC và ổn định hơn.  
- Vấn đề chính: private score drop so với public → cần giảm overfitting bằng:
  - Regularization mạnh hơn  
  - Ensemble nhiều seeds  
  - Feature engineering dựa trên tín dụng thực tế (ratio, trend, interaction).  
"""

# Ghi ra file
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_text)

print("✅ README.md generated successfully!")
