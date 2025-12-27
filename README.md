# Dự án: Hệ thống phân tích văn bản & cảnh báo rủi ro tâm lý từ ngôn ngữ viết

## Giới thiệu dự án

Dự án xây dựng một hệ thống có khả năng phân tích văn bản người dùng
nhập vào (tiếng Việt hoặc tiếng Anh), đánh giá mức độ tiêu cực và nhận
diện nguy cơ liên quan đến ý định tự làm hại (self-harm).\
Hệ thống sử dụng Machine Learning, TF-IDF, rule-based safety layer, và
Streamlit để hiển thị kết quả.

## Cấu trúc thư mục

    mental_health_detection/
    │
    ├── app/
    │   └── app.py
    │
    ├── data/
    │   ├── raw/
    │   │   └── Suicide_Detection.csv
    │   └── processed/
    │       └── clean_data.csv
    │
    ├── models/
    │   ├── vectorizer.pkl
    │   ├── classifier.pkl
    │   └── label_map.json
    │
    ├── messages/
    │   ├── low.txt
    │   ├── medium.txt
    │   └── high.txt
    │
    ├── src/
    |   ├── _init_.py
    │   ├── preprocessing.py
    │   ├── train.py
    │   ├── predict.py
    │   ├── risk_assessment.py
    │   └── untils.py
    │
    ├── README.md
    └── requirements.txt

## Cài đặt

### 1) Tạo môi trường Python

    python -m venv venv
    venv\Scripts\activate

### 2) Cài thư viện

    pip install -r requirements.txt

## Tiền xử lý dữ liệu

    python -m src.preprocessing

## Huấn luyện mô hình

    python -m src.train

## Chạy ứng dụng Streamlit

    streamlit run app/app.py

## Công nghệ sử dụng

-   Pandas\
-   Scikit-learn\
-   Streamlit\
-   Deep-translator\
-   Regex\
-   Logistic Regression

## Lưu ý đạo đức

Hệ thống không thay thế chuyên gia tâm lý. Chỉ mang tính hỗ trợ tham
khảo.

## Tác giả

-   Tên sinh viên:
    + Trần Thanh Luôn       MSSV: 24110280
    + Bùi Phạm Huỳnh Hương  MSSV: 24133026
    + Hồ Nhật Minh:         MSSV: 24133039
    + Nguyễn Quang Vinh 
-   Môn học: Lập trình Python
