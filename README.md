# SIFT Landmark Recognition Dashboard

Hệ thống nhận diện địa danh tích hợp luồng tiền xử lý tự phục hồi góc xoay (Self-Correcting Pipeline) sử dụng thuật toán SIFT, mô hình túi từ vựng thị giác (BoVW) và máy vector hỗ trợ (SVM).

---

## 📌 Tính năng chính

- **Trích xuất đặc trưng bất biến:**
  Sử dụng SIFT để tìm các điểm đặc trưng không thay đổi dưới tác động xoay và scale.

- **Mô hình BoVW & SPM:**
  Chuyển đặc trưng cục bộ thành vector histogram cố định, kết hợp Spatial Pyramid Matching (1x1 và 2x2) để giữ thông tin không gian.

- **Tiền xử lý tự phục hồi:**
  Tự động phát hiện góc lệch và xoay lại ảnh bằng Histogram Voting + RANSAC trước khi phân loại.

- **Giao diện Dashboard:**
  Xây dựng bằng CustomTkinter, hỗ trợ đa luồng giúp chạy mượt.

---

## 🛠 Cài đặt môi trường

Yêu cầu: **Python >= 3.9**

### 🪟 Đối với Windows
1. Mở Terminal (Command Prompt hoặc PowerShell).
2. Cài đặt các thư viện yêu cầu:
   ```cmd
   pip install opencv-python numpy scikit-learn customtkinter Pillow matplotlib seaborn
   ```

### 🍎 Đối với macOS / Linux
1. Mở Terminal.
2. Cài đặt các thư viện yêu cầu:
   ```bash
   pip3 install opencv-python numpy scikit-learn customtkinter Pillow matplotlib seaborn
   ```

---

## 🚀 Hướng dẫn sử dụng

### 1. Chạy ứng dụng

```bash
python gui_dashboard.py
```

### 2. Huấn luyện (TRAIN)

- Nhấn nút **TRAIN**
- Hệ thống sẽ:
  - Trích xuất SIFT
  - Train K-Means + SVM

- Model sẽ lưu trong thư mục `models/`

### 3. Chọn ảnh

- Nhấn **CHỌN ẢNH**

**Lưu ý:**
Nên dùng ảnh mới (không nằm trong dataset) để test khách quan

### 4. Nhận diện

- Nhấn **NHẬN DIỆN**
- Hệ thống sẽ:
  - Tự xoay ảnh về đúng góc
  - Dự đoán label

---

## 📂 Cấu trúc project

- `gui_dashboard.py`
  → Giao diện + xử lý đa luồng

- `landmark_sift_bovw.py`
  → SIFT + BoVW + SVM

- `sift_landmark_pipeline.py`
  → Pipeline xử lý xoay ảnh

---

## 👥 Nhóm thực hiện

**Nhóm 4 - D22CQPTUD01-N**

- Nguyễn Chí Thịnh (N22DCPT089)
- Huỳnh Thanh Trà (N22DCPT097)
- Tô Duy Hào (N22DCPT025)

---

## 🎓 Giảng viên hướng dẫn

Nguyễn Ngọc Duy
