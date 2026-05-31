# SIFT Landmark Recognition Dashboard

Hệ thống nhận diện địa danh tích hợp luồng tiền xử lý tự phục hồi góc xoay (Self-Correcting Pipeline) sử dụng thuật toán SIFT, mô hình túi từ vựng thị giác (BoVW) và máy vector hỗ trợ (SVM).

---

## 📌 Tính năng chính

- **Trích xuất đặc trưng bất biến:**
  Sử dụng SIFT để tìm các điểm đặc trưng không thay đổi dưới tác động xoay.

- **Mô hình BoVW & SPM:**
  Chuyển đặc trưng cục bộ thành vector histogram cố định, kết hợp Spatial Pyramid Matching (1x1 và 2x2) để giữ thông tin không gian.

- **Tiền xử lý tự phục hồi:**
  Tự động phát hiện góc lệch và xoay lại ảnh bằng Histogram Voting + RANSAC trước khi phân loại.

- **Giao diện Dashboard:**
  Xây dựng bằng CustomTkinter, hỗ trợ đa luồng giúp chạy mượt.

---

## 🛠 Cài đặt môi trường

Yêu cầu: **Python >= 3.9**

### 🍎 Đối với macOS / Linux
Trên macOS (đặc biệt là các phiên bản mới sử dụng Homebrew Python), việc cài đặt thư viện trực tiếp có thể gặp lỗi `externally-managed-environment`. Vì vậy, khuyến nghị sử dụng môi trường ảo (`venv`):

1. Mở Terminal và di chuyển vào thư mục dự án.
2. Khởi tạo môi trường ảo:
   ```bash
   python3 -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   ```bash
   source venv/bin/activate
   ```
4. Cài đặt các thư viện yêu cầu từ file `requirements.txt`:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 🪟 Đối với Windows
Bạn có thể cài đặt trực tiếp hoặc sử dụng môi trường ảo để quản lý thư viện sạch sẽ:

**Cách 1: Sử dụng môi trường ảo (Khuyến nghị)**
1. Mở Terminal (PowerShell hoặc Command Prompt) tại thư mục dự án.
2. Khởi tạo môi trường ảo:
   ```powershell
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   * Trên **PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * Trên **Command Prompt**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
4. Cài đặt các thư viện từ file `requirements.txt`:
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**Cách 2: Cài đặt trực tiếp vào hệ thống**
1. Mở Terminal.
2. Chạy lệnh cài đặt:
   ```cmd
   pip install -r requirements.txt
   ```

### ⚠️ Lưu ý quan trọng về Git LFS (Large File Storage)
Dự án này sử dụng **Git LFS** để quản lý các file ảnh lớn (trong thư mục `test_images` và `dataset`). Nếu bạn chỉ clone dự án bằng lệnh `git clone` thông thường mà không cài đặt Git LFS, các file ảnh sẽ chỉ là các file con trỏ văn bản nhỏ và OpenCV/PIL sẽ báo lỗi không thể đọc được ảnh.

**Hướng dẫn xử lý:**
1. Cài đặt Git LFS trên thiết bị:
   * **macOS** (sử dụng Homebrew):
     ```bash
     brew install git-lfs
     ```
   * **Windows**: Tải xuống và cài đặt bộ cài từ trang chủ [git-lfs.github.com](https://git-lfs.github.com/).
2. Di chuyển vào thư mục dự án và chạy các lệnh sau để tải về các file ảnh thực tế:
   ```bash
   git lfs install
   git lfs pull
   ```

## 📦 Dữ liệu (Dataset)
Bộ dữ liệu huấn luyện địa danh của dự án được lưu trữ dưới dạng file nén `dataset.zip` và chia sẻ công khai:
* **Link tải xuống**: [dataset.zip (Google Drive)](https://drive.google.com/file/d/1SZepSgLzSte3b7ktNxRfcuXe7dKvymRh/view?usp=sharing)

### 📂 Hướng dẫn lưu trữ chi tiết:

#### 1. Khi chạy trên máy tính cá nhân (Local / Offline)
* **Vị trí lưu**: Di chuyển file `dataset.zip` vừa tải về đặt trực tiếp vào thư mục gốc của dự án (thư mục `Nhom4_Project/`, nằm cùng cấp với file `gui_dashboard.py`).
* **Cách giải nén**: Tiến hành giải nén file `dataset.zip` ngay tại thư mục đó để tạo ra thư mục `dataset/` (đảm bảo cấu trúc đường dẫn đúng là `Nhom4_Project/dataset/`, bên trong chứa 10 thư mục con tương ứng với 10 địa danh như `caucongvang`, `chuamotcot`,...).

#### 2. Khi chạy trên Google Colab (Online / Cloud)
* **Vị trí lưu**: Mở **Google Drive** cá nhân của bạn. Tiến hành tải file `dataset.zip` trực tiếp lên và lưu ở **thư mục gốc** của Drive của bạn (tức là nằm ngay tại mục **Drive của tôi** / **My Drive**, không lưu bên trong bất kỳ thư mục con nào).
* **Lý do**: Cấu trúc này đảm bảo lệnh sao chép tự động `!cp /content/drive/MyDrive/dataset.zip .` của Notebook hoạt động chính xác và tìm thấy file.

---

## 🚀 Hướng dẫn sử dụng

### 1. Chạy trên máy tính cá nhân (Giao diện GUI)
Kích hoạt giao diện bảng điều khiển Dashboard tương tác bằng lệnh:
```bash
python gui_dashboard.py
```

* **Huấn luyện (TRAIN)**: Nhấn nút **TRAIN** trên bảng điều khiển nếu chưa có mô hình. Hệ thống sẽ trích xuất SIFT và huấn luyện K-Means + SVM. Mô hình sau khi huấn luyện xong tự động được lưu trữ vào thư mục `models/`.
* **Chọn ảnh kiểm thử**: Nhấn nút **CHỌN ẢNH** và chọn một ảnh kiểm thử bất kỳ trong thư mục `test_images/` (ảnh ngoài dataset huấn luyện) để kiểm tra tính khách quan.
* **Nhận diện**: Nhấn nút **NHẬN DIỆN**. Hệ thống tự động thực hiện: Dự đoán nhãn ban đầu -> Tìm ảnh mẫu khớp đặc trưng -> Tự xoay ảnh về đúng góc thẳng đứng -> Dự đoán lại nhãn sau chỉnh -> Xuất biểu đồ so sánh chi tiết lưu vào thư mục `results_pipeline/`.

### 2. Chạy trên Google Colab (Notebook)
Dự án tích hợp sẵn file Notebook [sift_landmark_colab.ipynb](file:///d:/Python/Projects/KhaiPhaDuLieu/Nhom4_Project/sift_landmark_colab.ipynb) được tối ưu hóa để huấn luyện và nhận diện đầy đủ trên đám mây:
1. Tải file `dataset.zip` từ đường dẫn Google Drive ở trên về máy tính cá nhân.
2. Tải file `dataset.zip` đó lên **thư mục gốc** của Google Drive cá nhân của bạn (mục **Drive của tôi** / **My Drive**).
3. Truy cập [Google Colab](https://colab.research.google.com/).
4. Chọn thẻ **Tải lên (Upload)** và tải file `sift_landmark_colab.ipynb` từ thư mục máy tính của bạn lên.
5. Tiến hành chạy tuần tự các ô code từ **Bước 1** đến **Bước 7** theo hướng dẫn chi tiết trong Notebook để nhận kết quả trực quan hóa.

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

| 🧑‍💻 Họ và tên | 🆔 Mã sinh viên | 🐙 GitHub |
|:---|:---:|:---:|
| **Nguyễn Chí Thịnh** | N22DCPT089 | [@hnihTyoB](https://github.com/hnihTyoB) |
| **Huỳnh Thanh Trà** | N22DCPT097 | [@TraDeThuong](https://github.com/TraDeThuong) |
| **Tô Duy Hào** | N22DCPT025 | [@Shunnio](https://github.com/Shunnio) |

---

## 🎓 Giảng viên hướng dẫn

Nguyễn Ngọc Duy
