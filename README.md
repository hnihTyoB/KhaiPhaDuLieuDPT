# SIFT Landmark Recognition Pipeline

Hệ thống nhận diện địa danh tích hợp luồng tiền xử lý tự phục hồi góc xoay (Self-Correcting Pipeline) sử dụng thuật toán SIFT, mô hình túi từ vựng thị giác (BoVW) và máy vector hỗ trợ (SVM).

---

## 📖 Mục lục

- [📌 Tính năng chính](#tính-năng-chính)
- [🛠 Cài đặt môi trường](#cài-đặt-môi-trường)
  - [🍎 Đối với macOS / Linux](#đối-với-macos--linux)
  - [🪟 Đối với Windows](#đối-với-windows)
  - [⚠️ Lưu ý quan trọng về Git LFS (Large File Storage)](#lưu-ý-quan-trọng-về-git-lfs-large-file-storage)
- [📦 Dữ liệu (Dataset)](#dữ-liệu-dataset)
- [🚀 Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [1. Chạy trên máy tính cá nhân (Dòng lệnh CLI)](#1-chạy-trên-máy-tính-cá-nhân-dòng-lệnh-cli)
  - [2. Chạy trên Google Colab (Notebook)](#2-chạy-trên-google-colab-notebook)
- [📂 Cấu trúc project](#cấu-trúc-project)
- [👥 Nhóm thực hiện](#nhóm-thực-hiện)
- [🎓 Giảng viên hướng dẫn](#giảng-viên-hướng-dẫn)

---

## Tính năng chính

- **Trích xuất đặc trưng bất biến:**
  Sử dụng SIFT để tìm các điểm đặc trưng không thay đổi dưới tác động xoay.

- **Mô hình BoVW & SPM:**
  Chuyển đặc trưng cục bộ thành vector histogram cố định, kết hợp Spatial Pyramid Matching (1x1 và 2x2) để giữ thông tin không gian.

- **Tiền xử lý tự phục hồi:**
  Tự động phát hiện góc lệch và xoay lại ảnh bằng Histogram Voting + RANSAC trước khi phân loại.

- **Tìm kiếm nhanh Hybrid (FAST-SIFT):**
  Tích hợp điểm số xác suất SVM và Cosine Similarity trên đặc trưng BoVW 1x1 của bộ nhớ cache để rút ngắn thời gian tìm ảnh mẫu từ ~37 giây xuống dưới 0.8 giây.

---

## Cài đặt môi trường

Yêu cầu: **Python >= 3.9**

### Đối với macOS / Linux

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

### Đối với Windows

Bạn có thể cài đặt trực tiếp hoặc sử dụng môi trường ảo để quản lý thư viện sạch sẽ:

**Cách 1: Sử dụng môi trường ảo (Khuyến nghị)**

1. Mở Terminal (PowerShell hoặc Command Prompt) tại thư mục dự án.
2. Khởi tạo môi trường ảo:
   ```powershell
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   - Trên **PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Trên **Command Prompt**:
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

### Lưu ý quan trọng về Git LFS (Large File Storage)

Dự án này sử dụng **Git LFS** để quản lý các file ảnh lớn (trong thư mục `test_images` và `dataset`). Nếu bạn chỉ clone dự án bằng lệnh `git clone` thông thường mà không cài đặt Git LFS, các file ảnh sẽ chỉ là các file con trỏ văn bản nhỏ và OpenCV/PIL sẽ báo lỗi không thể đọc được ảnh.

**Hướng dẫn xử lý:**

1. Cài đặt Git LFS trên thiết bị:
   - **macOS** (sử dụng Homebrew):
     ```bash
     brew install git-lfs
     ```
   - **Windows**: Tải xuống và cài đặt bộ cài từ trang chủ [git-lfs.github.com](https://git-lfs.github.com/).
2. Di chuyển vào thư mục dự án và chạy các lệnh sau để tải về các file ảnh thực tế:
   ```bash
   git lfs install
   git lfs pull
   ```

## Dữ liệu (Dataset)

Bộ dữ liệu huấn luyện địa danh của dự án được lưu trữ dưới dạng file nén `dataset.zip` và chia sẻ công khai:

- **Link tải xuống**: [dataset.zip (Google Drive)](https://drive.google.com/file/d/1SZepSgLzSte3b7ktNxRfcuXe7dKvymRh/view?usp=sharing)

### Hướng dẫn lưu trữ chi tiết

#### 1. Khi chạy trên máy tính cá nhân (Local / Offline)

- **Vị trí lưu**: Di chuyển file `dataset.zip` vừa tải về đặt trực tiếp vào thư mục gốc của dự án (`Nhom4_Project/`).
- **Cách giải nén**: Tiến hành giải nén file `dataset.zip` ngay tại thư mục đó để tạo ra thư mục `dataset/` (đảm bảo cấu trúc đường dẫn đúng là `Nhom4_Project/dataset/`, bên trong chứa 10 thư mục con tương ứng với 10 địa danh như `caucongvang`, `chuamotcot`,...).

#### 2. Khi chạy trên Google Colab (Online / Cloud)

- **Vị trí lưu**: Mở **Google Drive** cá nhân của bạn. Tiến hành tải file `dataset.zip` trực tiếp lên và lưu ở **thư mục gốc** của Drive của bạn (tức là nằm ngay tại mục **Drive của tôi** / **My Drive**, không lưu bên trong bất kỳ thư mục con nào).
- **Lý do**: Cấu trúc này đảm bảo lệnh sao chép tự động `!cp /content/drive/MyDrive/dataset.zip .` của Notebook hoạt động chính xác và tìm thấy file.

---

## Mô hình đã huấn luyện (Pre-trained Models)

Nếu bạn không muốn mất thời gian huấn luyện lại mô hình từ đầu, bạn có thể tải về bộ mô hình đã được nhóm huấn luyện sẵn:

- **Link tải xuống**: [models.zip (Google Drive)](https://drive.google.com/file/d/1298FmCefx6tUlkOO5VNC-iAuFRTWgSJo/view?usp=sharing)

### Hướng dẫn lưu trữ và cài đặt chi tiết

#### 1. Khi chạy trên máy tính cá nhân (Local / Offline)

1. Tải file `models.zip` về máy tính.
2. Tạo thư mục đặt tên là `models` ở thư mục gốc của dự án (`Nhom4_Project/`).
3. Giải nén file `models.zip` và di chuyển toàn bộ các file bên trong (`kmeans_model.pkl`, `svm_model.pkl`, `label_names.pkl`, `accuracy_report.txt`, `confusion_matrix.png`, `reference_cache.pkl`) đặt vào thư mục `models/` vừa tạo.

#### 2. Khi chạy trên Google Colab (Online / Cloud)

- **Vị trí lưu**: Tải file `models.zip` về máy tính, sau đó tải nó lên **thư mục gốc** của Google Drive cá nhân của bạn (mục **Drive của tôi** / **My Drive**, nằm cùng cấp với file `dataset.zip`).
- **Lý do**: Cấu trúc này giúp câu lệnh sao chép tự động `!cp /content/drive/MyDrive/models.zip .` trong Notebook tự động tải và giải nén mô hình mà không cần huấn luyện lại từ đầu.

---

## Hướng dẫn sử dụng

### 1. Chạy trên máy tính cá nhân (Dòng lệnh CLI)

Chạy chương trình giao diện dòng lệnh tương tác bằng lệnh:

```bash
python sift_landmark_pipeline.py
```

- **Nhận diện và sửa xoay tự động**:
  1. Nhập lựa chọn `1` từ menu.
  2. Nhập đường dẫn ảnh cần kiểm thử (ví dụ: `test_images/chuamotcot_rotated.jpg`).
  3. (Tùy chọn) Nhập góc giả lập xoay nếu muốn mô phỏng ảnh bị nghiêng, hoặc nhấn Enter để bỏ qua.
  4. Hệ thống tự động thực hiện: Dự đoán nhãn ban đầu -> Tìm ảnh mẫu khớp đặc trưng bằng thuật toán Hybrid FAST-SIFT -> Tự xoay ảnh về đúng góc thẳng đứng -> Dự đoán lại nhãn sau khi chỉnh -> In báo cáo so sánh độ tin cậy và lưu biểu đồ trực quan vào thư mục `results_pipeline/`.

### 2. Chạy trên Google Colab (Notebook)

Dự án tích hợp sẵn file Notebook [sift_landmark_colab.ipynb](./sift_landmark_colab.ipynb) được tối ưu hóa để huấn luyện và nhận diện đầy đủ trên đám mây:

1. Tải file `dataset.zip` từ đường dẫn Google Drive ở trên về máy tính cá nhân.
2. Tải file `dataset.zip` đó lên **thư mục gốc** của Google Drive cá nhân của bạn (mục **Drive của tôi** / **My Drive**).
3. Truy cập [Google Colab](https://colab.research.google.com/).
4. Chọn thẻ **Tải lên (Upload)** và tải file `sift_landmark_colab.ipynb` từ thư mục máy tính của bạn lên.
5. Tiến hành chạy tuần tự các ô code từ **Bước 1** đến **Bước 7** theo hướng dẫn chi tiết trong Notebook để nhận kết quả trực quan hóa.

---

## Cấu trúc project

- `landmark_sift_bovw.py`
  → Định nghĩa trích xuất SIFT, gom cụm K-Means, tạo histogram SPM và huấn luyện/dự đoán bằng SVM.

- `sift_landmark_pipeline.py`
  → Chương trình chạy pipeline, tìm kiếm ảnh mẫu (Hybrid FAST-SIFT), phát hiện góc lệch và tự xoay ảnh.

---

## Nhóm thực hiện

**Nhóm 4 - D22CQPTUD01-N**

| 🧑‍💻 Họ và tên | 🆔 Mã sinh viên | 🐙 GitHub | 🛠️ Nhiệm vụ chính trong dự án | Đóng góp |
| :--- | :---: | :---: | :--- | :---: |
| **Nguyễn Chí Thịnh**<br>(Nhóm trưởng) | N22DCPT089 | [@hnihTyoB](https://github.com/hnihTyoB) | – Nghiên cứu đề xuất kiến trúc nhận diện kết hợp SIFT, BoVW và SVM.<br>– Lập trình xây dựng từ điển thị giác Codebook (K-Means) và trích xuất SIFT descriptors cục bộ.<br>– Phát triển cơ chế Spatial Pyramid Matching (SPM 1x1 + 2x2) giúp giữ thông tin cấu trúc không gian của địa danh.<br>– Phân tích, đánh giá chéo lựa chọn tham số tối ưu ($K = 600$) và xuất báo cáo ma trận nhầm lẫn (Confusion Matrix). | 34% |
| **Huỳnh Thanh Trà** | N22DCPT097 | [@TraDeThuong](https://github.com/TraDeThuong) | – Thiết lập mô hình phân loại SVM (Linear Kernel) kết hợp tính toán xác suất dự đoán (Confidence).<br>– Lập trình bộ so khớp SIFT (Lowe's ratio test) và lọc RANSAC loại bỏ outliers.<br>– Phát triển thuật toán ước lượng góc xoay lệch (Orientation Histogram Voting) và tự động xoay ảnh bằng ma trận WarpAffine.<br>– Xây dựng chương trình pipeline nhận diện tự sửa góc xoay dạng dòng lệnh tương tác (CLI). | 33% |
| **Tô Duy Hào** | N22DCPT025 | [@Shunnio](https://github.com/Shunnio) | – Thu thập, chuẩn hóa dữ liệu 10 địa danh gồm 618 ảnh mẫu và bộ ảnh kiểm thử.<br>– Thiết kế Reference Cache và lập trình thuật toán tìm kiếm lai **FAST-SIFT** (SVM + Cosine Similarity) giúp tăng tốc độ tìm ảnh mẫu gấp 26 lần.<br>– Xây dựng và tối ưu hóa file Notebook chạy trên Google Colab (tích hợp tải/giải nén dataset và models tự động từ Google Drive).<br>– Đo lường độ tương đồng cấu trúc SSIM của ảnh sau phục hồi và biên soạn cẩm nang bảo vệ đồ án. | 33% |

---

## Giảng viên hướng dẫn

Nguyễn Ngọc Duy
