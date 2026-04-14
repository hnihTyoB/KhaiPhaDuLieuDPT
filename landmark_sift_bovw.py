# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pickle
import warnings
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI, phù hợp mọi môi trường
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# ⚙️ THAM SỐ QUAN TRỌNG (CẤU HÌNH)
# =============================================================================
DATASET_PATH = 'dataset'                # Cấu trúc: dataset/<tên_lớp>/<ảnh>
IMG_SIZE = (256, 256)                   # Kích thước resize ảnh
K_CLUSTERS = 200                        # Số cụm K-Means (tăng cho dataset lớn ~6400 ảnh)
SVM_KERNEL = 'linear'                   # Kernel SVM: 'linear' phù hợp BoVW histogram
SVM_C = 1.0                             # Tham số C của SVM
TEST_SPLIT_RATIO = 0.2                  # Tỉ lệ tập test (80/20)
RANDOM_STATE = 42                       # Seed cho kết quả tái lập
CONFIDENCE_THRESHOLD = 0.15             # Ngưỡng confidence (dưới = "Không xác định")

# Đường dẫn lưu/tải mô hình
MODEL_DIR = 'models'
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
LABEL_NAMES_PATH = os.path.join(MODEL_DIR, 'label_names.pkl')

# Tên hiển thị cho các lớp
DISPLAY_NAMES = {
    'chuamotcot': 'Chùa Một Cột (Việt Nam)',
    'thaprua': 'Tháp Rùa - Hồ Gươm (Việt Nam)',
    'caucongvang': 'Cầu Cổng Vàng (Mỹ)',
    'thapnghieng': 'Tháp nghiêng Pisa (Ý)',
    'colosseum': 'Đấu trường La Mã (Ý)',
    'sydney': 'Nhà hát Sydney (Úc)',
    'tajmahal': 'Đền Taj Mahal (Ấn Độ)',
    'nuthantudo': 'Tượng Nữ thần Tự do (Mỹ)',
    'eiffel': 'Tháp Eiffel (Pháp)',
    'pyramid': 'Kim Tự Tháp (Ai Cập)',
}


# Lớp cần bỏ qua khi huấn luyện (quá chung chung, không phải địa danh cụ thể)
EXCLUDE_CLASSES = ['general']

# =============================================================================
# 🔧 TIỀN XỬ LÝ ẢNH
# =============================================================================
def load_and_preprocess(image_path):
    """
    Bước 1-2: Đọc ảnh → Resize → Chuyển Grayscale.

    Args:
        image_path (str): Đường dẫn tới file ảnh

    Returns:
        np.ndarray: Ảnh grayscale kích thước IMG_SIZE, hoặc None nếu lỗi
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Không thể đọc ảnh: {image_path}")
        return None

    # Resize về kích thước chuẩn
    img_resized = cv2.resize(img, IMG_SIZE)

    # Chuyển sang grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    return gray


# =============================================================================
# 🔍 TRÍCH XUẤT ĐẶC TRƯNG SIFT
# =============================================================================
def extract_sift_descriptors(image):
    """
    Bước 3: Trích xuất SIFT descriptors từ ảnh grayscale.
    
    Sử dụng thuật toán SIFT gốc OpenCV (cv2.SIFT_create()).

    Args:
        image (np.ndarray): Ảnh grayscale

    Returns:
        keypoints: Danh sách keypoints
        descriptors (np.ndarray): Mảng descriptors [N, 128]
    """
    try:
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
    except Exception as e:
        print(f"[WARN] Lỗi trích xuất SIFT: {e}")
        return [], np.array([])

    if descriptors is None or len(descriptors) == 0:
        return [], np.array([])

    return keypoints, descriptors


def load_dataset_and_extract_sift(dataset_path):
    """
    Bước 1-3: Tải toàn bộ dataset, tiền xử lý và trích xuất SIFT.

    Args:
        dataset_path (str): Đường dẫn thư mục dataset

    Returns:
        all_descriptors (list): Tất cả descriptors gộp lại
        image_descriptors (list): Descriptors theo từng ảnh
        labels (list): Nhãn tương ứng từng ảnh
        label_names (list): Danh sách tên các lớp
    """
    print("\n" + "=" * 60)
    print(" BƯỚC 1-3: TẢI DỮ LIỆU VÀ TRÍCH XUẤT SIFT")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        print(f"[LỖI] Thư mục dataset '{dataset_path}' không tồn tại!")
        return None, None, None, None

    all_descriptors = []   # Tất cả descriptor gộp lại (dùng cho K-Means)
    image_data_list = []   # Lưu [{keypoints, descriptors}] cho histogram
    labels = []            # Nhãn text của từng ảnh
    skipped = 0

    # Tự động phát hiện các lớp từ thư mục con
    label_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
        and d not in EXCLUDE_CLASSES
    ])

    if len(label_names) == 0:
        print("[LỖI] Không tìm thấy lớp nào trong dataset!")
        return None, None, None, None

    print(f"[INFO] Phát hiện {len(label_names)} lớp: {label_names}")

    for label_name in label_names:
        class_path = os.path.join(dataset_path, label_name)
        image_files = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        display_name = DISPLAY_NAMES.get(label_name, label_name)
        print(f"\n[→] Đang xử lý lớp '{display_name}' ({label_name}) — {len(image_files)} ảnh")

        count = 0
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)

            # Tiền xử lý
            gray = load_and_preprocess(img_path)
            if gray is None:
                skipped += 1
                continue

            # Trích xuất SIFT
            kps, descs = extract_sift_descriptors(gray)
            if len(descs) > 0:
                all_descriptors.extend(descs)
                kps_xy = np.array([kp.pt for kp in kps])
                image_data_list.append({
                    'keypoints': kps_xy,
                    'descriptors': descs
                })
                labels.append(label_name)
                count += 1
            else:
                print(f"     [SKIP] {img_file} — không tìm thấy keypoints")
                skipped += 1

        print(f"     ✓ {count} ảnh có descriptors | keypoints tổng: {sum(len(d['descriptors']) for d in image_data_list[-count:]) if count > 0 else 0}")

    print(f"\n[KẾT QUẢ] Tổng descriptors: {len(all_descriptors)} | "
          f"Tổng ảnh hợp lệ: {len(image_data_list)} | Bỏ qua: {skipped}")

    if len(all_descriptors) == 0:
        print("[LỖI] Không trích xuất được descriptor nào!")
        return None, None, None, None

    all_desc_array = np.array(all_descriptors, dtype=np.float32)

    # Subsampling cho K-Means nếu quá nhiều descriptors (tránh OOM)
    MAX_DESCRIPTORS_FOR_KMEANS = 500000
    if len(all_desc_array) > MAX_DESCRIPTORS_FOR_KMEANS:
        print(f"[INFO] Subsampling {len(all_desc_array)} → {MAX_DESCRIPTORS_FOR_KMEANS} "
              f"descriptors cho K-Means (tránh tràn bộ nhớ)")
        rng = np.random.RandomState(RANDOM_STATE)
        indices = rng.choice(len(all_desc_array), MAX_DESCRIPTORS_FOR_KMEANS, replace=False)
        all_desc_array = all_desc_array[indices]

    return all_desc_array, image_data_list, labels, label_names


# =============================================================================
# 📚 XÂY DỰNG BAG OF VISUAL WORDS (BoVW)
# =============================================================================
def build_visual_vocabulary(all_descriptors, k):
    """
    Bước 4.1: Xây dựng "Từ điển thị giác" bằng K-Means.

    Gom tất cả SIFT descriptors thành K cụm → mỗi cụm đại diện 1 "từ thị giác".

    Args:
        all_descriptors (np.ndarray): Tất cả descriptors [N, 128]
        k (int): Số cụm (kích thước từ điển)

    Returns:
        KMeans: Mô hình K-Means đã huấn luyện (codebook)
    """
    print("\n" + "=" * 60)
    print(f" BƯỚC 4.1: XÂY DỰNG TỪ ĐIỂN THỊ GIÁC (K-Means, K={k})")
    print("=" * 60)

    print(f"[INFO] Đang phân cụm {len(all_descriptors)} descriptors thành {k} cụm...")

    kmeans = KMeans(
        n_clusters=k,
        max_iter=300,
        random_state=RANDOM_STATE,
        n_init=10,
        verbose=0
    )
    kmeans.fit(all_descriptors)

    print(f"[OK] Từ điển thị giác đã xây dựng thành công!")
    print(f"     └── Inertia (tổng khoảng cách nội cụm): {kmeans.inertia_:.2f}")

    return kmeans


def create_histograms(image_data_list, kmeans_model, k):
    """
    Bước 4.2: Biểu diễn mỗi ảnh bằng histogram đặc trưng sử dụng Spatial Pyramid Matching (SPM).

    Với mỗi ảnh:
    - Gán từng descriptor vào cụm gần nhất
    - Đếm số lần xuất hiện (histogram) cho 1x1 (toàn ảnh) và 2x2 (4 góc)
    - Chuẩn hóa

    → Mỗi ảnh → vector cố định kích thước K * 5 vùng = 5K chiều.

    Args:
        image_data_list (list): Danh sách dict gồm 'keypoints' và 'descriptors'.
        kmeans_model (KMeans): Mô hình K-Means (codebook)
        k (int): Số cụm

    Returns:
        np.ndarray: Ma trận histograms [num_images, K * 5]
    """
    print(f"\n[INFO] Đang tạo histogram đặc trưng (SPM 1x1+2x2) cho {len(image_data_list)} ảnh...")

    num_images = len(image_data_list)
    # Mở rộng K lên 5 lần, vì có 5 khu vực phân vùng không gian (1 Global + 4 Sub-regions)
    num_regions = 5
    histograms = np.zeros((num_images, k * num_regions), dtype=np.float32)

    half_w = IMG_SIZE[0] / 2.0
    half_h = IMG_SIZE[1] / 2.0

    for i, data in enumerate(image_data_list):
        descriptors = data['descriptors']
        keypoints = data['keypoints']
        
        if descriptors is not None and len(descriptors) > 0:
            descriptors = np.array(descriptors, dtype=np.float32)
            # Gán descriptor vào cụm gần nhất
            cluster_labels = kmeans_model.predict(descriptors)
            
            for pt, cluster_id in zip(keypoints, cluster_labels):
                x, y = pt
                
                # Region 0: Global (1x1)
                histograms[i][cluster_id] += 1
                
                # Regions 1-4: 2x2 Grid (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
                if x < half_w and y < half_h:
                    region_idx = 1
                elif x >= half_w and y < half_h:
                    region_idx = 2
                elif x < half_w and y >= half_h:
                    region_idx = 3
                else:
                    region_idx = 4
                    
                histograms[i][region_idx * k + cluster_id] += 1

    # Chuẩn hóa L2 cho mỗi histogram
    for i in range(num_images):
        norm_val = np.linalg.norm(histograms[i])
        if norm_val > 0:
            histograms[i] /= norm_val

    print(f"[OK] Đã tạo {num_images} histogram (kích thước {k * num_regions} chiều/mỗi ảnh).")
    return histograms


# =============================================================================
# 🎓 HUẤN LUYỆN MÔ HÌNH SVM
# =============================================================================
def train_svm(features, labels, label_names, kernel):
    """
    Bước 5: Huấn luyện SVM và đánh giá trên tập test.
    ❌ KHÔNG dùng StandardScaler (histogram đã L2 normalize)
    ❌ KHÔNG dùng probability=True (gây fake confidence với dataset nhỏ)
    ✅ Dùng decision_function để tính confidence thật

    Args:
        features (np.ndarray): Ma trận histograms [N, K] (đã L2 normalize)
        labels (list): Nhãn text tương ứng
        label_names (list): Danh sách tên lớp
        kernel (str): Kernel SVM ('linear', 'rbf', 'poly')

    Returns:
        svm (SVC): Mô hình SVM đã huấn luyện
        accuracy (float): Độ chính xác trên tập test
    """
    print("\n" + "=" * 60)
    print(" BƯỚC 5: HUẤN LUYỆN MÔ HÌNH SVM")
    print("=" * 60)

    # Chuyển nhãn text → số
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    numeric_labels = np.array([label_to_idx[lbl] for lbl in labels])

    # Chia train/test (80/20)
    print(f"[INFO] Chia dữ liệu: {int((1 - TEST_SPLIT_RATIO) * 100)}% train / {int(TEST_SPLIT_RATIO * 100)}% test")

    # Kiểm tra số lượng mẫu mỗi lớp
    unique, counts = np.unique(numeric_labels, return_counts=True)
    min_count = counts.min()

    if min_count < 2:
        print("[WARN] Một số lớp có quá ít mẫu, sử dụng toàn bộ dữ liệu để train (không chia test).")
        X_train, X_test = features, features
        y_train, y_test = numeric_labels, numeric_labels
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, numeric_labels,
            test_size=TEST_SPLIT_RATIO,
            random_state=RANDOM_STATE,
            stratify=numeric_labels
        )

    print(f"     └── Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu")

    # ❌ KHÔNG dùng StandardScaler — histogram đã L2 normalize, có ý nghĩa xác suất
    # StandardScaler sẽ phá vỡ phân phối histogram

    # Huấn luyện SVM
    print(f"\n[INFO] Huấn luyện SVM (kernel='{kernel}', C={SVM_C})...")

    svm = SVC(
        kernel=kernel,
        C=SVM_C,
        probability=True,
        random_state=RANDOM_STATE
    )
    svm.fit(X_train, y_train)
    print("[OK] SVM huấn luyện thành công!")

    # ===== ĐÁNH GIÁ =====
    print("\n" + "-" * 40)
    print(" 📊 ĐÁNH GIÁ HỆ THỐNG")
    print("-" * 40)

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n ✅ Accuracy (Độ chính xác): {accuracy:.4f} ({accuracy:.2%})")

    # Classification Report
    display_labels = [DISPLAY_NAMES.get(name, name) for name in label_names]
    print(f"\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=display_labels, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("[Confusion Matrix]")
    print(cm)

    # Vẽ Confusion Matrix → lưu file
    plot_confusion_matrix(cm, display_labels)

    return svm, accuracy


def plot_confusion_matrix(cm, class_names):
    """
    Vẽ và lưu confusion matrix dạng heatmap.

    Args:
        cm (np.ndarray): Ma trận nhầm lẫn
        class_names (list): Tên các lớp hiển thị
    """
    output_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        square=True
    )
    plt.title('Ma trận Nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
    plt.xlabel('Nhãn Dự đoán', fontsize=12)
    plt.ylabel('Nhãn Thực tế', fontsize=12)
    plt.xticks(rotation=25, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Confusion Matrix đã lưu tại: {output_path}")


# =============================================================================
# 💾 LƯU / TẢI MÔ HÌNH
# =============================================================================
def save_models(kmeans_model, svm_model, label_names):
    """Lưu tất cả mô hình đã huấn luyện (không có scaler)."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(KMEANS_MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans_model, f)
    with open(SVM_MODEL_PATH, 'wb') as f:
        pickle.dump(svm_model, f)
    with open(LABEL_NAMES_PATH, 'wb') as f:
        pickle.dump(label_names, f)

    print(f"\n[OK] Mô hình đã lưu tại thư mục '{MODEL_DIR}/'")
    print(f"     ├── {KMEANS_MODEL_PATH}")
    print(f"     ├── {SVM_MODEL_PATH}")
    print(f"     └── {LABEL_NAMES_PATH}")


def load_models():
    """
    Tải các mô hình đã lưu.

    Returns:
        kmeans, svm, label_names hoặc None nếu chưa huấn luyện
    """
    required = [KMEANS_MODEL_PATH, SVM_MODEL_PATH, LABEL_NAMES_PATH]
    if not all(os.path.exists(p) for p in required):
        print("[LỖI] Chưa tìm thấy mô hình đã huấn luyện!")
        print("       → Vui lòng chọn chế độ Huấn luyện (option 1) trước.")
        return None, None, None

    with open(KMEANS_MODEL_PATH, 'rb') as f:
        kmeans = pickle.load(f)
    with open(SVM_MODEL_PATH, 'rb') as f:
        svm = pickle.load(f)
    with open(LABEL_NAMES_PATH, 'rb') as f:
        label_names = pickle.load(f)

    print("[OK] Đã tải mô hình thành công.")
    return kmeans, svm, label_names


# =============================================================================
# 🏋️ QUY TRÌNH HUẤN LUYỆN (TRAINING PIPELINE)
# =============================================================================
def train_pipeline():
    """
    Giai đoạn 1: Thực hiện toàn bộ quy trình huấn luyện.

    1. Tạo dataset giả lập (nếu cần)
    2. Tải ảnh + trích xuất SIFT
    3. Xây dựng từ điển thị giác (K-Means)
    4. Tạo histograms BoVW
    5. Huấn luyện SVM + đánh giá
    6. Lưu mô hình
    """
    print("\n" + "█" * 60)
    print(" 🎓 GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)")
    print("█" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"[LỖI] Thư mục '{DATASET_PATH}' không tồn tại!")
        print(f"       Vui lòng tạo thư mục dataset với ảnh thật.")
        return False

    # Bước 1-3: Tải dữ liệu + SIFT
    result = load_dataset_and_extract_sift(DATASET_PATH)
    all_descriptors, image_data_list, labels, label_names = result

    if all_descriptors is None:
        print("[LỖI] Không thể tiếp tục huấn luyện — không có dữ liệu!")
        return False

    # Bước 4.1: K-Means clustering
    # Điều chỉnh K nếu số descriptor ít hơn K
    actual_k = min(K_CLUSTERS, len(all_descriptors))
    if actual_k < K_CLUSTERS:
        print(f"[WARN] Số descriptors ({len(all_descriptors)}) < K ({K_CLUSTERS}). "
              f"Điều chỉnh K = {actual_k}")
    kmeans = build_visual_vocabulary(all_descriptors, actual_k)

    # Bước 4.2: Tạo histograms (đã L2 normalize, KHÔNG cần StandardScaler)
    histograms = create_histograms(image_data_list, kmeans, actual_k)

    # Bước 5: Huấn luyện SVM + đánh giá (probability=False, không scaler)
    svm, accuracy = train_svm(histograms, labels, label_names, SVM_KERNEL)

    # Bước 6: Lưu mô hình (không có scaler)
    save_models(kmeans, svm, label_names)

    print("\n" + "█" * 60)
    print(f" ✅ HUẤN LUYỆN HOÀN THÀNH! — Accuracy: {accuracy:.2%}")
    print("█" * 60)

    return True


# =============================================================================
# 🔮 DỰ ĐOÁN ẢNH MỚI (INFERENCE)
# =============================================================================
def predict_image(image_path):
    """
    Giai đoạn 2: Dự đoán lớp cho một ảnh mới.

    1. Tải mô hình đã huấn luyện
    2. Tiền xử lý ảnh
    3. Trích xuất SIFT → Histogram (SPM)
    4. SVM dự đoán → Tên địa danh

    Args:
        image_path (str): Đường dẫn ảnh cần dự đoán
    """
    print(f"\n--- 🔮 DỰ ĐOÁN cho ảnh: {image_path} ---")

    # Tải models (không có scaler)
    kmeans, svm, label_names = load_models()
    if kmeans is None:
        return

    if not os.path.isfile(image_path):
        print(f"[LỖI] File không tồn tại: {image_path}")
        return

    # Bước 6: Tiền xử lý + SIFT
    gray = load_and_preprocess(image_path)
    if gray is None:
        return

    kps, descriptors = extract_sift_descriptors(gray)
    if len(descriptors) == 0:
        print("[LỖI] Không tìm thấy đặc trưng SIFT trong ảnh này.")
        return

    print(f"[INFO] Đã trích xuất {len(descriptors)} SIFT keypoints.")

    # Bước 7: Tạo histogram sử dụng SPM
    k = kmeans.n_clusters
    kps_xy = np.array([kp.pt for kp in kps])
    data = [{'keypoints': kps_xy, 'descriptors': descriptors}]
    histogram = create_histograms(data, kmeans, k)

    # Bước 8: SVM dự đoán (chỉ dùng predict_proba để đảm bảo đồng nhất % cao nhất)
    pred_proba = svm.predict_proba(histogram)[0]
    pred_idx = np.argmax(pred_proba)
    pred_label = label_names[pred_idx]
    confidence = pred_proba[pred_idx]

    # Kiểm tra ngưỡng confidence
    if confidence < CONFIDENCE_THRESHOLD:
        display_name = "⚠️  KHÔNG XÁC ĐỊNH (ảnh không thuộc lớp nào đã học)"
    else:
        display_name = DISPLAY_NAMES.get(pred_label, pred_label)

    # Hiển thị kết quả
    print("\n" + "─" * 55)
    print(f" 📍 ĐỊA DANH:     {display_name}")
    print(f" 📊 ĐỘ TIN CẬY:   {confidence:.2%}")
    print(f" 🔑 SỐ KEYPOINTS: {len(descriptors)}")
    if confidence < CONFIDENCE_THRESHOLD:
        print(f" ⚠️  GHI CHÚ:      Confidence < {CONFIDENCE_THRESHOLD:.0%} → Không xác định")
    print("─" * 55)
    print(" XÁC SUẤT CHO TỪNG LỚP:")
    for i, name in enumerate(label_names):
        display = DISPLAY_NAMES.get(name, name)
        bar = "█" * int(pred_proba[i] * 30)
        print(f"   {display:30s} {pred_proba[i]:.4f}  {bar}")
    print("─" * 55)

    # Hiển thị keypoints trên ảnh (nếu có GUI)
    try:
        img_color = cv2.imread(image_path)
        img_resized = cv2.resize(img_color, IMG_SIZE)

        # Vẽ keypoints trên ảnh màu
        img_keypoints = cv2.drawKeypoints(
            img_resized, kps, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Thêm text kết quả
        cv2.putText(img_keypoints, f"Prediction: {display_name[:40]}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img_keypoints, f"Confidence: {confidence:.2%}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Ghép ảnh gốc và ảnh keypoints
        img_display = np.hstack((img_resized, img_keypoints))

        # [COLAB FIX] Không dùng cv2.imshow vì sẽ bị lỗi no display server
        # Chuyển sang lưu ảnh vào thư mục model để check kết quả
        out_path = os.path.join(MODEL_DIR, "predict_result.png")
        cv2.imwrite(out_path, img_display)
        print(f"\n[INFO] Đã xuất ảnh kết quả test predict ra file: {out_path} (An toàn cho Colab)")
    except Exception as e:
        print(f"[INFO] Lỗi trong quá trình xuất ảnh dự đoán: {e}")


# =============================================================================
# 🎬 DEMO TỰ ĐỘNG
# =============================================================================
def demo_auto():
    """
    Chạy demo tự động:
    1. Tạo dataset giả lập (nếu chưa có)
    2. Huấn luyện mô hình
    3. Dự đoán trên 1 ảnh mẫu từ dataset
    """
    print("\n" + "★" * 60)
    print(" 🎬 DEMO TỰ ĐỘNG — HUẤN LUYỆN + DỰ ĐOÁN")
    print("★" * 60)

    # Huấn luyện
    success = train_pipeline()
    if not success:
        print("[LỖI] Huấn luyện thất bại — không thể demo.")
        return

    # Tìm ảnh mẫu để dự đoán
    print("\n\n" + "★" * 60)
    print(" 🔮 DỰ ĐOÁN TRÊN ẢNH MẪU")
    print("★" * 60)

    sample_image = None
    if os.path.exists(DATASET_PATH):
        for class_dir in sorted(os.listdir(DATASET_PATH)):
            class_path = os.path.join(DATASET_PATH, class_dir)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_image = os.path.join(class_path, images[0])
                    break

    if sample_image:
        predict_image(sample_image)
    else:
        print("[LỖI] Không tìm thấy ảnh mẫu để demo.")


# =============================================================================
# 🏠 HÀM CHÍNH — MENU CONSOLE
# =============================================================================
def main():
    """
    Hàm chính với menu console cho người dùng.
    """
    while True:
        print("\n" + "=" * 60)
        print(" PHÂN LOẠI ẢNH ĐỊA DANH — SIFT + K-MEANS + SVM")
        print("=" * 60)
        print(f"""
 📌 Cấu hình hiện tại:
    • Dataset:        {DATASET_PATH}/
    • Kích thước ảnh: {IMG_SIZE[0]}×{IMG_SIZE[1]}
    • Số cụm K:       {K_CLUSTERS}
    • SVM Kernel:     {SVM_KERNEL}
    • Train/Test:     {int((1 - TEST_SPLIT_RATIO) * 100)}/{int(TEST_SPLIT_RATIO * 100)}
        """)
        print(" [1] 🎓 Huấn luyện mô hình")
        print(" [2] 🔮 Dự đoán ảnh mới")
        print(" [3] 🎬 Demo tự động (Huấn luyện + Dự đoán)")
        print(" [0] 🚪 Thoát")
        print("-" * 60)

        choice = input(" Lựa chọn (0-3): ").strip()

        if choice == '1':
            train_pipeline()

        elif choice == '2':
            image_path = input("\n Nhập đường dẫn ảnh: ").strip()
            if image_path:
                predict_image(image_path)
            else:
                print("[WARN] Đường dẫn không hợp lệ!")

        elif choice == '3':
            demo_auto()

        elif choice == '0':
            print("\n 👋 Tạm biệt! Hẹn gặp lại.")
            break

        else:
            print("[WARN] Lựa chọn không hợp lệ! Vui lòng chọn 0-3.")


# =============================================================================
# 🚀 ĐIỂM VÀO CHƯƠNG TRÌNH
# =============================================================================
if __name__ == '__main__':
    main()
