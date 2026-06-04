# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from landmark_sift_bovw import (
    load_models,
    load_and_preprocess,
    extract_sift_descriptors,
    create_histograms,
    train_pipeline,
    IMG_SIZE,
    DATASET_PATH,
    DISPLAY_NAMES,
    CONFIDENCE_THRESHOLD,
    REFERENCE_CACHE_PATH,
    EXCLUDE_CLASSES,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, 'results_pipeline'))
RATIO_THRESHOLD = 0.75 # Ngưỡng Lowe's ratio test
NUM_REF_PER_CLASS = 40 # Số ảnh tham chiếu trên mỗi lớp
MIN_GOOD_MATCHES = 11 # Số matches tối thiểu để phát hiện xoay


def sift_extract(image):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)


def sift_match(desc1, desc2, kp1=None, kp2=None):
    # BFMatcher + Lowe's ratio test + RANSAC Homography validation.
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    # Lowe's ratio test
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_THRESHOLD * n.distance:
                good.append(m)
                
    # Lọc bằng RANSAC nếu có đủ keypoints và truyền kp1, kp2
    if kp1 is not None and kp2 is not None and len(good) >= 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if mask is not None:
            good = [m for i, m in enumerate(good) if mask[i][0] == 1]

    return good


def detect_rotation_angle(kp_ref, kp_user, good_matches):
    # Phát hiện góc xoay bằng histogram voting trên SIFT keypoint orientation
    if len(good_matches) < MIN_GOOD_MATCHES:
        return None, 0

    # Tính hiệu góc orientation
    angle_diffs = []
    for m in good_matches:
        a_ref = kp_ref[m.queryIdx].angle
        a_user = kp_user[m.trainIdx].angle
        diff = (a_ref - a_user) % 360
        angle_diffs.append(diff)

    angle_diffs = np.array(angle_diffs)

    # Histogram voting: 360 bins, mỗi bin = 1 độ
    hist, _ = np.histogram(angle_diffs, bins=360, range=(0, 360))
    peak_bin = np.argmax(hist)
    peak_angle = peak_bin + 0.5

    # Refine: lấy các diffs trong ±5° quanh peak
    mask = np.abs((angle_diffs - peak_angle + 180) % 360 - 180) < 5.0
    votes = int(mask.sum())

    if votes == 0:
        return float(peak_angle), 1

    # Circular mean (tránh lỗi trung bình ở biên 0/360)
    nearby = angle_diffs[mask]
    sin_mean = np.mean(np.sin(np.radians(nearby)))
    cos_mean = np.mean(np.cos(np.radians(nearby)))
    refined = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360

    # Chuyển về [-180, 180]
    if refined > 180:
        refined -= 360

    return float(refined), votes


def rotate_image(image, angle_deg):
    # Xoay ảnh quanh tâm
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def compute_ssim(img1, img2):
    # Tính SSIM giữa 2 ảnh (0=khác hoàn toàn, 1=giống hệt)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
    s1 = cv2.GaussianBlur(i1 ** 2, (11, 11), 1.5) - mu1 ** 2
    s2 = cv2.GaussianBlur(i2 ** 2, (11, 11), 1.5) - mu2 ** 2
    s12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return float(ssim_map.mean())


# TÌM ẢNH MẪU PHÙ HỢP NHẤT
def get_class_images(dataset_path, class_name, num_images=10):
    # Lấy ảnh đại diện từ MỘT lớp cụ thể trong dataset
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        return []

    images = sorted([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    if not images:
        return []

    # Phân bố đều
    if len(images) <= num_images:
        selected = images
    else:
        indices = np.linspace(0, len(images) - 1, num_images, dtype=int)
        selected = [images[i] for i in indices]

    return [os.path.join(class_dir, f) for f in selected]


# =====================================================================
# HỆ THỐNG CACHE ẢNH MẪU (Reference Cache)
# Tính trước histogram BoVW 1x1 (bất biến xoay) + lưu SIFT descriptors
# để tìm ảnh mẫu nhanh bằng so sánh vector thay vì duyệt tuần tự.
# =====================================================================

def _serialize_keypoints(keypoints):
    """Chuyển danh sách cv2.KeyPoint thành list of tuples để pickle được."""
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]


def _deserialize_keypoints(data):
    """Khôi phục danh sách cv2.KeyPoint từ list of tuples."""
    keypoints = []
    for (pt, size, angle, response, octave, class_id) in data:
        kp = cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle,
                          response=response, octave=octave, class_id=class_id)
        keypoints.append(kp)
    return keypoints


def build_reference_cache(kmeans_model):
    """
    Xây dựng cache ảnh mẫu (chạy 1 lần sau khi Train).
    Với mỗi ảnh mẫu trong dataset:
      - Trích xuất SIFT keypoints + descriptors
      - Tính histogram BoVW 1x1 toàn cục (K chiều, bất biến xoay)
      - Lưu tất cả vào file pickle
    """
    print("\n" + "=" * 60)
    print("XÂY DỰNG REFERENCE CACHE (Histogram 1x1 + SIFT)")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"[LỖI] Không tìm thấy thư mục dataset: {DATASET_PATH}")
        return False

    k = kmeans_model.n_clusters
    entries = []
    all_histograms = []

    # Phát hiện các lớp (loại trừ EXCLUDE_CLASSES)
    label_names = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
        and d not in EXCLUDE_CLASSES
    ])

    print(f"[INFO] Đang cache {len(label_names)} lớp, mỗi lớp tối đa {NUM_REF_PER_CLASS} ảnh...\n")

    total_cached = 0
    for class_name in label_names:
        ref_images = get_class_images(DATASET_PATH, class_name, NUM_REF_PER_CLASS)
        if not ref_images:
            continue

        display = DISPLAY_NAMES.get(class_name, class_name)
        count = 0

        for ref_path in ref_images:
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                continue
            ref_img = cv2.resize(ref_img, IMG_SIZE)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

            # Trích xuất SIFT (dùng SIFT mặc định, giống sift_extract)
            kp_ref, desc_ref = sift_extract(ref_gray)
            if desc_ref is None or len(desc_ref) < 2:
                continue

            # Tính histogram 1x1 toàn cục (K chiều, bất biến xoay)
            hist = np.zeros(k, dtype=np.float32)
            cluster_labels = kmeans_model.predict(desc_ref.astype(np.float32))
            for cid in cluster_labels:
                hist[cid] += 1
            norm = np.linalg.norm(hist)
            if norm > 0:
                hist /= norm

            all_histograms.append(hist)
            entries.append({
                'class_name': class_name,
                'file_path': ref_path,
                'keypoints_data': _serialize_keypoints(kp_ref),
                'descriptors': desc_ref,
            })
            count += 1

        print(f"  [{display}] {count} ảnh đã cache")
        total_cached += count

    if total_cached == 0:
        print("[LỖI] Không cache được ảnh nào!")
        return False

    cache = {
        'histograms': np.array(all_histograms, dtype=np.float32),
        'entries': entries,
    }

    with open(REFERENCE_CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(REFERENCE_CACHE_PATH) / (1024 * 1024)
    print(f"\n[OK] Đã cache {total_cached} ảnh mẫu → {REFERENCE_CACHE_PATH} ({size_mb:.1f} MB)")
    return True


def load_reference_cache():
    """Tải reference cache từ file pickle. Trả về dict hoặc None."""
    if not os.path.exists(REFERENCE_CACHE_PATH):
        return None
    try:
        with open(REFERENCE_CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        n = len(cache['entries'])
        print(f"[OK] Đã tải reference cache: {n} ảnh mẫu.")
        return cache
    except Exception as e:
        print(f"[WARN] Lỗi tải reference cache: {e}")
        return None


TOP_N_CANDIDATES = 15  # Số ảnh ứng viên lấy ra từ vector search


def find_best_reference_fast(user_gray, reference_cache, kmeans_model):
    """
    Tìm ảnh mẫu phù hợp nhất bằng cache (phiên bản tối ưu).
    1. Tính histogram 1x1 cho ảnh user (bất biến xoay)
    2. Cosine similarity với toàn bộ cache → Top N
    3. Chỉ chạy SIFT matching chi tiết trên Top N
    Trả về dict cùng format như find_best_reference.
    """
    # SIFT trích xuất ảnh user
    kp_user, desc_user = sift_extract(user_gray)
    if desc_user is None or len(desc_user) < 2:
        print("[LỖI] Không trích xuất được SIFT từ ảnh user!")
        return None

    print(f"\n[FAST-SIFT] Ảnh user: {len(kp_user)} keypoints")

    # Tính histogram 1x1 cho ảnh user
    k = kmeans_model.n_clusters
    user_hist = np.zeros(k, dtype=np.float32)
    cluster_labels = kmeans_model.predict(desc_user.astype(np.float32))
    for cid in cluster_labels:
        user_hist[cid] += 1
    norm = np.linalg.norm(user_hist)
    if norm > 0:
        user_hist /= norm

    # Cosine similarity với toàn bộ cache (phép nhân ma trận, cực nhanh)
    cached_hists = reference_cache['histograms']  # (N, K)
    similarities = cached_hists @ user_hist        # (N,)

    # Lấy Top N ứng viên
    top_indices = np.argsort(similarities)[::-1][:TOP_N_CANDIDATES]

    print(f"[FAST-SIFT] Vector search xong → Top {TOP_N_CANDIDATES} ứng viên (từ {len(cached_hists)} ảnh cache)\n")

    # Chạy SIFT matching chi tiết chỉ trên Top N
    best = None
    candidates_meta = []

    for idx in top_indices:
        entry = reference_cache['entries'][idx]
        kp_ref = _deserialize_keypoints(entry['keypoints_data'])
        desc_ref = entry['descriptors']
        class_name = entry['class_name']
        display = DISPLAY_NAMES.get(class_name, class_name)

        # SIFT matching + RANSAC
        good = sift_match(desc_ref, desc_user, kp_ref, kp_user)
        num = len(good)

        if num >= MIN_GOOD_MATCHES:
            _, votes = detect_rotation_angle(kp_ref, kp_user, good)

            print(f"[{display}] {os.path.basename(entry['file_path']):20s} | {num:3d} matches | {votes} votes")

            score = votes

            candidates_meta.append({
                'class_name': class_name,
                'file_name': os.path.basename(entry['file_path']),
                'matches': num,
                'votes': votes,
                'score': score
            })

            if best is None or score > best['score']:
                # Đọc ảnh gốc chỉ cho ảnh tốt nhất (1 lần duy nhất, để hiển thị)
                ref_img = cv2.imread(entry['file_path'])
                if ref_img is None:
                    continue
                ref_img = cv2.resize(ref_img, IMG_SIZE)
                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

                best = {
                    'class_name': class_name,
                    'ref_path': entry['file_path'],
                    'ref_gray': ref_gray,
                    'kp_ref': kp_ref,
                    'desc_ref': desc_ref,
                    'kp_user': kp_user,
                    'desc_user': desc_user,
                    'good_matches': good,
                    'num_matches': num,
                    'score': score,
                }

    if best and best['score'] >= MIN_GOOD_MATCHES - 2:
        print(f"\n  ✓ Ảnh mẫu reference: {os.path.basename(best['ref_path'])} "
              f"(Lớp: {DISPLAY_NAMES.get(best['class_name'], best['class_name'])}) "
              f"— {best['num_matches']} matches (Votes: {best['score']})")

        # Trích lọc Top 3 đại diện (cùng logic như find_best_reference)
        candidates_meta.sort(key=lambda x: x['score'], reverse=True)
        distinct_top = []
        seen_classes = set()
        for c in candidates_meta:
            if c['class_name'] not in seen_classes:
                distinct_top.append(c)
                seen_classes.add(c['class_name'])
            if len(distinct_top) == 3: break

        if len(distinct_top) < 3:
            for c in candidates_meta:
                if c not in distinct_top:
                    distinct_top.append(c)
                if len(distinct_top) == 3: break

        best['top_candidates'] = distinct_top

        return best
    else:
        print(f"\n✗ Không tìm thấy ảnh mẫu phù hợp")
        return None



def find_best_reference(user_gray, dataset_path, classes_to_search):
    # Tìm ảnh mẫu phù hợp nhất trong các lớp được chỉ định
    if isinstance(classes_to_search, str):
        classes_to_search = [classes_to_search]
        
    print(f"\n[SIFT] Sẽ tìm ảnh mẫu tốt nhất trong {len(classes_to_search)} lớp...")

    # SIFT trích xuất ảnh user
    kp_user, desc_user = sift_extract(user_gray)
    if desc_user is None or len(desc_user) < 2:
        print("[LỖI] Không trích xuất được SIFT từ ảnh user!")
        return None

    print(f"[SIFT] Ảnh user: {len(kp_user)} keypoints\n")

    best = None
    candidates_meta = []

    for class_name in classes_to_search:
        ref_images = get_class_images(dataset_path, class_name, NUM_REF_PER_CLASS)
        if not ref_images:
            continue
            
        display = DISPLAY_NAMES.get(class_name, class_name)

        for ref_path in ref_images:
            ref_img = cv2.imread(ref_path)
            if ref_img is None:
                continue
            ref_img = cv2.resize(ref_img, IMG_SIZE)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

            # SIFT match
            kp_ref, desc_ref = sift_extract(ref_gray)
            if desc_ref is None or len(desc_ref) < 2:
                continue

            # Sử dụng phiên bản SIFT match có tích hợp RANSAC
            good = sift_match(desc_ref, desc_user, kp_ref, kp_user)
            num = len(good)

            if num >= MIN_GOOD_MATCHES:
                # KIỂM CHỨNG GÓC KÉP
                _, votes = detect_rotation_angle(kp_ref, kp_user, good)
                
                print(f"[{display}] {os.path.basename(ref_path):20s} | {num:3d} matches | {votes} votes")

                score = votes
                
                # Lưu metadata
                candidates_meta.append({
                    'class_name': class_name,
                    'file_name': os.path.basename(ref_path),
                    'matches': num,
                    'votes': votes,
                    'score': score
                })
                
                if best is None or score > best['score']:
                    best = {
                        'class_name': class_name,
                        'ref_path': ref_path,
                        'ref_gray': ref_gray,
                        'kp_ref': kp_ref,
                        'desc_ref': desc_ref,
                        'kp_user': kp_user,
                        'desc_user': desc_user,
                        'good_matches': good,
                        'num_matches': num,
                        'score': score
                    }

    if best and best['score'] >= MIN_GOOD_MATCHES - 2:
        print(f"\n  ✓ Ảnh mẫu reference: {os.path.basename(best['ref_path'])} "
              f"(Lớp: {DISPLAY_NAMES.get(best['class_name'], best['class_name'])}) "
              f"— {best['num_matches']} matches (Votes: {best['score']})")
              
        # Trích lọc Top 3 đại diện
        candidates_meta.sort(key=lambda x: x['score'], reverse=True)
        distinct_top = []
        seen_classes = set()
        for c in candidates_meta:
            if c['class_name'] not in seen_classes:
                distinct_top.append(c)
                seen_classes.add(c['class_name'])
            if len(distinct_top) == 3: break
            
        # Nếu không đủ 3 lớp khác nhau, chèn thêm các ảnh mạnh thứ nhì của một lớp
        if len(distinct_top) < 3:
            for c in candidates_meta:
                if c not in distinct_top:
                    distinct_top.append(c)
                if len(distinct_top) == 3: break
                
        best['top_candidates'] = distinct_top
        
        return best
    else:
        print(f"\n✗ Không tìm thấy ảnh mẫu phù hợp")
        return None


# DỰ ĐOÁN MỘT ẢNH
def predict_single(gray, kmeans, svm, label_names):
    # Dự đoán lớp cho một ảnh grayscale đã resize
    keypoints, descriptors = extract_sift_descriptors(gray)

    if len(descriptors) == 0:
        return None

    k = kmeans.n_clusters
    
    # Format lại dữ liệu keypoints để truyền vào create_histograms hỗ trợ SPM
    kps_xy = np.array([kp.pt for kp in keypoints])
    data = [{'keypoints': kps_xy, 'descriptors': descriptors, 'shape': gray.shape}]
    
    histogram = create_histograms(data, kmeans, k)

    # Ép chọn lớp có xác suất cao nhất
    pred_proba = svm.predict_proba(histogram)[0]
    pred_idx = np.argmax(pred_proba)
    pred_label = label_names[pred_idx]
    confidence = pred_proba[pred_idx]

    # Xác suất từng lớp
    all_proba = {}
    for i, name in enumerate(label_names):
        all_proba[name] = pred_proba[i]

    if confidence < CONFIDENCE_THRESHOLD:
        display_name = "KHÔNG XÁC ĐỊNH"
    else:
        display_name = DISPLAY_NAMES.get(pred_label, pred_label)

    return {
        'label': pred_label,
        'display_name': display_name,
        'confidence': confidence,
        'all_proba': all_proba,
        'num_keypoints': len(descriptors),
    }


# IN KẾT QUẢ
def print_prediction(result, header=""):
    if header:
        print(f"\n  {header}")
    print(f"Dự đoán:    {result['display_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Keypoints:  {result['num_keypoints']}")
    print(f"Xác suất:")
    for name, prob in result['all_proba'].items():
        display = DISPLAY_NAMES.get(name, name)
        bar = "█" * int(prob * 30)
        print(f"     {display:30s}  {prob:.4f}  {bar}")


def print_comparison(before, after, detected_angle, simulated_angle):
    print(f"\n{'═' * 65}")
    print("KẾT QUẢ SO SÁNH: TRƯỚC vs SAU SIFT PREPROCESSING")
    print(f"{'═' * 65}")

    # Bảng so sánh
    print(f"\n{'Tiêu chí':20s} {'TRƯỚC CHỈNH':>18s}    {'SAU CHỈNH':>18s}    {'THAY ĐỔI':>10s}")
    print(f"{'─' * 20} {'─' * 18}    {'─' * 18}    {'─' * 10}")

    # Dự đoán
    print(f"{'Dự đoán':20s} {before['display_name']:>18s}    {after['display_name']:>18s}")

    # Confidence
    conf_change = after['confidence'] - before['confidence']
    print(f"{'Confidence':20s} {before['confidence']:>17.2%}    {after['confidence']:>17.2%}    {conf_change:>+9.2%}")

    # Keypoints
    kp_change = after['num_keypoints'] - before['num_keypoints']
    print(f"{'Keypoints':20s} {before['num_keypoints']:>18d}    {after['num_keypoints']:>18d}    {kp_change:>+10d}")

    # Góc xoay
    if detected_angle is not None:
        print(f"\nGóc xoay phát hiện bởi SIFT: {detected_angle:+.1f}°")
    if simulated_angle is not None:
        print(f"Góc xoay thực tế (giả lập):  {simulated_angle:+.1f}°")
        error = abs(simulated_angle - (detected_angle or 0))
        print(f"Sai số phát hiện góc:         {error:.1f}°")

    # Kết luận
    print(f"\n{'*' * 65}")
    print("KẾT LUẬN")
    print(f"{'*' * 65}")

    if after['confidence'] > before['confidence']:
        improve = after['confidence'] - before['confidence']
        print(f"""
  SIFT PREPROCESSING CẢI THIỆN KẾT QUẢ NHẬN DẠNG!

  → Sau khi chỉnh xoay bằng SIFT, confidence tăng từ
    {before['confidence']:.2%} lên {after['confidence']:.2%} (+{improve:.2%}).
  → Điều này chứng minh tính bất biến xoay của SIFT giúp
    tiền xử lý ảnh hiệu quả trước khi đưa vào mô hình ML.
  → Mô hình nhận dạng chính xác hơn khi ảnh đã được chỉnh.""")

    elif before['label'] != after['label']:
        print(f"""
  SIFT PREPROCESSING THAY ĐỔI KẾT QUẢ DỰ ĐOÁN!

  → Trước chỉnh: {before['display_name']} ({before['confidence']:.2%})
  → Sau chỉnh:   {after['display_name']} ({after['confidence']:.2%})
  → Ảnh sau khi chỉnh xoay cho kết quả nhận dạng khác.
  → Cần kiểm tra thêm để xác định kết quả nào chính xác.""")

    else:
        print(f"""
  KẾT QUẢ TƯƠNG ĐƯƠNG

  → Ảnh có thể đã ở hướng chuẩn, SIFT preprocessing không
    thay đổi đáng kể kết quả nhận dạng.
  → Confidence: {before['confidence']:.2%} → {after['confidence']:.2%}""")

    print(f"{'*' * 65}")

def save_pipeline_results(user_gray, corrected_gray, ref_info,
                          result_before, result_after, detected_angle):
    # Lưu ảnh kết quả pipeline vào thư mục results_pipeline/.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Ảnh so sánh 3 panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(ref_info['ref_gray'], cmap='gray')
    display_ref = DISPLAY_NAMES.get(ref_info['class_name'], ref_info['class_name'])
    axes[0].set_title(
        f'ẢNH MẪU (Reference)\n{display_ref}',
        fontsize=11, fontweight='bold', color='green'
    )
    axes[0].axis('off')

    axes[1].imshow(user_gray, cmap='gray')
    axes[1].set_title(
        f'ẢNH USER (TRƯỚC CHỈNH)\n{result_before["display_name"]}\n'
        f'Confidence: {result_before["confidence"]:.2%}',
        fontsize=10, fontweight='bold', color='red'
    )
    axes[1].axis('off')

    axes[2].imshow(corrected_gray, cmap='gray')
    axes[2].set_title(
        f'SAU SIFT CHỈNH ({detected_angle:+.1f}°)\n{result_after["display_name"]}\n'
        f'Confidence: {result_after["confidence"]:.2%}',
        fontsize=10, fontweight='bold', color='blue'
    )
    axes[2].axis('off')

    fig.suptitle(
        'PIPELINE: SIFT PREPROCESSING + NHẬN DẠNG ĐỊA DANH',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    path_comp = os.path.join(RESULTS_DIR, 'pipeline_comparison.png')
    plt.savefig(path_comp, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

    # SIFT matches visualization
    display_matches = sorted(
        ref_info['good_matches'], key=lambda x: x.distance
    )[:50]
    match_img = cv2.drawMatches(
        ref_info['ref_gray'], ref_info['kp_ref'],
        user_gray, ref_info['kp_user'],
        display_matches, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    path_match = os.path.join(RESULTS_DIR, 'pipeline_matches.png')
    cv2.imwrite(path_match, match_img)

    # Ảnh đã chỉnh
    path_corrected = os.path.join(RESULTS_DIR, 'pipeline_corrected.png')
    cv2.imwrite(path_corrected, corrected_gray)

    # Ảnh user gốc
    path_user = os.path.join(RESULTS_DIR, 'pipeline_user_input.png')
    cv2.imwrite(path_user, user_gray)

    # In danh sách file đã lưu
    print(f"\nKết quả đã lưu tại: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith('pipeline_'):
            size = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1024
            print(f"{f:40s} ({size:.1f} KB)")


# PIPELINE CHÍNH
def full_pipeline(image_path, simulate_angle=None):
    print("\n" + "█" * 65)
    print("PIPELINE: SIFT PREPROCESSING + NHẬN DẠNG ẢNH ĐỊA DANH")
    print("█" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # BƯỚC 0: Tải mô hình
    print("\n[0] Tải mô hình đã huấn luyện...")
    kmeans, svm, label_names = load_models()
    if kmeans is None:
        print("[LỖI] Chưa có mô hình! Vui lòng huấn luyện trước (option 1).")
        return

    # BƯỚC 1: Tải ảnh user
    print(f"\n[1] Tải ảnh: {image_path}")
    if not os.path.isfile(image_path):
        print(f"[LỖI] File không tồn tại: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[LỖI] Không thể đọc ảnh: {image_path}")
        return

    img = cv2.resize(img, IMG_SIZE)
    user_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_gray = user_gray.copy()  # Lưu bản gốc (dùng cho SSIM nếu giả lập)
    print(f"OK — {user_gray.shape[1]}x{user_gray.shape[0]}")

    # Giả lập xoay (nếu có)
    if simulate_angle is not None and simulate_angle != 0:
        print(f"\n[*] GIẢ LẬP: Xoay ảnh {simulate_angle}° "
              f"(mô phỏng ảnh user bị xoay/nghiêng)")
        user_gray = rotate_image(user_gray, simulate_angle)

    # BƯỚC 2: Dự đoán TRƯỚC khi chỉnh
    print(f"\n{'─' * 65}")
    print("BƯỚC 2: DỰ ĐOÁN TRƯỚC KHI CHỈNH (ẢNH GỐC CỦA USER)")
    print(f"{'─' * 65}")

    result_before = predict_single(user_gray, kmeans, svm, label_names)
    if result_before is None:
        print("[LỖI] Không trích xuất được đặc trưng SIFT từ ảnh!")
        return

    print_prediction(result_before)

    # BƯỚC 3: SIFT matching → Tìm ảnh mẫu + phát hiện góc xoay
    print(f"\n{'─' * 65}")
    print("BƯỚC 3: SIFT MATCHING → PHÁT HIỆN GÓC XOAY")
    print(f"{'─' * 65}")

    # Truyền tất cả các lớp vào hàm SIFT matching thay vì chỉ lớp có xác suất cao nhất
    classes_to_search = list(result_before['all_proba'].keys())
    
    # Thử dùng reference cache để tìm kiếm nhanh
    ref_cache = load_reference_cache()
    if ref_cache is not None:
        print("[INFO] Đang dùng Reference Cache (BoVW 1x1 Vector Search) tìm ảnh mẫu nhanh...")
        ref_info = find_best_reference_fast(user_gray, ref_cache, kmeans)
    else:
        print("[WARN] Chưa tìm thấy Reference Cache. Sử dụng duyệt tuần tự (Chậm)...")
        ref_info = find_best_reference(user_gray, DATASET_PATH, classes_to_search)

    if ref_info is None:
        print("\n→ Không tìm được ảnh mẫu phù hợp.")
        print("→ Giữ nguyên kết quả dự đoán ban đầu.")
        print_comparison(result_before, result_before, None, simulate_angle)
        return

    # Phát hiện góc xoay
    detected_angle, votes = detect_rotation_angle(
        ref_info['kp_ref'], ref_info['kp_user'], ref_info['good_matches']
    )

    if detected_angle is None or votes < 5 or (abs(detected_angle) > 135 and votes < 8):
        print(f"\nRotation không đáng tin (góc: {detected_angle or 0:.1f}°, votes: {votes}) → BỎ QUA XOAY ẢNH.")
        print_comparison(result_before, result_before, None, simulate_angle)
        
        # Lưu kết quả kể cả khi bỏ qua xoay
        save_pipeline_results(
            user_gray, user_gray, ref_info,
            result_before, result_before, detected_angle or 0.0
        )
        return

    print(f"\nGóc xoay phát hiện: {detected_angle:+.1f}°")
    print(f"Votes: {votes}/{len(ref_info['good_matches'])} good matches")

    # BƯỚC 4: Chỉnh xoay ảnh
    print(f"\n{'─' * 65}")
    print("BƯỚC 4: CHỈNH XOAY ẢNH")
    print(f"{'─' * 65}")

    correction = -detected_angle
    corrected_gray = rotate_image(user_gray, correction)
    print(f"↻ Xoay ảnh {correction:+.1f}° (ngược lại góc phát hiện {detected_angle:+.1f}°)")

    # SSIM (so sánh với ảnh gốc trước khi giả lập xoay)
    if simulate_angle is not None and simulate_angle != 0:
        ssim_before = compute_ssim(original_gray, user_gray)
        ssim_after = compute_ssim(original_gray, corrected_gray)
        print(f"SSIM (so với ảnh gốc) trước chỉnh: {ssim_before:.4f}")
        print(f"SSIM (so với ảnh gốc) sau chỉnh: {ssim_after:.4f} "
              f"({ssim_after - ssim_before:+.4f})")

    # BƯỚC 5: Dự đoán SAU khi chỉnh
    print(f"\n{'─' * 65}")
    print("BƯỚC 5: DỰ ĐOÁN SAU KHI CHỈNH SIFT")
    print(f"{'─' * 65}")

    result_after = predict_single(corrected_gray, kmeans, svm, label_names)
    if result_after is None:
        print("[LỖI] Không trích xuất được đặc trưng sau chỉnh!")
        result_after = result_before
    else:
        print_prediction(result_after)

    # BƯỚC 6: So sánh & Kết luận
    print_comparison(result_before, result_after, detected_angle, simulate_angle)

    # Lưu ảnh kết quả
    save_pipeline_results(
        user_gray, corrected_gray, ref_info,
        result_before, result_after, detected_angle
    )


# MENU
def main():
    while True:
        print("\n" + "=" * 65)
        print("SIFT LANDMARK PIPELINE — TỰ XOAY CHỈNH VÀ NHẬN DIỆN")
        print("=" * 65)
        print(f"""
  Mô tả:
     Dùng SIFT để chỉnh ảnh (xoay, nghiêng) TRƯỚC KHI đưa cho
     mô hình ML nhận dạng → cải thiện độ chính xác.

  Cấu hình:
     • Dataset:     {DATASET_PATH}/
     • Models:      models/
     • Kết quả:     {RESULTS_DIR}/
        """)
        print("[1] Chạy Pipeline nhận diện ảnh tự sửa góc (Nhập đường dẫn thủ công)")
        print("[0] Thoát")
        print("-" * 65)

        choice = input("Lựa chọn (0-1): ").strip()

        if choice == '1':
            image_path = input("\nNhập đường dẫn ảnh: ").strip().strip('"').strip("'")
            if not image_path:
                print("[WARN] Đường dẫn không hợp lệ!")
                continue

            # Hỏi có muốn giả lập xoay không
            sim = input("Giả lập xoay? Nhập góc lệch (hoặc Enter để bỏ qua): ").strip()
            simulate_angle = None
            if sim:
                try:
                    simulate_angle = float(sim)
                except ValueError:
                    print("[WARN] Góc không hợp lệ, bỏ qua giả lập xoay.")

            full_pipeline(image_path, simulate_angle)

        elif choice == '0':
            print("\nTạm biệt!")
            break

        else:
            print("[WARN] Lựa chọn không hợp lệ! Vui lòng chọn 0-1.")


if __name__ == '__main__':
    main()
