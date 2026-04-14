import os
import sys
import cv2
import numpy as np
import threading
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from tkinter import filedialog, messagebox

from landmark_sift_bovw import load_models, IMG_SIZE, DISPLAY_NAMES
from sift_landmark_pipeline import (
    find_best_reference, detect_rotation_angle, 
    rotate_image, predict_single, DATASET_PATH
)

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

matplotlib.use('Agg')

class LandmarkDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SIFT Landmark Recognition Dashboard")
        self.geometry("1400x850")
        
        self.kmeans = None
        self.svm = None
        self.label_names = None
        self.image_path = None
        self.current_user_img = None
        self.save_dir = os.path.abspath('results_pipeline')
        
        self.grid_columnconfigure(0, weight=0, minsize=400) 
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.setup_left_panel()
        self.setup_right_panel()
        
        self.log_msg("Hệ thống khởi động. Đang tải mô hình...")
        threading.Thread(target=self.init_models, daemon=True).start()

    def init_models(self):
        kmeans_model, svm_model, labels = load_models()
        if kmeans_model is not None:
            self.kmeans = kmeans_model
            self.svm = svm_model
            self.label_names = labels
            self.log_msg("Tải mô hình thành công! Đã sẵn sàng nhận diện.")
        else:
            self.log_msg("LỖI: Không tìm thấy file mô hình. Hãy nhấn Train Dữ Liệu.")

    def setup_left_panel(self):
        self.left_frame = ctk.CTkFrame(self, corner_radius=10, width=400)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        self.left_frame.pack_propagate(False)
        
        lbl_title = ctk.CTkLabel(self.left_frame, text="BẢNG ĐIỀU KHIỂN", font=ctk.CTkFont(size=20, weight="bold"))
        lbl_title.pack(pady=(20, 10))
        
        self.tabview = ctk.CTkTabview(self.left_frame)
        self.tabview.pack(padx=10, pady=(0, 10), fill="both", expand=True)
        
        self.tab_main = self.tabview.add("Nhận Diện")
        self.tab_log = self.tabview.add("Log")
        
        self.setup_tab_main()
        self.setup_tab_log()

    def setup_tab_main(self):
        # 1. Chọn ảnh
        frame_top = ctk.CTkFrame(self.tab_main, fg_color="transparent")
        frame_top.pack(fill="x", pady=5)
        self.btn_select = ctk.CTkButton(frame_top, text="Chọn ảnh", command=self.select_image, height=35)
        self.btn_select.pack(fill="x", padx=5)
        
        path_frame = ctk.CTkFrame(self.tab_main, fg_color="transparent")
        path_frame.pack(fill="x", pady=(2, 10))
        self.lbl_path = ctk.CTkLabel(path_frame, text="Chưa chọn ảnh...", text_color="gray", anchor="w")
        self.lbl_path.pack(side="left", padx=5, fill="x", expand=True)
        self.btn_copy_path = ctk.CTkButton(path_frame, text="❐", width=40, state="disabled", fg_color="#34495e", hover_color="#2c3e50", font=ctk.CTkFont(size=18),
                                           command=lambda: self.copy_to_clipboard(self.image_path))
        self.btn_copy_path.pack(side="right", padx=5)
        
        # 2. Giả lập xoay ảnh
        rot_frame = ctk.CTkFrame(self.tab_main)
        rot_frame.pack(fill="x", pady=5, padx=5, ipadx=4, ipady=4)
        lbl_rot = ctk.CTkLabel(rot_frame, text="Giả lập xoay ảnh (°):", font=ctk.CTkFont(weight="bold"))
        lbl_rot.pack(pady=(5,0))
        
        slider_inner = ctk.CTkFrame(rot_frame, fg_color="transparent")
        slider_inner.pack(fill="x", pady=2)
        self.slider_rot = ctk.CTkSlider(slider_inner, from_=-180, to=180, command=self.on_slider_change)
        self.slider_rot.set(0)
        self.slider_rot.pack(side="left", fill="x", expand=True, padx=(5,0))
        
        btn_reset_rot = ctk.CTkButton(slider_inner, text="↺ 0°", width=40, fg_color="#e74c3c", hover_color="#c0392b", command=self.reset_rotation)
        btn_reset_rot.pack(side="right", padx=(5, 10))
        
        self.lbl_angle_val = ctk.CTkLabel(rot_frame, text="0°")
        self.lbl_angle_val.pack()
        
        # 3. BẮT ĐẦU NHẬN DIỆN
        self.btn_run = ctk.CTkButton(self.tab_main, text="NHẬN DIỆN", command=self.run_pipeline_thread,
                                     height=50, fg_color="#2ecc71", hover_color="#27ae60", font=ctk.CTkFont(size=15, weight="bold"))
        self.btn_run.pack(pady=15, fill="x", padx=5)

        # 4. Quản lý Dataset & Train
        ds_frame = ctk.CTkFrame(self.tab_main)
        ds_frame.pack(fill="x", pady=5, padx=5, ipadx=4, ipady=4)
        
        abs_ds = os.path.abspath(DATASET_PATH)
        lbl_ds = ctk.CTkLabel(ds_frame, text="Vị trí lưu Dataset:", font=ctk.CTkFont(weight="bold"))
        lbl_ds.pack(anchor="w", padx=5, pady=(2,0))
        
        ds_path_inner = ctk.CTkFrame(ds_frame, fg_color="transparent")
        ds_path_inner.pack(fill="x", pady=2)
        
        lbl_ds_path = ctk.CTkLabel(ds_path_inner, text=self.truncate_text(abs_ds, 35), anchor="w", text_color="gray")
        lbl_ds_path.pack(side="left", padx=5, fill="x", expand=True)
        
        btn_copy_ds = ctk.CTkButton(ds_path_inner, text="❐", width=40, fg_color="#34495e", hover_color="#2c3e50", font=ctk.CTkFont(size=18), command=lambda: self.copy_to_clipboard(abs_ds))
        btn_copy_ds.pack(side="right", padx=(2, 10))
        
        # Thống kê thư mục và file
        ds_info_text = ""
        if os.path.exists(abs_ds):
            subdirs = sorted([d for d in os.listdir(abs_ds) if os.path.isdir(os.path.join(abs_ds, d))])
            total_files = 0
            details = []
            for d in subdirs:
                count = len([f for f in os.listdir(os.path.join(abs_ds, d)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                total_files += count
                details.append(f"{d} ({count})")
            ds_info_text = f"Tổng số ảnh: {total_files} files / {len(subdirs)} lớp\nChi tiết: {', '.join(details)}"
        else:
            ds_info_text = "Không tìm thấy dữ liệu gốc."
            
        lbl_ds_info = ctk.CTkLabel(ds_frame, text=ds_info_text, wraplength=333, justify="left", text_color="#34495e", font=ctk.CTkFont(size=12))
        lbl_ds_info.pack(anchor="w", padx=15, pady=(2, 5))
        
        # Nút Train
        train_btn_frame = ctk.CTkFrame(ds_frame, fg_color="transparent")
        train_btn_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        self.btn_train = ctk.CTkButton(train_btn_frame, text="TRAIN", fg_color="#8e44ad", hover_color="#9b59b6", font=ctk.CTkFont(weight="bold"), height=35, command=self.run_training_thread)
        self.btn_train.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        self.btn_cm = ctk.CTkButton(train_btn_frame, text="THỐNG KÊ", fg_color="#e74c3c", hover_color="#c0392b", font=ctk.CTkFont(weight="bold"), height=35, command=self.show_confusion_matrix)
        self.btn_cm.pack(side="left", fill="x", expand=True, padx=(2, 0))

        # 5. Lưu kết quả
        save_frame = ctk.CTkFrame(self.tab_main)
        save_frame.pack(fill="x", pady=10, padx=5, ipadx=4, ipady=4)
        
        lbl_save = ctk.CTkLabel(save_frame, text="Vị trí lưu kết quả:", font=ctk.CTkFont(weight="bold"))
        lbl_save.pack(anchor="w", padx=5)
        
        save_path_inner = ctk.CTkFrame(save_frame, fg_color="transparent")
        save_path_inner.pack(fill="x", pady=2)
        
        self.lbl_save_dir = ctk.CTkLabel(save_path_inner, text=self.truncate_text(self.save_dir, 20), anchor="w", text_color="gray")
        self.lbl_save_dir.pack(side="left", padx=5, fill="x", expand=True)
        
        btn_change_save = ctk.CTkButton(save_path_inner, text="⚙", width=40, fg_color="#2980b9", hover_color="#3498db", font=ctk.CTkFont(size=18), command=self.change_save_dir)
        btn_change_save.pack(side="right", padx=(5, 10))
        
        btn_copy_save = ctk.CTkButton(save_path_inner, text="❐", width=40, fg_color="#34495e", hover_color="#2c3e50", font=ctk.CTkFont(size=18), command=lambda: self.copy_to_clipboard(self.save_dir))
        btn_copy_save.pack(side="right", padx=2)

    def setup_tab_log(self):
        # Tab 2
        lbl_log = ctk.CTkLabel(self.tab_log, text="Terminal Log:", font=ctk.CTkFont(weight="bold"))
        lbl_log.pack(pady=(5, 5), anchor="w", padx=5)
        
        self.log_box = ctk.CTkTextbox(self.tab_log, font=ctk.CTkFont(family="Consolas", size=11), fg_color="#e0e0e0", text_color="black")
        self.log_box.pack(pady=5, fill="both", expand=True)
        self.log_box.configure(state="disabled")

    def setup_right_panel(self):
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(2, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)
        
        self.col1_frame = self.create_display_column(self.right_frame, "TRƯỚC CHỈNH SỬA", 0)
        self.col2_frame = self.create_display_column(self.right_frame, "SIFT MATCHING", 1)
        self.col3_frame = self.create_display_column(self.right_frame, "SAU CHỈNH SỬA", 2)
        
    def create_display_column(self, parent, title, col_index):
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=0, column=col_index, padx=5, pady=0, sticky="nsew")
        
        lbl = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold"), text_color="#2980b9")
        lbl.pack(pady=15)
        
        img_lbl = ctk.CTkLabel(frame, text="", bg_color="#d6d6d6", width=256, height=256)
        img_lbl.pack(pady=10)
        
        pred_lbl = ctk.CTkLabel(frame, text="...", font=ctk.CTkFont(size=15, weight="bold"), text_color="black")
        pred_lbl.pack(pady=5)
        
        extra_frame = ctk.CTkFrame(frame, fg_color="transparent")
        extra_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        setattr(self, f'col{col_index+1}_img', img_lbl)
        setattr(self, f'col{col_index+1}_pred', pred_lbl)
        setattr(self, f'col{col_index+1}_extra', extra_frame)
        
        return frame

    def log_msg(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        
    def copy_to_clipboard(self, text):
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
            self.log_msg(f"Đã Copy: {text}")
            
    def truncate_text(self, text, limit=50):
        if len(text) > limit:
            return "..." + text[-(limit-3):]
        return text

    def change_save_dir(self):
        new_dir = filedialog.askdirectory(initialdir=self.save_dir, title="Chọn thư mục lưu kết quả")
        if new_dir:
            self.save_dir = os.path.abspath(new_dir)
            self.lbl_save_dir.configure(text=self.truncate_text(self.save_dir, 20))
            self.log_msg(f"Đã đổi thư mục xuất ảnh sang: {self.save_dir}")

    def reset_rotation(self):
        self.slider_rot.set(0)
        self.on_slider_change(0)
        
    def on_slider_change(self, value):
        angle = int(value)
        self.lbl_angle_val.configure(text=f"{angle}°")
        if self.image_path and os.path.exists(self.image_path):
            img = cv2.imread(self.image_path)
            if img is not None:
                self.raw_user_img = img.copy()
                if angle != 0:
                    self.raw_user_img = rotate_image(self.raw_user_img, angle)
                    
                img_resized = cv2.resize(img, IMG_SIZE)
                if angle != 0:
                    img_resized = rotate_image(img_resized, angle)
                    
                self.current_user_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                raw_rgb = cv2.cvtColor(self.raw_user_img, cv2.COLOR_BGR2RGB)
                self.set_image(self.col1_img, self.current_user_img, full_res_img=raw_rgb)

    def show_confusion_matrix(self):
        try:
            from landmark_sift_bovw import MODEL_DIR
            cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
            if not os.path.exists(cm_path):
                self.log_msg("LỖI: Không tìm thấy ảnh Ma trận nhầm lẫn. Bác hãy thử Train dữ liệu lại 1 lần nhé.")
                return
            self.open_saved_image(cm_path, "Ma trận nhầm lẫn (Confusion Matrix)")
        except Exception as e:
            self.log_msg(f"LỖI gọi Ma trận nhầm lẫn: {e}")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.image_path = path
            self.btn_copy_path.configure(state="normal")
            
            self.slider_rot.set(0)
            self.lbl_angle_val.configure(text="0°")
            self.clear_ui()
            
            img = cv2.imread(self.image_path)
            if img is not None:
                h_orig, w_orig = img.shape[:2]
                self.lbl_path.configure(text=f"{self.truncate_text(path, 30)} ({w_orig}x{h_orig}px)")
                
                self.raw_user_img = img.copy()
                
                img_resized = cv2.resize(img, IMG_SIZE)
                self.current_user_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                raw_rgb = cv2.cvtColor(self.raw_user_img, cv2.COLOR_BGR2RGB)
                self.set_image(self.col1_img, self.current_user_img, full_res_img=raw_rgb)
                self.log_msg(f"Đã mở ảnh (Gốc: {w_orig}x{h_orig}): {path}")

    def set_image(self, label_widget, cv2_rgb_img, target_size=(256, 256), full_res_img=None):
        if cv2_rgb_img is None:
            label_widget.configure(image="", text="[Trống]")
            if hasattr(label_widget, 'image'):
                label_widget.image = None
            label_widget.unbind("<Button-1>")
            label_widget.configure(cursor="arrow")
            return
            
        h, w = cv2_rgb_img.shape[:2]
        ratio = min(target_size[0]/w, target_size[1]/h)
        new_size = (int(w*ratio), int(h*ratio))
        img_resized = cv2.resize(cv2_rgb_img, new_size)
        
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(img_resized)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=new_size)
        label_widget.configure(image=ctk_img, text="")
        label_widget.image = ctk_img
        
        try:
            label_widget.unbind("<Button-1>")
        except:
            pass
        label_widget.configure(cursor="")

    def open_saved_image(self, path, title):
        if not os.path.exists(path):
            self.log_msg(f"Không tìm thấy file: {path}")
            return
        img = cv2.imread(path)
        if img is not None:
            self.open_full_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), title=title)

    def open_full_image(self, img_arr, title="Chi tiết"):
        if img_arr is None: return
        
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        
        h, w = img_arr.shape[:2]
        
        max_w = int(screen_w * 0.85)
        max_h = int(screen_h * 0.85)
        
        scale = 1.0
        if w > max_w or h > max_h:
            scale = min(max_w / float(w), max_h / float(h))
            w, h = int(w * scale), int(h * scale)
            img_arr = cv2.resize(img_arr, (w, h), interpolation=cv2.INTER_AREA)

        if len(img_arr.shape) == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
            
        from PIL import Image
        pil_img = Image.fromarray(img_arr)
        
        top = ctk.CTkToplevel(self)
        top.title(title)
        top.geometry(f"{w+40}x{h+40}")
        top.transient(self)
        top.focus()
        
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(w, h))
        lbl = ctk.CTkLabel(top, text="", image=ctk_img)
        lbl.pack(expand=True, fill="both", padx=20, pady=20)

    def plot_bar_chart(self, frame_widget, proba_dict, is_percentage=True):
        for widget in frame_widget.winfo_children():
            widget.destroy()
            
        top3 = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        labels = [DISPLAY_NAMES.get(k, k) for k, v in top3]
        values = [v for k, v in top3]
        
        bg_color = '#ebebeb'
        fig, ax = plt.subplots(figsize=(3.5, 2.5), facecolor=bg_color) 
        ax.set_facecolor(bg_color)
        
        main_color = '#e74c3c' if is_percentage else '#8e44ad'
        bars = ax.barh(labels, values, color=[main_color if i==0 else '#7f8c8d' for i in range(len(values))])
        
        if is_percentage:
            ax.set_xlim(0, 1.0)
        else:
            max_val = max(values) if values else 1
            ax.set_xlim(0, max_val * 1.25)
            
        ax.invert_yaxis()
        ax.tick_params(colors='black', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(bg_color)
            
        for bar, val in zip(bars, values):
            if is_percentage:
                text_val = f'{val:.1%}'
                offset = 0.02
            else:
                text_val = f'{int(val)} votes'
                offset = max_val * 0.05
                
            ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2, text_val, 
                    va='center', color='black', fontsize=8, fontweight='bold')
            
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame_widget)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # Hàm tính SSIM
    @staticmethod
    def compute_ssim(img1, img2):
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        i1 = img1.astype(np.float64)
        i2 = img2.astype(np.float64)
        mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
        s1 = cv2.GaussianBlur(i1 ** 2, (11, 11), 1.5) - mu1 ** 2
        s2 = cv2.GaussianBlur(i2 ** 2, (11, 11), 1.5) - mu2 ** 2
        s12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
        return float(ssim_map.mean())

    # Hàm vẽ ảnh match
    @staticmethod
    def draw_match_image(img1, kp1, img2, kp2, matches, title, path):
        top_matches = sorted(matches, key=lambda x: x.distance)[:80]
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, top_matches, None,
                                    matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        h, w = match_img.shape[:2]
        cv2.rectangle(match_img, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.putText(match_img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(path, cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR) if len(match_img.shape)==3 else match_img)

    @staticmethod
    def save_summary_bar_chart(ssim_before, ssim_after, matches_before, matches_after, path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Biểu đồ SSIM
        ax1.bar([0], [ssim_before], width=0.4, label='Trước chỉnh xoay', color='#EF5350', edgecolor='#C62828')
        ax1.bar([0.5], [ssim_after], width=0.4, label='Sau chỉnh xoay', color='#42A5F5', edgecolor='#1565C0')
        ax1.annotate(f"{ssim_before:.4f}", (0, ssim_before), xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold', color='#C62828')
        ax1.annotate(f"{ssim_after:.4f}", (0.5, ssim_after), xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold', color='#1565C0')
        ax1.set_xticks([0.25])
        ax1.set_xticklabels(['SSIM Tương đồng'])
        ax1.set_ylabel('SSIM Score')
        ax1.set_title('ĐỘ TƯƠNG ĐỒNG CẤU TRÚC (SSIM)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(ssim_after, ssim_before) * 1.3)
        
        # 2. Biểu đồ Matches
        ax2.bar([0], [matches_before], width=0.4, label='Trước chỉnh xoay', color='#66BB6A', edgecolor='#2E7D32')
        ax2.bar([0.5], [matches_after], width=0.4, label='Sau chỉnh xoay', color='#FFCA28', edgecolor='#F9A825')
        ax2.annotate(f"{matches_before}", (0, matches_before), xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
        ax2.annotate(f"{matches_after}", (0.5, matches_after), xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')
        ax2.set_xticks([0.25])
        ax2.set_xticklabels(['Số lượng Khớp SIFT'])
        ax2.set_ylabel('Matches')
        ax2.set_title('ĐIỂM ĐỒNG THUẬN (MATCHES)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(matches_before, matches_after) * 1.3)
        
        fig.suptitle('SỰ CẢI THIỆN SAU KHI CHỈNH SIFT', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def clear_ui(self):
        for col in [1, 2, 3]:
            getattr(self, f'col{col}_pred').configure(text="...", text_color="black")
            for w in getattr(self, f'col{col}_extra').winfo_children():
                w.destroy()
        self.col2_img.configure(image=None, text=" (Trống) ")
        self.col3_img.configure(image=None, text=" (Trống) ")

    # TRAINING
    def run_training_thread(self):
        ans = messagebox.askyesno("Thông báo", "Quá trình Train sẽ mất nhiều thời gian.\nBạn có muốn bắt đầu không?")
        if not ans: return
        
        self.btn_train.configure(state="disabled", text="⏳ Training...")
        self.btn_run.configure(state="disabled")
        threading.Thread(target=self.execute_training, daemon=True).start()
        
    def execute_training(self):
        from landmark_sift_bovw import train_pipeline
        self.log_msg("==============================")
        self.log_msg("ĐANG BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN")
        self.log_msg("Xem chi tiết trích xuất đặc trưng SIFT ở màn hình Terminal đen.")
        try:
            success = train_pipeline()
            if success:
                self.log_msg("QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT!")
                self.log_msg("Đang tải nạp tự động mô hình mới vào GUI...")
                self.init_models()
                messagebox.showinfo("Thành công", "Huấn luyện dữ liệu thành công! Ứng dụng đã reload AI model mới nhất.")
            else:
                self.log_msg("Huấn luyện gặp lỗi, hãy xem log chi tiết.")
        except Exception as e:
            self.log_msg(f"LỖI: {e}")
        finally:
            self.btn_train.configure(state="normal", text="TRAIN")
            self.btn_run.configure(state="normal")

    # CHẠY PIPELINE
    def run_pipeline_thread(self):
        if not self.image_path:
            messagebox.showwarning("Thiếu ảnh", "Vui lòng chọn ảnh trước!")
            return
        if self.kmeans is None:
            messagebox.showwarning("Thiếu mô hình", "Mô hình đang tải hoặc chưa Train!")
            return
            
        self.btn_run.configure(state="disabled", text="⏳ ĐANG XỬ LÝ...")
        threading.Thread(target=self.execute_pipeline, daemon=True).start()

    def execute_pipeline(self):
        try:
            self.log_msg("-" * 30)
            self.log_msg("Đang rút trích SPM 1x1+2x2...")
            
            user_gray = cv2.cvtColor(self.current_user_img, cv2.COLOR_RGB2GRAY)
            res_before = predict_single(user_gray, self.kmeans, self.svm, self.label_names)
            if not res_before:
                self.log_msg("Lỗi: Không trích được SIFT gốc.")
                return
                
            self.col1_pred.configure(text=f"{res_before['display_name']}\n({res_before['confidence']:.1%})", text_color="#c0392b")
            self.plot_bar_chart(self.col1_extra, res_before['all_proba'])
            
            self.log_msg("Đang quét tham chiếu SIFT đối chiếu chéo lớp...")
            classes_to_search = list(res_before['all_proba'].keys())
            ref_info = find_best_reference(user_gray, DATASET_PATH, classes_to_search)
            
            os.makedirs(self.save_dir, exist_ok=True)
            path_user_out = os.path.join(self.save_dir, 'gui_out_1_before.jpg')
            path_match_out = os.path.join(self.save_dir, 'gui_out_2_matches.jpg')
            path_corrected_out = os.path.join(self.save_dir, 'gui_out_3_after.jpg')
            
            cv2.imwrite(path_user_out, user_gray)
            
            if ref_info is None:
                self.log_msg("LỖI: SIFT Không tìm thấy ảnh mẫu phù hợp (Thiếu điểm đồng thuận).")
                self.col2_pred.configure(text="KHÔNG TÌM THẤY ẢNH MẪU", text_color="#e74c3c")
                
                # Xóa ảnh SIFT matching nếu không có
                self.col2_img.configure(image="", text="[Trống]")
                if hasattr(self.col2_img, 'image'):
                    self.col2_img.image = None
                    
                self.after_pipeline_end(user_gray, res_before, path_corrected_out)
                return
                
            self.log_msg(f"Ảnh mẫu: {ref_info['class_name']} ({ref_info['num_matches']} matches)")
            
            # ẢNH KEYPOINTS
            try:
                path_kp_user = os.path.join(self.save_dir, 'keypoints.png')
                
                img_kp_user = cv2.drawKeypoints(user_gray, ref_info['kp_user'], None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                
                cv2.imwrite(path_kp_user, cv2.cvtColor(img_kp_user, cv2.COLOR_RGB2BGR) if len(img_kp_user.shape)==3 else img_kp_user)
            except Exception as e:
                self.log_msg(f"Lỗi khi xuất ảnh Keypoints: {e}")
            
            # TOP 3 SIFT MATCHING
            top_cands = ref_info.get('top_candidates', [])
            if top_cands:
                sift_votes_dict = {cand['class_name']: cand['score'] for cand in top_cands}
                self.plot_bar_chart(self.col2_extra, sift_votes_dict, is_percentage=False)
            else:
                for w in self.col2_extra.winfo_children(): w.destroy()
            
            self.log_msg("Đang kiểm chứng hình học RANSAC/Voting...")
            angle, votes = detect_rotation_angle(ref_info['kp_ref'], ref_info['kp_user'], ref_info['good_matches'])
            
            match_img = cv2.drawMatches(
                cv2.cvtColor(ref_info['ref_gray'], cv2.COLOR_GRAY2RGB), ref_info['kp_ref'],
                self.current_user_img, ref_info['kp_user'],
                sorted(ref_info['good_matches'], key=lambda x: x.distance)[:30], None,
                matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            self.set_image(self.col2_img, match_img, target_size=(350, 256))
            cv2.imwrite(path_match_out, cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR))
            
            if angle is None or votes < 5 or (abs(angle) > 135 and votes < 8):
                self.log_msg(f"GÓC XOAY RỦI RO LỚN ({angle}°, {votes} votes) → TỪ CHỐI XOAY ĐỂ BẢO CẢNH!")
                self.col2_pred.configure(text=f"TỪ CHỐI XOAY\n(Dữ liệu quá yếu: {votes} votes)", text_color="#c0392b")
                
                self.after_pipeline_end(user_gray, res_before, path_corrected_out)
                return
                
            self.log_msg(f"Hợp lệ! Góc xoay bù: {-angle:+.1f}°")
            self.col2_pred.configure(text=f"CHỈNH XOAY: {-angle:+.1f}°\n({votes} votes)", text_color="#27ae60")
            
            corrected_gray = rotate_image(user_gray, -angle)
            res_after = predict_single(corrected_gray, self.kmeans, self.svm, self.label_names)
            
            try:
                from sift_landmark_pipeline import sift_match
                
                path_match_b = os.path.join(self.save_dir, 'match_before.png')
                path_match_a = os.path.join(self.save_dir, 'match_after.png')
                path_comp = os.path.join(self.save_dir, 'summary_chart.png')
                
                # 1. Vẽ Match Before
                self.draw_match_image(ref_info['ref_gray'], ref_info['kp_ref'], user_gray, ref_info['kp_user'], ref_info['good_matches'], 
                                 f"TRUOC CHINH | Matches: {len(ref_info['good_matches'])}", path_match_b)
                
                # 2. Vẽ Match After
                sift_eval = cv2.SIFT_create()
                kp_corr, desc_corr = sift_eval.detectAndCompute(corrected_gray, None)
                good_after = sift_match(ref_info['desc_ref'], desc_corr)
                
                self.draw_match_image(ref_info['ref_gray'], ref_info['kp_ref'], corrected_gray, kp_corr, good_after, 
                                 f"SAU CHINH | {angle:+.1f} deg | Matches: {len(good_after)}", path_match_a)
                
                # 3. Vẽ biểu đồ
                ssim_before = self.compute_ssim(ref_info['ref_gray'], user_gray)
                ssim_after = self.compute_ssim(ref_info['ref_gray'], corrected_gray)
                self.save_summary_bar_chart(ssim_before, ssim_after, len(ref_info['good_matches']), len(good_after), path_comp)
                                
                btn_frame = ctk.CTkFrame(self.col2_extra, fg_color="transparent")
                btn_frame.pack(fill="x", pady=(10,0))
                
                ctk.CTkButton(btn_frame, text="Biểu đồ", fg_color="#34495e", height=30, width=40, font=("Arial", 11),
                              command=lambda p=path_comp: self.open_saved_image(p, "Biểu đồ")).pack(side="left", expand=True, fill="x", padx=2)
                ctk.CTkButton(btn_frame, text="Trước", fg_color="#34495e", height=30, width=40, font=("Arial", 11),
                              command=lambda p=path_match_b: self.open_saved_image(p, "Trước")).pack(side="left", expand=True, fill="x", padx=2)
                ctk.CTkButton(btn_frame, text="Sau", fg_color="#34495e", height=30, width=40, font=("Arial", 11),
                              command=lambda p=path_match_a: self.open_saved_image(p, "Sau")).pack(side="left", expand=True, fill="x", padx=2)
            except Exception as e:
                self.log_msg(f"Cảnh báo: Không thể nạp chức năng vẽ Chart: {e}")
            
            self.after_pipeline_end(corrected_gray, res_after, path_corrected_out, full_res_rgb=None)
            self.log_msg("Cập nhật dự đoán thành công!")
            
        except Exception as e:
            self.log_msg(f"LỖI PIPELINE: {e}")
        finally:
            self.btn_run.configure(state="normal", text="NHẬN DIỆN")

    def after_pipeline_end(self, final_gray, final_res, path_save_final, full_res_rgb=None):
        final_rgb = cv2.cvtColor(final_gray, cv2.COLOR_GRAY2RGB)
        self.set_image(self.col3_img, final_rgb, full_res_img=full_res_rgb)
        
        color = "#27ae60" if final_res['confidence'] > 0.4 else "#d35400"
        self.col3_pred.configure(text=f"{final_res['display_name']}\n({final_res['confidence']:.1%})", text_color=color)
        
        self.plot_bar_chart(self.col3_extra, final_res['all_proba'])
        
        cv2.imwrite(path_save_final, final_gray)
        self.log_msg(f"Các ảnh nhận diện đã chụp lại thư mục kết quả.")

if __name__ == "__main__":
    app = LandmarkDashboard()
    app.mainloop()
