import sys
import cv2
import numpy as np
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QGridLayout, QSlider, QFormLayout, QFileDialog
)
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ultralytics import YOLO

# ==================== 新增：PIL 用于中文绘制 ====================
from PIL import Image, ImageDraw, ImageFont

# ==================== 全局模型加载 ====================
CONFIG_PATH = "config.json"

MODEL_PATH = "best-seg.onnx"
SEG_MODEL_PATH = "best_seg_DW.onnx"

model = YOLO(MODEL_PATH)
seg_model = YOLO(SEG_MODEL_PATH, task='segment')

# ==================== 工具函数 ====================
def cv2_to_qpixmap(mat: np.ndarray) -> QPixmap:
    if mat is None or mat.size == 0:
        return QPixmap()
    if mat.ndim == 2:
        height, width = mat.shape
        bytes_per_line = width
        q_img = QImage(mat.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytes_per_line = channel * width
        q_img = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)

# ==================== 中文绘制函数（解决 cv2.putText 中文乱码） ====================
def draw_chinese_text(img: np.ndarray, text: str, position: tuple, font_size: int = 50, color: tuple = (255, 0, 0), thickness: int = 4):
    """
    在 OpenCV 图像上绘制中文文字（使用 PIL，避免乱码）
    img: BGR numpy array
    text: 中文字符串
    position: (x, y) 左上角坐标
    font_size: 字体大小（像素）
    color: BGR 颜色元组
    thickness: 描边粗细（模拟加粗）
    返回: 绘制后的 BGR 图像
    """
    # 字体路径（Windows 中仿宋字体，常见路径；如果乱码或报错，请修改为本机实际路径）
    font_path = r"C:\Windows\Fonts\simfang.ttf"  # 仿宋（如果没有，可改为 simhei.ttf 黑体）

    cv_img = img.copy()
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # 如果字体路径不对，fallback 到默认（会乱码，但不崩溃）
        font = ImageFont.load_default()
        print(f"警告：未找到字体 {font_path}，使用默认字体（中文可能乱码）")

    # PIL 颜色为 RGB，描边实现加粗效果
    rgb_color = color[::-1]  # BGR -> RGB
    draw.text(position, text, font=font, fill=rgb_color, stroke_width=thickness, stroke_fill=rgb_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高更盛04T11安装检测")
        icon_path = "1.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("警告：未找到 1.ico 文件，使用默认图标")
        self.resize(1400, 800)

        self.total_count = 0
        self.good_count = 0

        self.init_ui()
        self.load_config()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.setStyleSheet("""
            QMainWindow { background-color: #f0f4f8; }
            QLabel { font-size: 14px; color: #333; }
            QGroupBox { 
                font-weight: bold; font-size: 16px; 
                border: 2px solid #a0c4ff; border-radius: 10px; 
                margin-top: 10px; padding-top: 10px;
                background-color: #ffffff;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { 
                background-color: #4a90e2; color: white; font-size: 16px; 
                border-radius: 8px; padding: 10px; min-width: 120px;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2a6aaf; }
            QLineEdit { 
                padding: 8px; border: 2px solid #bdc3c7; border-radius: 6px; 
                font-size: 14px;
            }
            QSlider::groove:horizontal { 
                height: 8px; background: #e0e0e0; border-radius: 4px; 
            }
            QSlider::handle:horizontal { 
                background: #4a90e2; width: 20px; border-radius: 10px; margin: -6px 0;
            }
            QTextEdit { border: 2px solid #bdc3c7; border-radius: 8px; }
        """)

        image_layout = QHBoxLayout()
        self.show_window = QLabel()
        self.show_window.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
        self.show_window.setAlignment(Qt.AlignCenter)
        self.show_window.setScaledContents(True)

        self.picture_box1 = QLabel()
        self.picture_box1.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
        self.picture_box1.setAlignment(Qt.AlignCenter)
        self.picture_box1.setScaledContents(True)

        image_layout.addWidget(self.show_window)
        image_layout.addWidget(self.picture_box1)
        main_layout.addLayout(image_layout, stretch=6)

        mid_layout = QHBoxLayout()

        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()

        ai_layout = QHBoxLayout()
        ai_label = QLabel("AI安装检测:")
        ai_label.setFont(QFont("Arial", 20, QFont.Bold))
        ai_label.setAlignment(Qt.AlignCenter)
        ai_label.setFixedWidth(80)

        self.text_ai = QLabel("--")
        self.text_ai.setAlignment(Qt.AlignCenter)
        self.text_ai.setFixedHeight(60)
        self.text_ai.setFixedWidth(120)
        self.text_ai.setFont(QFont("Arial", 20, QFont.Bold))
        self.text_ai.setStyleSheet("background-color: #f0f0f0; color: #333; border-radius: 8px;")

        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.text_ai)
        result_layout.addLayout(ai_layout)

        trad_layout = QHBoxLayout()
        trad_label = QLabel("白点检测:")
        trad_label.setFont(QFont("Arial", 20, QFont.Bold))
        trad_label.setAlignment(Qt.AlignCenter)
        trad_label.setFixedWidth(80)

        self.text_trad = QLabel("--")
        self.text_trad.setAlignment(Qt.AlignCenter)
        self.text_trad.setFixedHeight(60)
        self.text_trad.setFixedWidth(120)
        self.text_trad.setFont(QFont("Arial", 20, QFont.Bold))
        self.text_trad.setStyleSheet("background-color: #f0f0f0; color: #333; border-radius: 8px;")

        trad_layout.addWidget(trad_label)
        trad_layout.addWidget(self.text_trad)
        result_layout.addLayout(trad_layout)

        final_layout = QHBoxLayout()
        final_label = QLabel("综合:")
        final_label.setFont(QFont("Arial", 36, QFont.Bold))
        final_label.setAlignment(Qt.AlignCenter)
        final_label.setFixedWidth(100)

        self.text_final = QLabel("--")
        self.text_final.setAlignment(Qt.AlignCenter)
        self.text_final.setFixedHeight(100)
        self.text_final.setFixedWidth(180)
        self.text_final.setFont(QFont("Arial", 36, QFont.Bold))
        self.text_final.setStyleSheet("background-color: #f0f0f0; color: #333; border-radius: 12px;")

        final_layout.addWidget(final_label)
        final_layout.addWidget(self.text_final)
        result_layout.addLayout(final_layout)

        result_group.setLayout(result_layout)
        result_group.setFixedWidth(300)
        mid_layout.addWidget(result_group)

        param_group = QGroupBox("参数调节")
        param_layout = QGridLayout()

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(50, 100)
        self.conf_slider.setValue(70)
        self.conf_edit = QLineEdit("0.7")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_edit.setText(f"{v/100:.2f}"))
        self.conf_edit.textChanged.connect(lambda t: self.conf_slider.setValue(int(float(t)*100)) if t else None)
        param_layout.addWidget(QLabel("AI置信度阈值："), 0, 0)
        param_layout.addWidget(self.conf_slider, 0, 1)
        param_layout.addWidget(self.conf_edit, 0, 2)
        param_layout.addWidget(QLabel("(0.5~1.0)"), 0, 3)

        self.tgray_slider = QSlider(Qt.Horizontal)
        self.tgray_slider.setRange(240, 254)
        self.tgray_slider.setValue(254)
        self.tgray_edit = QLineEdit("254")
        self.tgray_slider.valueChanged.connect(lambda v: self.tgray_edit.setText(str(v)))
        self.tgray_edit.textChanged.connect(lambda t: self.tgray_slider.setValue(int(t)) if t.isdigit() else None)
        param_layout.addWidget(QLabel("较亮白点阈值："), 1, 0)
        param_layout.addWidget(self.tgray_slider, 1, 1)
        param_layout.addWidget(self.tgray_edit, 1, 2)
        param_layout.addWidget(QLabel("(240~254)"), 1, 3)

        self.tarea_slider = QSlider(Qt.Horizontal)
        self.tarea_slider.setRange(15, 50)
        self.tarea_slider.setValue(45)
        self.tarea_edit = QLineEdit("45")
        self.tarea_slider.valueChanged.connect(lambda v: self.tarea_edit.setText(str(v)))
        self.tarea_edit.textChanged.connect(lambda t: self.tarea_slider.setValue(int(t)) if t.isdigit() else None)
        param_layout.addWidget(QLabel("较亮最小面积："), 2, 0)
        param_layout.addWidget(self.tarea_slider, 2, 1)
        param_layout.addWidget(self.tarea_edit, 2, 2)
        param_layout.addWidget(QLabel("(15~50)"), 2, 3)

        self.tgray1_slider = QSlider(Qt.Horizontal)
        self.tgray1_slider.setRange(230, 245)
        self.tgray1_slider.setValue(240)
        self.tgray1_edit = QLineEdit("240")
        self.tgray1_slider.valueChanged.connect(lambda v: self.tgray1_edit.setText(str(v)))
        self.tgray1_edit.textChanged.connect(lambda t: self.tgray1_slider.setValue(int(t)) if t.isdigit() else None)
        param_layout.addWidget(QLabel("大面积白点阈值："), 3, 0)
        param_layout.addWidget(self.tgray1_slider, 3, 1)
        param_layout.addWidget(self.tgray1_edit, 3, 2)
        param_layout.addWidget(QLabel("(230~245)"), 3, 3)

        self.tarea1_slider = QSlider(Qt.Horizontal)
        self.tarea1_slider.setRange(200, 300)
        self.tarea1_slider.setValue(200)
        self.tarea1_edit = QLineEdit("200")
        self.tarea1_slider.valueChanged.connect(lambda v: self.tarea1_edit.setText(str(v)))
        self.tarea1_edit.textChanged.connect(lambda t: self.tarea1_slider.setValue(int(t)) if t.isdigit() else None)
        param_layout.addWidget(QLabel("大面积最小面积："), 4, 0)
        param_layout.addWidget(self.tarea1_slider, 4, 1)
        param_layout.addWidget(self.tarea1_edit, 4, 2)
        param_layout.addWidget(QLabel("(200~300)"), 4, 3)

        self.ai_area_slider = QSlider(Qt.Horizontal)
        self.ai_area_slider.setRange(2000, 8000)
        self.ai_area_slider.setValue(3000)
        self.ai_area_edit = QLineEdit("200")
        self.ai_area_slider.valueChanged.connect(lambda v: self.ai_area_edit.setText(str(v)))
        self.ai_area_edit.textChanged.connect(lambda t: self.ai_area_slider.setValue(int(t)) if t.isdigit() else None)
        param_layout.addWidget(QLabel("AI最小面积："), 5, 0)
        param_layout.addWidget(self.ai_area_slider, 5, 1)
        param_layout.addWidget(self.ai_area_edit, 5, 2)
        param_layout.addWidget(QLabel("(2000~8000)"), 5, 3)

        param_group.setLayout(param_layout)
        mid_layout.addWidget(param_group)

        main_layout.addLayout(mid_layout, stretch=2)

        bottom_layout = QHBoxLayout()

        btn_layout = QVBoxLayout()
        self.btn_detect = QPushButton("一键检测")
        self.btn_detect.setFixedSize(160, 80)
        self.btn_detect.clicked.connect(self.on_detect)

        self.btn_save = QPushButton("保存参数")
        self.btn_save.clicked.connect(self.save_config)

        self.btn_load = QPushButton("载入参数")
        self.btn_load.clicked.connect(self.load_config)

        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addStretch()
        bottom_layout.addLayout(btn_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        bottom_layout.addWidget(self.log_text, stretch=3)

        stats_group = QGroupBox("生产统计")
        stats_vlayout = QVBoxLayout()

        stats_form = QFormLayout()
        self.label_total = QLabel("0")
        self.label_good = QLabel("0")
        self.label_rate = QLabel("0.00%")
        self.label_rate.setFont(QFont("Arial", 18, QFont.Bold))
        stats_form.addRow("生产总数：", self.label_total)
        stats_form.addRow("良品总数：", self.label_good)
        stats_form.addRow("良　　率：", self.label_rate)
        stats_vlayout.addLayout(stats_form)

        self.btn_clear_stats = QPushButton("清零统计")
        self.btn_clear_stats.setStyleSheet("""
                  background-color: #a0c4ff; 
                  color: white; 
                  font-size: 16px; 
                  padding: 10px; 
                  border-radius: 8px;
              """)
        self.btn_clear_stats.setFixedHeight(50)
        self.btn_clear_stats.clicked.connect(self.clear_stats)
        stats_vlayout.addWidget(self.btn_clear_stats, alignment=Qt.AlignCenter)

        stats_vlayout.addStretch()
        stats_group.setLayout(stats_vlayout)
        stats_group.setFixedWidth(250)
        bottom_layout.addWidget(stats_group)

        main_layout.addLayout(bottom_layout, stretch=1)

        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.on_detect)

        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(self.on_detect)

        enter_shortcut2 = QShortcut(QKeySequence(Qt.Key_Enter), self)
        enter_shortcut2.activated.connect(self.on_detect)

    def save_config(self):
        config = {
            "conf": self.conf_edit.text(),
            "tgray": self.tgray_edit.text(),
            "tarea": self.tarea_edit.text(),
            "tgray1": self.tgray1_edit.text(),
            "tarea1": self.tarea1_edit.text(),
            "ai_min_area": self.ai_area_edit.text(),
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.log_text.append("参数保存成功！")
        except Exception as e:
            self.log_text.append(f"保存失败: {str(e)}")

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            self.log_text.append("未找到配置文件，使用默认参数")
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.conf_edit.setText(config.get("conf", "0.7"))
            self.tgray_edit.setText(config.get("tgray", "254"))
            self.tarea_edit.setText(config.get("tarea", "45"))
            self.tgray1_edit.setText(config.get("tgray1", "240"))
            self.tarea1_edit.setText(config.get("tarea1", "200"))
            self.ai_area_edit.setText(config.get("ai_min_area", "200"))

            self.conf_slider.setValue(int(float(config.get("conf", "0.7")) * 100))
            self.tgray_slider.setValue(int(config.get("tgray", "254")))
            self.tarea_slider.setValue(int(config.get("tarea", "45")))
            self.tgray1_slider.setValue(int(config.get("tgray1", "240")))
            self.tarea1_slider.setValue(int(config.get("tarea1", "200")))
            self.ai_area_slider.setValue(int(config.get("ai_min_area", "200")))
            self.log_text.append("参数载入成功！")
        except Exception as e:
            self.log_text.append(f"载入失败: {str(e)}")

    def grab_one_frame(self) -> np.ndarray:
        cap = None
        for i in range(5):
            cap_try = cv2.VideoCapture(i)
            if cap_try.isOpened():
                ret, frame = cap_try.read()
                if ret and frame is not None:
                    cap = cap_try
                    print(f"成功打开相机 index = {i}")
                    break
            cap_try.release()

        if cap is None:
            raise Exception("未找到可用相机")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        for _ in range(10):
            cap.grab()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("读取帧失败")
        return frame

    def rotate_and_crop(self, src: np.ndarray) -> np.ndarray:
        rotated = cv2.rotate(src, cv2.ROTATE_180)
        row1, col1, row2, col2 = 684, 808, 1076, 1582
        roi = rotated[row1:row2, col1:col2]
        return roi.copy()

    def yolo_detect(self, frame: np.ndarray):
        conf = float(self.conf_edit.text())
        results = model(frame, conf=conf, iou=0.9, verbose=False)
        
        plot_rgb = results[0].plot()
        plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)
        
        ai_min_area = int(self.ai_area_edit.text())
        significant = False
        
        masks = results[0].masks
        if masks is not None and len(masks.xy) > 0:
            for xy in masks.xy:
                if len(xy) < 3:
                    continue
                contour = np.round(xy).astype(np.int32).reshape((-1, 1, 2))
                area = cv2.contourArea(contour)
                if area >= ai_min_area:
                    significant = True
                    break
        
        if significant:
            display_img = plot_bgr
            ai_ok = False
        else:
            display_img = frame.copy()
            ai_ok = True
        
        return display_img, ai_ok

    def traditional_detect(self, src: np.ndarray):
        tgray = int(self.tgray_edit.text())
        tarea = int(self.tarea_edit.text())
        tgray1 = int(self.tgray1_edit.text())
        tarea1 = int(self.tarea1_edit.text())

        if len(src.shape) == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src.copy()

        results = seg_model(src, conf=0.25, iou=0.7, verbose=False)[0]

        region_mask = np.zeros(gray.shape, dtype=np.uint8)

        if results.masks is not None and len(results.masks.xy) > 0:
            max_area = 0
            max_poly = None

            for poly in results.masks.xy:
                if len(poly) < 3:
                    continue
                poly_int = np.round(poly).astype(np.int32)
                area = cv2.contourArea(poly_int)
                if area > max_area:
                    max_area = area
                    max_poly = poly_int

            if max_poly is not None:
                cv2.fillPoly(region_mask, [max_poly], 255)

        reduced = cv2.bitwise_and(gray, region_mask)

        _, bright_small = cv2.threshold(reduced, tgray, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sel1 = np.zeros(src.shape[:2], dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= tarea:
                cv2.drawContours(sel1, [c], -1, 255, -1)

        _, bright_large = cv2.threshold(reduced, tgray1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sel2 = np.zeros(src.shape[:2], dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= tarea1:
                cv2.drawContours(sel2, [c], -1, 255, -1)

        union = cv2.bitwise_or(sel1, sel2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        union = cv2.dilate(union, kernel_dilate)

        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        trad_ok = len(contours) == 0

        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if np.any(region_mask > 0):
            blue_overlay = np.full_like(color, (255, 0, 0), dtype=np.uint8)
            masked_blue = cv2.bitwise_and(blue_overlay, blue_overlay, mask=region_mask)
            color = cv2.addWeighted(color, 0.9, masked_blue, 0.1, 0)

        defect_color = (0, 255, 0) if trad_ok else (0, 0, 255)
        color = cv2.drawContours(color, contours, -1, defect_color, 3)

        return color, trad_ok

    def update_result_text(self, text_box: QLabel, ok: bool):
        text_box.setText("OK" if ok else "NG")
        if ok:
            text_box.setStyleSheet("""
                background-color: #32CD32; 
                color: white; 
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            """)
        else:
            text_box.setStyleSheet("""
                background-color: #FF4444; 
                color: yellow; 
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            """)

    def update_stats(self, final_ok: bool):
        self.total_count += 1
        if final_ok:
            self.good_count += 1
        self.label_total.setText(str(self.total_count))
        self.label_good.setText(str(self.good_count))
        rate = (self.good_count / self.total_count * 100) if self.total_count > 0 else 0
        self.label_rate.setText(f"{rate:.2f}%")

    def on_detect(self):
        try:
            frame = self.grab_one_frame()
            roi = self.rotate_and_crop(frame)
            if roi.size == 0:
                self.log_text.append("ROI无效")
                return

            yolo_img, ai_ok = self.yolo_detect(roi.copy())

            # === 左侧标题：使用 PIL 绘制中文（解决乱码）===
            yolo_img = draw_chinese_text(yolo_img, "AI检测安装到位", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)

            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.update_result_text(self.text_ai, ai_ok)

            trad_img, trad_ok = self.traditional_detect(roi.copy())

            # === 右侧标题：使用 PIL 绘制中文（解决乱码）===
            trad_img = draw_chinese_text(trad_img, "白点灰尘检测", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)

            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_trad, trad_ok)

            final_ok = ai_ok and trad_ok
            self.update_result_text(self.text_final, final_ok)

            self.update_stats(final_ok)

            self.log_text.append(
                f"检测完成 | AI安装检测: {'OK' if ai_ok else 'NG'} | "
                f"白点灰尘检测: {'OK' if trad_ok else 'NG'} | 综合: {'OK' if final_ok else 'NG'}"
            )
        except Exception as e:
            self.log_text.append(f"错误: {str(e)}")

    def clear_stats(self):
        self.total_count = 0
        self.good_count = 0
        self.label_total.setText("0")
        self.label_good.setText("0")
        self.label_rate.setText("0.00%")
        self.log_text.append("【统计】生产统计数据已清零")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())