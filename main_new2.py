import sys
import cv2
import numpy as np
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QGridLayout, QSlider, QFormLayout, QFileDialog, QShortcut
)
from PyQt5.QtGui import QKeySequence, QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt

from ultralytics import YOLO

# 新增：使用 Harvester 替代 imagingcontrol4
from harvesters.core import Harvester

# ==================== 全局模型加载 ====================
CONFIG_PATH = "config.json"

MODEL_PATH = "best-seg.onnx"
SEG_MODEL_PATH = "best_seg_DW.onnx"

try:
    model = YOLO(MODEL_PATH)
    seg_model = YOLO(SEG_MODEL_PATH, task='segment')
except ImportError as e:
    if 'onnxruntime' in str(e):
        raise ImportError("缺少 onnxruntime 库，请运行: pip install onnxruntime")
    else:
        raise

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

        # 上部图像显示
        image_layout = QHBoxLayout()
        self.show_window = QLabel("YOLO结果显示区")
        self.show_window.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
        self.show_window.setAlignment(Qt.AlignCenter)
        self.show_window.setScaledContents(True)

        self.picture_box1 = QLabel("传统检测结果显示区")
        self.picture_box1.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
        self.picture_box1.setAlignment(Qt.AlignCenter)
        self.picture_box1.setScaledContents(True)

        image_layout.addWidget(self.show_window)
        image_layout.addWidget(self.picture_box1)
        main_layout.addLayout(image_layout, stretch=6)

        # 中部结果 + 参数
        mid_layout = QHBoxLayout()

        # 结果显示
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()

        ai_layout = QHBoxLayout()
        ai_label = QLabel("AI:")
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
        trad_label = QLabel("传统:")
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

        # 参数区（新增最小掩膜面积）
        param_group = QGroupBox("参数调节")
        param_layout = QGridLayout()

        # 0 AI置信度
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

        # 1 较亮白点阈值
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

        # 2 较亮最小面积
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

        # 3 大面积白点阈值
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

        # 4 大面积最小面积
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

        # 5 新增：最小掩膜面积（用于AI检测过滤小掩膜）
        self.min_mask_area_slider = QSlider(Qt.Horizontal)
        self.min_mask_area_slider.setRange(50, 500)
        self.min_mask_area_slider.setValue(200)
        self.min_mask_area_edit = QLineEdit("200")
        self.min_mask_area_slider.valueChanged.connect(lambda v: self.min_mask_area_edit.setText(str(v)))
        self.min_mask_area_edit.textChanged.connect(lambda t: self.min_mask_area_slider.setValue(int(t)) if t.isdigit() else None)
        param_layout.addWidget(QLabel("最小掩膜面积："), 5, 0)
        param_layout.addWidget(self.min_mask_area_slider, 5, 1)
        param_layout.addWidget(self.min_mask_area_edit, 5, 2)
        param_layout.addWidget(QLabel("(50~500)"), 5, 3)

        param_group.setLayout(param_layout)
        mid_layout.addWidget(param_group)

        main_layout.addLayout(mid_layout, stretch=2)

        # 下部按钮 + 日志 + 统计
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
        self.btn_clear_stats.setStyleSheet("background-color: #a0c4ff; color: white; font-size: 16px; padding: 10px; border-radius: 8px;")
        self.btn_clear_stats.setFixedHeight(50)
        self.btn_clear_stats.clicked.connect(self.clear_stats)
        stats_vlayout.addWidget(self.btn_clear_stats, alignment=Qt.AlignCenter)
        stats_vlayout.addStretch()
        stats_group.setLayout(stats_vlayout)
        stats_group.setFixedWidth(250)
        bottom_layout.addWidget(stats_group)

        main_layout.addLayout(bottom_layout, stretch=1)

        # 快捷键
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.on_detect)
        QShortcut(QKeySequence(Qt.Key_Return), self).activated.connect(self.on_detect)
        QShortcut(QKeySequence(Qt.Key_Enter), self).activated.connect(self.on_detect)

    # ==================== 参数保存/载入 ====================
    def save_config(self):
        config = {
            "conf": self.conf_edit.text(),
            "tgray": self.tgray_edit.text(),
            "tarea": self.tarea_edit.text(),
            "tgray1": self.tgray1_edit.text(),
            "tarea1": self.tarea1_edit.text(),
            "min_mask_area": self.min_mask_area_edit.text(),  # 新增
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
            self.min_mask_area_edit.setText(config.get("min_mask_area", "200"))  # 新增

            self.conf_slider.setValue(int(float(config.get("conf", "0.7")) * 100))
            self.tgray_slider.setValue(int(config.get("tgray", "254")))
            self.tarea_slider.setValue(int(config.get("tarea", "45")))
            self.tgray1_slider.setValue(int(config.get("tgray1", "240")))
            self.tarea1_slider.setValue(int(config.get("tarea1", "200")))
            self.min_mask_area_slider.setValue(int(config.get("min_mask_area", "200")))  # 新增

            self.log_text.append("参数载入成功！")
        except Exception as e:
            self.log_text.append(f"载入失败: {str(e)}")
    def grab_one_frame(self, ia, exposure_us):
        node_map = ia.remote_device.node_map

        # 设置曝光
        node_map.ExposureTime.value = float(exposure_us)

        with ia.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            h, w = component.height, component.width
            data = component.data.reshape(h, w)

            # Bayer → BGR（与你跑通代码一致）
            if component.data_format.endswith("BayerRG8"):
                frame = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2BGR)
            elif component.data_format == "Mono8":
                frame = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)  # 假设其他格式也是灰度，转换为 BGR

        return frame
   
   
    def grab_two_frames(self):
        cti_path = os.path.join(os.getcwd(), "ic4-gentl-u3v_x64.cti")
        if not os.path.exists(cti_path):
            raise Exception("CTI 文件不存在，请检查路径")

        h = Harvester()
        h.add_file(cti_path)
        h.update()

        if len(h.device_info_list) == 0:
            raise Exception("未检测到 JAI USB3 相机")

        # 创建采集器（和你跑通代码完全一致）
        ia = h.create_image_acquirer(0)

        node_map = ia.remote_device.node_map

        # === 参数设置（与你示例一致）===
        try:
            node_map.Width.value = 2448
            node_map.Height.value = 2048
            node_map.PixelFormat.value = "BayerRG8"  # 黑白相机改为 "Mono8"
            node_map.ExposureAuto.value = "Off"
            node_map.GainAuto.value = "Off"
            node_map.AcquisitionFrameRateEnable.value = True
            node_map.AcquisitionFrameRate.value = 10.0
        except Exception as e:
            self.log_text.append(f"相机参数设置警告: {e}")

        # === 开始采集（只 start 一次）===
        ia.start_acquisition()

        # 第一张：高曝光
        self.log_text.append("采集高曝光图像...")
        frame_6 = self.grab_one_frame(ia, 20000)

        # 第二张：低曝光
        self.log_text.append("采集低曝光图像...")
        frame_7 = self.grab_one_frame(ia, 1000)

        ia.stop_acquisition()
        ia.destroy()
        h.reset()

        self.log_text.append("两张图像采集完成")
        return frame_6, frame_7
     

     

    def rotate_and_crop(self, src: np.ndarray) -> np.ndarray:
        rotated = cv2.rotate(src, cv2.ROTATE_180)
        row1, col1, row2, col2 = 684, 808, 1076, 1582
        roi = rotated[row1:row2, col1:col2]
        return roi.copy()

    # ==================== AI检测（新增面积过滤） ====================
    def yolo_detect(self, frame6: np.ndarray, frame7: np.ndarray):
        conf = float(self.conf_edit.text())
        min_area = int(self.min_mask_area_edit.text())  # 新增阈值

        results = model(frame6, conf=conf, iou=0.9, verbose=False)
        annotated = frame7.copy()
        ai_ok = True

        r = results[0]
        if r.masks is None:
            return annotated, True

        masks = r.masks.data.cpu().numpy()
        has_large_mask = False

        for mask in masks:
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            mask_bin = cv2.resize(mask_bin, (annotated.shape[1], annotated.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 计算面积
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            area = cv2.contourArea(contours[0])

            if area < min_area:
                continue  # 小于阈值，忽略

            # 面积足够大，才绘制并标记NG
            has_large_mask = True
            color = np.array([0, 0, 255], dtype=np.uint8)
            colored_mask = np.zeros_like(annotated)
            colored_mask[mask_bin == 255] = color
            annotated = cv2.addWeighted(annotated, 1.0, colored_mask, 0.5, 0)

        if has_large_mask:
            ai_ok = False

        return annotated, ai_ok

    # ==================== 传统检测（保持不变） ====================
    def traditional_detect(self, src6: np.ndarray, src7: np.ndarray):
        tgray = int(self.tgray_edit.text())
        tarea = int(self.tarea_edit.text())
        tgray1 = int(self.tgray1_edit.text())
        tarea1 = int(self.tarea1_edit.text())

        gray = cv2.cvtColor(src6, cv2.COLOR_BGR2GRAY) if len(src6.shape) == 3 else src6.copy()
        gray7 = cv2.cvtColor(src7, cv2.COLOR_BGR2GRAY) if len(src7.shape) == 3 else src7.copy()

        results = seg_model(src6, conf=0.25, iou=0.7, verbose=False)[0]

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
        sel1 = np.zeros(src6.shape[:2], dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= tarea:
                cv2.drawContours(sel1, [c], -1, 255, -1)

        _, bright_large = cv2.threshold(reduced, tgray1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sel2 = np.zeros(src6.shape[:2], dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= tarea1:
                cv2.drawContours(sel2, [c], -1, 255, -1)

        union = cv2.bitwise_or(sel1, sel2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        union = cv2.dilate(union, kernel_dilate)

        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        trad_ok = len(contours) == 0

        color = cv2.cvtColor(gray7, cv2.COLOR_GRAY2BGR)
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
            text_box.setStyleSheet("background-color: #32CD32; color: white; border-radius: 8px; padding: 10px; font-weight: bold;")
        else:
            text_box.setStyleSheet("background-color: #FF4444; color: yellow; border-radius: 8px; padding: 10px; font-weight: bold;")

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
            frame_6, frame_7 = self.grab_two_frames()
            roi_6 = self.rotate_and_crop(frame_6)
            roi_7 = self.rotate_and_crop(frame_7)
            if roi_6.size == 0 or roi_7.size == 0:
                self.log_text.append("ROI无效")
                return

            yolo_img, ai_ok = self.yolo_detect(roi_6.copy(), roi_7.copy())
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.update_result_text(self.text_ai, ai_ok)

            trad_img, trad_ok = self.traditional_detect(roi_6.copy(), roi_7.copy())
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_trad, trad_ok)

            final_ok = ai_ok and trad_ok
            self.update_result_text(self.text_final, final_ok)
            self.update_stats(final_ok)

            self.log_text.append(
                f"检测完成 | AI: {'OK' if ai_ok else 'NG'} | "
                f"传统: {'OK' if trad_ok else 'NG'} | 综合: {'OK' if final_ok else 'NG'}"
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