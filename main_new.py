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
from PyQt5.QtGui import QPixmap, QImage, QFont,QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ultralytics import YOLO

# ==================== 全局模型加载 ====================

CONFIG_PATH = "config.json"  # 参数保存文件（根目录）


MODEL_PATH = "best-seg.onnx"          # 主 AI 检测模型（detect 或 seg）
SEG_MODEL_PATH = "best_seg_DW.onnx"   # 传统检测用的分割掩膜模型

model = YOLO(MODEL_PATH)                        # 主 AI 模型
seg_model = YOLO(SEG_MODEL_PATH, task='segment')  # 传统检测专用的分割模型
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
        # === 新增：设置窗口图标 ===
        icon_path = "1.ico"  # 根目录下的 1.ico（假设与脚本同目录）
        if os.path.exists(icon_path):  # 可选：检查文件是否存在，避免报错
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("警告：未找到 1.ico 文件，使用默认图标")
        self.resize(1400, 800)

        # 生产统计
        self.total_count = 0
        self.good_count = 0

        self.init_ui()
        self.load_config()  # 启动时自动载入参数

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ==================== 全局现代化样式 ====================
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

        # ==================== 上部图像显示 ====================
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

        # ==================== 中部结果 + 参数 ====================
        mid_layout = QHBoxLayout()

        # 结果显示（固定标签 + 结果色块，标签永不被覆盖）
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()

        # AI 结果行
        ai_layout = QHBoxLayout()
        ai_label = QLabel("AI:")
        ai_label.setFont(QFont("Arial", 20, QFont.Bold))
        ai_label.setAlignment(Qt.AlignCenter)
        ai_label.setFixedWidth(80)

        self.text_ai = QLabel("--")  # 初始显示 --
        self.text_ai.setAlignment(Qt.AlignCenter)
        self.text_ai.setFixedHeight(60)
        self.text_ai.setFixedWidth(120)
        self.text_ai.setFont(QFont("Arial", 20, QFont.Bold))
        self.text_ai.setStyleSheet("background-color: #f0f0f0; color: #333; border-radius: 8px;")

        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.text_ai)
        result_layout.addLayout(ai_layout)

        # 传统结果行
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

        # 综合结果行（更大、更突出）
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
        result_group.setFixedWidth(300)  # 稍宽一点，容纳标签+结果
        mid_layout.addWidget(result_group)

        # 参数区
        param_group = QGroupBox("参数调节")
        param_layout = QGridLayout()

        # AI置信度 (0.5~1.0)
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

        # 较亮白点阈值 (240~254)
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

        # 较亮最小面积 (15~50)
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

        # 大面积白点阈值 (230~245)
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

        # 大面积最小面积 (200~300)
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

        param_group.setLayout(param_layout)
        mid_layout.addWidget(param_group)

        main_layout.addLayout(mid_layout, stretch=2)

        # ==================== 下部：按钮 + 日志 + 统计 ====================
        bottom_layout = QHBoxLayout()

        # 按钮区
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

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        bottom_layout.addWidget(self.log_text, stretch=3)

        # 生产统计
        stats_group = QGroupBox("生产统计")
        stats_vlayout = QVBoxLayout()  # 主垂直布局

        # 统计表单部分
        stats_form = QFormLayout()
        self.label_total = QLabel("0")
        self.label_good = QLabel("0")
        self.label_rate = QLabel("0.00%")
        self.label_rate.setFont(QFont("Arial", 18, QFont.Bold))
        stats_form.addRow("生产总数：", self.label_total)
        stats_form.addRow("良品总数：", self.label_good)
        stats_form.addRow("良　　率：", self.label_rate)
        stats_vlayout.addLayout(stats_form)

        # 清零按钮
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

        stats_vlayout.addStretch()  # 按钮下方留空
        stats_group.setLayout(stats_vlayout)
        stats_group.setFixedWidth(250)
        bottom_layout.addWidget(stats_group)

        main_layout.addLayout(bottom_layout, stretch=1)
        # === 新增：键盘快捷键触发一键检测 ===
        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(self.on_detect)

        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(self.on_detect)

        enter_shortcut2 = QShortcut(QKeySequence(Qt.Key_Enter), self)
        enter_shortcut2.activated.connect(self.on_detect)
    # ==================== 参数保存/载入 ====================
    def save_config(self):
        config = {
            "conf": self.conf_edit.text(),
            "tgray": self.tgray_edit.text(),
            "tarea": self.tarea_edit.text(),
            "tgray1": self.tgray1_edit.text(),
            "tarea1": self.tarea1_edit.text(),
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
            # 同步滑块
            self.conf_slider.setValue(int(float(config.get("conf", "0.7")) * 100))
            self.tgray_slider.setValue(int(config.get("tgray", "254")))
            self.tarea_slider.setValue(int(config.get("tarea", "45")))
            self.tgray1_slider.setValue(int(config.get("tgray1", "240")))
            self.tarea1_slider.setValue(int(config.get("tarea1", "200")))
            self.log_text.append("参数载入成功！")
        except Exception as e:
            self.log_text.append(f"载入失败: {str(e)}")

    # ==================== 相机采集 ====================
    def grab_two_frames(self):
        cap = None

        # 依次尝试相机
        for i in range(5):
            cap_try = cv2.VideoCapture(i)
            if cap_try.isOpened():
                ret, frame = cap_try.read()
                if ret and frame is not None:
                    cap = cap_try
                    break
            cap_try.release()

        if cap is None:
            raise Exception("未找到可用相机")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

        # ========= 曝光 -6 =========
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        for _ in range(10):
            cap.grab()
        ret6, frame_6 = cap.read()
        if not ret6:
            cap.release()
            raise Exception("曝光 -6 采集失败")

        # ========= 曝光 -7 =========
        cap.set(cv2.CAP_PROP_EXPOSURE, -9)
        for _ in range(10):
            cap.grab()
        ret7, frame_7 = cap.read()
        if not ret7:
            cap.release()
            raise Exception("曝光 -7 采集失败")

        cap.release()
        return frame_6, frame_7

    def rotate_and_crop(self, src: np.ndarray) -> np.ndarray:
        rotated = cv2.rotate(src, cv2.ROTATE_180)
        row1, col1, row2, col2 = 684, 808, 1076, 1582
        roi = rotated[row1:row2, col1:col2]
        return roi.copy()

    # ==================== 检测逻辑 ====================
    def yolo_detect(self, frame: np.ndarray):
        conf = float(self.conf_edit.text())
        results = model(frame, conf=conf, iou=0.9, verbose=False)
        annotated = results[0].plot()
        ai_ok = len(results[0]) == 0
        return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), ai_ok

    def traditional_detect(self, src: np.ndarray):
        tgray = int(self.tgray_edit.text())
        tarea = int(self.tarea_edit.text())
        tgray1 = int(self.tgray1_edit.text())
        tarea1 = int(self.tarea1_edit.text())

        if len(src.shape) == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src.copy()

        # 使用全局加载的分割模型进行推理
        results = seg_model(src, conf=0.25, iou=0.7, verbose=False)[0]

        # 初始化 region_mask（感兴趣区域掩膜）
        region_mask = np.zeros(gray.shape, dtype=np.uint8)  # (h, w)

        # 只取最大块掩膜（如果有检测到实例）
        if results.masks is not None and len(results.masks.xy) > 0:
            max_area = 0
            max_poly = None

            for poly in results.masks.xy:  # 遍历每个实例的多边形
                if len(poly) < 3:  # 点太少无法构成有效轮廓
                    continue
                poly_int = np.round(poly).astype(np.int32)
                area = cv2.contourArea(poly_int)
                if area > max_area:
                    max_area = area
                    max_poly = poly_int

            # 只填充面积最大的一个实例
            if max_poly is not None:
                cv2.fillPoly(region_mask, [max_poly], 255)

        # 在感兴趣区域内做亮斑检测
        reduced = cv2.bitwise_and(gray, region_mask)

        # 较亮小白点
        _, bright_small = cv2.threshold(reduced, tgray, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sel1 = np.zeros(src.shape[:2], dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= tarea:
                cv2.drawContours(sel1, [c], -1, 255, -1)

        # 大面积白点
        _, bright_large = cv2.threshold(reduced, tgray1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bright_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sel2 = np.zeros(src.shape[:2], dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= tarea1:
                cv2.drawContours(sel2, [c], -1, 255, -1)

        # 合并并膨胀
        union = cv2.bitwise_or(sel1, sel2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        union = cv2.dilate(union, kernel_dilate)

        # 最终轮廓（缺陷）
        contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        trad_ok = len(contours) == 0

        # ==================== 可视化 ====================
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 叠加半透明蓝色掩膜（仅最大块区域）
        if np.any(region_mask > 0):
            blue_overlay = np.full_like(color, (255, 0, 0), dtype=np.uint8)  # BGR 蓝色
            masked_blue = cv2.bitwise_and(blue_overlay, blue_overlay, mask=region_mask)
            color = cv2.addWeighted(color, 0.9, masked_blue, 0.1, 0)

        # 在最上层绘制最终缺陷轮廓（绿色OK / 红色NG）
        defect_color = (0, 255, 0) if trad_ok else (0, 0, 255)
        color = cv2.drawContours(color, contours, -1, defect_color, 3)

        return color, trad_ok

    def update_result_text(self, text_box: QLabel, ok: bool):
        text_box.setText("OK" if ok else "NG")
        if ok:
            # OK：绿色背景 + 白色文字
            text_box.setStyleSheet("""
                background-color: #32CD32; 
                color: white; 
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
            """)
        else:
            # NG：红色背景 + 黄色文字（高对比、醒目）
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
            # 1. 采图
            frame_6, frame_7 = self.grab_two_frames()
            roi_6 = self.rotate_and_crop(frame_6)
            roi_7 = self.rotate_and_crop(frame_7)

            # 2. frame_6 推理
            yolo_img, ai_ok = self.yolo_detect(roi_6.copy())
            trad_img, trad_ok = self.traditional_detect(roi_6.copy())

            # =================================================
            # 左窗体：frame_6 的【推理掩膜 + 白点轮廓】叠加显示
            # =================================================
            # yolo_img 本身是彩色
            # trad_img 是灰度转 BGR + 轮廓
            left_show = cv2.addWeighted(
                yolo_img, 0.9,
                roi_7 , 0.1,
                0
            )
            right_show = cv2.addWeighted(
                trad_img, 0.9,
                roi_7, 0.1,
                0
            )

            self.show_window.setPixmap(cv2_to_qpixmap(left_show))
            self.picture_box1.setPixmap(cv2_to_qpixmap(right_show))
            # =================================================
            # 右窗体：frame_7 原图 + frame_6 的检测轮廓
            # =================================================
           # right_show = roi_7.copy()


            # =================================================
            # 结果 & 统计（仍然只以 frame_6 为准）
            # =================================================
            final_ok = ai_ok and trad_ok
            self.update_result_text(self.text_ai, ai_ok)
            self.update_result_text(self.text_trad, trad_ok)
            self.update_result_text(self.text_final, final_ok)
            self.update_stats(final_ok)

            self.log_text.append(
                f"检测完成 | AI:{'OK' if ai_ok else 'NG'} | "
                f"传统:{'OK' if trad_ok else 'NG'} | 综合:{'OK' if final_ok else 'NG'}"
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

