import sys
import os
import cv2
import numpy as np
import time
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QLineEdit, QTextEdit, QGridLayout, QMessageBox,
                             QFileDialog, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor

# 尝试导入 YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("请安装 ultralytics: pip install ultralytics")


class ProcessingThread(QThread):
    """后台处理线程 - 封装 AI 与 CV 逻辑"""
    update_log = pyqtSignal(str)
    update_result_ui = pyqtSignal(bool, bool)
    update_ai = pyqtSignal(np.ndarray)
    update_trad = pyqtSignal(np.ndarray)
    update_cam = pyqtSignal(np.ndarray)

    def __init__(self, frame, onnx_path, params):
        super().__init__()
        self.frame = frame
        self.onnx_path = onnx_path
        self.params = params

    def run(self):
        try:
            roi = self.frame.copy()
            cam_display = roi.copy()

            # AI 推理
            ai_display = roi.copy()
            ai_ok = True

            if os.path.exists(self.onnx_path):
                try:
                    model = YOLO(self.onnx_path, task='segment')
                    results = model.predict(source=roi, conf=self.params['conf'], verbose=False)
                    if len(results) > 0:
                        result = results[0]
                        ai_display = result.plot()
                        if result.masks is not None and len(result.masks) > 0:
                            ai_ok = False
                except Exception as e:
                    self.update_log.emit(f"AI模型加载/推理错误: {str(e)}")

            # 传统算法
            bd_ok, bd_display = self.action(roi, self.params)

            # 发送更新
            self.update_cam.emit(cam_display)
            self.update_ai.emit(ai_display)
            self.update_trad.emit(bd_display)
            self.update_result_ui.emit(ai_ok, bd_ok)
            self.update_log.emit(f"检测结束 - AI:{'OK' if ai_ok else 'NG'} 传统:{'OK' if bd_ok else 'NG'}")

        except Exception as e:
            self.update_log.emit(f"线程错误: {str(e)}")

    def action(self, src, p):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 140, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in cnts:
            area = cv2.contourArea(c)
            if area >= 20000:
                h = cv2.boundingRect(c)[3]
                if 200 <= h <= 800:
                    cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)))
        reduced = cv2.bitwise_and(gray, gray, mask=mask)

        def filter_pixels(img, t, a):
            _, b = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
            c_list, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res = np.zeros_like(img)
            for c in c_list:
                if cv2.contourArea(c) >= a:
                    cv2.drawContours(res, [c], -1, 255, -1)
            return res

        s1 = filter_pixels(reduced, p['tg'], p['ta'])
        s2 = filter_pixels(reduced, p['tg1'], p['ta1'])
        union = cv2.bitwise_or(s1, s2)
        final, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bd_ok = (len(final) == 0)

        # 可视化：缺陷区域半透明红色叠加
        display = src.copy()
        mask_colored = np.zeros_like(src)
        mask_colored[union > 0] = (0, 0, 255)  # BGR 红
        display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

        return bd_ok, display


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高更盛04T11安装检测系统")
        self.resize(1600, 900)
        self.current_image = None
        self.cap = None

        self.setup_ui()
        self.init_camera()

    def setup_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #252526; }
            QGroupBox { color: #569CD6; font-weight: bold; border: 1px solid #3F3F46; margin-top: 1ex; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QLabel { color: #DCDCDC; }
            QLineEdit { background-color: #3E3E42; color: #FFFFFF; border: 1px solid #555; padding: 2px; border-radius: 2px; }
            QPushButton { color: #DCDCDC; border: 1px solid #555; padding: 8px; border-radius: 4px; font-size: 12pt; }
            QPushButton:hover { border: 1px solid #007ACC; }
            QTextEdit { background-color: #1E1E1E; color: #B5CEA8; font-family: 'Consolas'; border: 1px solid #3F3F46; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧控制面板
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)

        # 检测结果灯
        result_group = QGroupBox("检测结果")
        result_layout = QHBoxLayout(result_group)
        for label, name in [("AI检测:", "ai"), ("传统检测:", "bd"), ("最终结果:", "final")]:
            result_layout.addWidget(QLabel(label))
            lamp = QLabel("待检测")
            lamp.setAlignment(Qt.AlignCenter)
            lamp.setFixedSize(120, 60)
            lamp.setStyleSheet("background-color: #555555; color: white; font-size: 18pt; border-radius: 10px;")
            result_layout.addWidget(lamp)
            setattr(self, f"{name}_lamp", lamp)
        left_panel.addWidget(result_group)

        # AI 参数
        gb_ai = QGroupBox("AI检测安装参数")
        layout_ai = QGridLayout(gb_ai)
        self.txt_conf = self.add_row(layout_ai, "置信度阈值:", "0.71", 0)
        left_panel.addWidget(gb_ai)

        # 较亮白点
        gb_bright = QGroupBox("较亮白点检测参数")
        layout_br = QGridLayout(gb_bright)
        self.txt_tg = self.add_row(layout_br, "白点阈值(Tgray):", "254", 0)
        self.txt_ta = self.add_row(layout_br, "最小面积(Tarea):", "45", 1)
        left_panel.addWidget(gb_bright)

        # 大面积白点
        gb_large = QGroupBox("大面积白点检测参数")
        layout_lg = QGridLayout(gb_large)
        self.txt_tg1 = self.add_row(layout_lg, "白点阈值(Tgray1):", "240", 0)
        self.txt_ta1 = self.add_row(layout_lg, "最小面积(Tarea1):", "200", 1)
        left_panel.addWidget(gb_large)

        # 日志
        left_panel.addWidget(QLabel("运行日志:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_panel.addWidget(self.log_box)

        # 底部按钮
        bottom_layout = QHBoxLayout()
        self.btn_check = QPushButton("一键检测")
        self.btn_check.setStyleSheet("background-color: #0E639C; color: white; font-weight: bold;")
        self.btn_check.clicked.connect(self.on_check)

        self.btn_snap = QPushButton("采集图像")
        self.btn_snap.setStyleSheet("background-color: #218838; color: white;")
        self.btn_snap.clicked.connect(self.on_snap)

        self.btn_open = QPushButton("选择其他图片")
        self.btn_open.setStyleSheet("background-color: #6f42c1; color: white;")
        self.btn_open.clicked.connect(self.on_open)

        self.btn_save = QPushButton("保存参数")
        self.btn_save.setStyleSheet("background-color: #fd7e14; color: white;")
        self.btn_save.clicked.connect(self.save_params)

        self.btn_clear = QPushButton("清空日志")
        self.btn_clear.setStyleSheet("background-color: #6c757d; color: white;")
        self.btn_clear.clicked.connect(self.log_box.clear)

        for btn in [self.btn_check, self.btn_snap, self.btn_open, self.btn_save, self.btn_clear]:
            btn.setFixedHeight(50)
            bottom_layout.addWidget(btn)

        left_panel.addLayout(bottom_layout)

        main_layout.addLayout(left_panel, 1)

        # 右侧图像区域（三个）
        right_panel = QHBoxLayout()
        self.ai_view = self.add_image_group(right_panel, "AI检测结果", stretch=3, placeholder_color=(255, 0, 0))  # 蓝色占位
        self.trad_view = self.add_image_group(right_panel, "传统算法结果", stretch=2,
                                              placeholder_color=(128, 128, 128))  # 灰色占位
        self.cam_view = self.add_image_group(right_panel, "相机预览", stretch=2, placeholder_color=(0, 0, 0))  # 黑色占位
        main_layout.addLayout(right_panel, 3)

        # 初始占位
        self.set_initial_placeholders()

    def add_row(self, layout, label, val, row):
        layout.addWidget(QLabel(label), row, 0)
        edit = QLineEdit(val)
        layout.addWidget(edit, row, 1)
        return edit

    def add_image_group(self, parent_layout, title, stretch=1, placeholder_color=(0, 0, 0)):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background-color: #1E1E1E; border: 2px solid #3F3F46;")
        label.setScaledContents(False)
        layout.addWidget(label)
        parent_layout.addWidget(group, stretch)
        setattr(self, f"{title.split()[0].lower()}_placeholder_color", placeholder_color)
        return label

    def set_initial_placeholders(self):
        self.display_image(self.ai_view, None, placeholder_color=(255, 0, 0))
        self.display_image(self.trad_view, None, placeholder_color=(128, 128, 128))
        self.display_image(self.cam_view, None, placeholder_color=(0, 0, 0))

    def display_image(self, label, img_np, placeholder_color=None):
        if img_np is None and placeholder_color is not None:
            h, w = 500, 700
            img_np = np.full((h, w, 3), placeholder_color, dtype=np.uint8)
        if img_np is not None:
            rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.clear()

    def update_cam(self, img):
        self.display_image(self.cam_view, img)

    def update_ai(self, img):
        self.display_image(self.ai_view, img)

    def update_trad(self, img):
        self.display_image(self.trad_view, img)

    def log(self, msg):
        self.log_box.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def set_lamp(self, lamp, text, ok):
        lamp.setText(text)
        if ok is True:
            lamp.setStyleSheet("background-color: #28a745; color: white; font-size: 18pt; border-radius: 10px;")
        elif ok is False:
            lamp.setStyleSheet("background-color: #dc3545; color: white; font-size: 18pt; border-radius: 10px;")
        else:
            lamp.setStyleSheet("background-color: #555555; color: white; font-size: 18pt; border-radius: 10px;")

    def init_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
            self.log("相机初始化成功")
        else:
            self.log("未检测到相机，请检查连接")

    def rotate_and_crop(self, src):
        rotated = cv2.rotate(src, cv2.ROTATE_180)
        y1, x1, y2, x2 = 684, 808, 1076, 1582
        h, w = rotated.shape[:2]
        return rotated[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.current_image = img
                self.update_cam(img)
                self.update_ai(None)
                self.update_trad(None)
                self.log(f"载入图片: {os.path.basename(path)}")

    def on_snap(self):
        if self.cap and self.cap.isOpened():
            for _ in range(5): self.cap.grab()
            ret, frame = self.cap.read()
            if ret:
                roi = self.rotate_and_crop(frame)
                self.current_image = roi
                self.update_cam(roi)
                self.update_ai(None)
                self.update_trad(None)
                self.log("相机采集图像成功")
            else:
                QMessageBox.warning(self, "错误", "采集图像失败")
        else:
            QMessageBox.warning(self, "错误", "相机未连接")

    def on_check(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先采集或加载图像")
            return

        try:
            params = {
                'conf': float(self.txt_conf.text()),
                'tg': int(self.txt_tg.text()),
                'ta': int(self.txt_ta.text()),
                'tg1': int(self.txt_tg1.text()),
                'ta1': int(self.txt_ta1.text())
            }
        except ValueError:
            QMessageBox.critical(self, "错误", "参数必须为数字")
            return

        # 检测前状态
        self.set_lamp(self.ai_lamp, "检测中", None)
        self.set_lamp(self.bd_lamp, "检测中", None)
        self.set_lamp(self.final_lamp, "检测中", None)
        self.btn_check.setEnabled(False)

        self.thread = ProcessingThread(self.current_image, "best-seg.onnx", params)
        self.thread.update_ai.connect(self.update_ai)
        self.thread.update_trad.connect(self.update_trad)
        self.thread.update_cam.connect(self.update_cam)
        self.thread.update_result_ui.connect(self.show_results)
        self.thread.update_log.connect(self.log)
        self.thread.finished.connect(lambda: self.btn_check.setEnabled(True))
        self.thread.start()

    def show_results(self, ai_ok, bd_ok):
        final_ok = ai_ok and bd_ok
        self.set_lamp(self.ai_lamp, "OK" if ai_ok else "NG", ai_ok)
        self.set_lamp(self.bd_lamp, "OK" if bd_ok else "NG", bd_ok)
        self.set_lamp(self.final_lamp, "PASS" if final_ok else "FAIL", final_ok)

    def save_params(self):
        try:
            params = {
                "conf": self.txt_conf.text(),
                "tg": self.txt_tg.text(),
                "ta": self.txt_ta.text(),
                "tg1": self.txt_tg1.text(),
                "ta1": self.txt_ta1.text()
            }
            path, _ = QFileDialog.getSaveFileName(self, "保存参数", "params.json", "JSON (*.json)")
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=4, ensure_ascii=False)
                self.log(f"参数已保存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(37, 37, 38))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())