import sys
import os
import cv2
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QGroupBox,
                             QLineEdit, QTextEdit, QGridLayout, QFileDialog,
                             QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

# 导入 YOLO 库
try:
    from ultralytics import YOLO
except ImportError:
    print("请安装依赖: pip install ultralytics PyQt5 opencv-python")


class WorkerThread(QThread):
    """后台处理线程：确保界面在 AI 计算时不卡顿"""
    update_log = pyqtSignal(str)
    update_image = pyqtSignal(np.ndarray)
    finished_res = pyqtSignal(bool, bool)  # AI_result, BD_result

    def __init__(self, frame, onnx_path, params, is_camera):
        super().__init__()
        self.frame = frame
        self.onnx_path = onnx_path
        self.params = params
        self.is_camera = is_camera

    def run(self):
        try:
            # 1. 预处理 (对应 C# RotateAndCrop)
            # 如果是相机抓拍则旋转裁剪，本地图片保持原样
            if self.is_camera:
                roi = self.rotate_and_crop(self.frame)
            else:
                roi = self.frame.copy()

            # 2. AI 分割推理 (对应 C# ProcessSegmentationAsync)
            ai_ok = True
            display_img = roi.copy()
            if os.path.exists(self.onnx_path):
                model = YOLO(self.onnx_path, task='segment')
                results = model.predict(source=roi, conf=self.params['conf'], verbose=False)
                if len(results) > 0:
                    display_img = results[0].plot()  # 绘制分割结果
                    if results[0].masks is not None:
                        ai_ok = False  # 检测到目标(缺陷)则为 NG

            # 3. 传统视觉算法 (对应 C# action)
            bd_ok, _ = self.action(roi, self.params)

            self.update_image.emit(display_img)
            self.finished_res.emit(ai_ok, bd_ok)
            self.update_log.emit(f"检测结束: AI={'OK' if ai_ok else 'NG'}, 白点={'OK' if bd_ok else 'NG'}")

        except Exception as e:
            self.update_log.emit(f"错误: {str(e)}")

    def rotate_and_crop(self, src):
        rotated = cv2.rotate(src, cv2.ROTATE_180)
        # C# 坐标: row1=684, col1=808, row2=1076, col2=1582
        return rotated[684:1076, 808:1582]

    def action(self, src, p):
        # 1:1 复刻您的 action 函数逻辑
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 140, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for c in cnts:
            h = cv2.boundingRect(c)[3]
            if 20000 <= cv2.contourArea(c) <= 300000 and 200 <= h <= 800:
                cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55)))
        reduced = cv2.bitwise_and(gray, gray, mask=mask)

        def select_pixels(img, t, a):
            _, b = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
            c_list, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res = np.zeros_like(img)
            for c in c_list:
                if cv2.contourArea(c) >= a:
                    cv2.drawContours(res, [c], -1, 255, -1)
            return res

        s1 = select_pixels(reduced, p['tg'], p['ta'])
        s2 = select_pixels(reduced, p['tg1'], p['ta1'])
        union = cv2.bitwise_or(s1, s2)
        final, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return (len(final) == 0), src


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高更盛 04T11 安装检测系统 (V2.0)")
        self.resize(1300, 850)
        self.cur_frame = None
        self.is_cam_source = False
        self.cap = None

        self.init_ui()
        self.init_camera()

    def init_ui(self):
        # 样式表设定
        self.setStyleSheet("""
            QMainWindow { background-color: #2D2D2D; }
            QGroupBox { color: #569CD6; font-weight: bold; border: 1px solid #454545; margin-top: 20px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QLabel { color: #D7D7D7; font-size: 10pt; }
            QLineEdit { background: #3C3C3C; color: #CE9178; border: 1px solid #555; border-radius: 2px; }
            QTextEdit { background: #1E1E1E; color: #DCDCDC; border: 1px solid #333; font-family: 'Consolas'; }
            QPushButton { background: #454545; color: white; border-radius: 3px; padding: 5px; }
            QPushButton:hover { background: #555555; }
            #BtnRun { background: #0E639C; font-size: 12pt; font-weight: bold; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # --- 左侧：参数与日志 ---
        left_side = QVBoxLayout()

        # GroupBox 1: AI
        self.gb_ai = QGroupBox("AI检测安装参数")
        g1 = QGridLayout(self.gb_ai)
        self.txt_conf = self.add_param(g1, "置信度阈值:", "0.7", 0)
        left_side.addWidget(self.gb_ai)

        # GroupBox 2: 亮白点
        self.gb_br = QGroupBox("较亮白点检测参数")
        g2 = QGridLayout(self.gb_br)
        self.txt_tg = self.add_param(g2, "白点阈值 Tgray:", "254", 0)
        self.txt_ta = self.add_param(g2, "最小面积 Tarea:", "45", 1)
        left_side.addWidget(self.gb_br)

        # GroupBox 3: 大白点
        self.gb_lg = QGroupBox("大面积白点检测参数")
        g3 = QGridLayout(self.gb_lg)
        self.txt_tg1 = self.add_param(g3, "白点阈值 Tgray1:", "240", 0)
        self.txt_ta1 = self.add_param(g3, "最小面积 Tarea1:", "200", 1)
        left_side.addWidget(self.gb_lg)

        # 按钮区
        left_side.addSpacing(10)
        self.btn_file = QPushButton("📂 打开本地图片")
        self.btn_file.clicked.connect(self.load_file)
        left_side.addWidget(self.btn_file)

        self.btn_snap = QPushButton("📸 相机实时抓拍")
        self.btn_snap.clicked.connect(self.snap_camera)
        left_side.addWidget(self.btn_snap)

        self.btn_run = QPushButton("🚀 开始一键检测")
        self.btn_run.setObjectName("BtnRun")
        self.btn_run.setFixedHeight(55)
        self.btn_run.clicked.connect(self.start_detection)
        left_side.addWidget(self.btn_run)

        # 结果看板
        self.res_lbl = QLabel("READY")
        self.res_lbl.setAlignment(Qt.AlignCenter)
        self.res_lbl.setFixedHeight(70)
        self.res_lbl.setStyleSheet("background: #333; font-size: 22pt; border-radius: 5px; color: #888;")
        left_side.addWidget(self.res_lbl)

        # 日志
        left_side.addWidget(QLabel("系统运行日志:"))
        self.log_txt = QTextEdit()
        left_side.addWidget(self.log_txt)

        layout.addLayout(left_side, 1)

        # --- 右侧：图像显示 ---
        right_side = QVBoxLayout()
        self.pic_box = QLabel("Waiting for Source...")
        self.pic_box.setAlignment(Qt.AlignCenter)
        self.pic_box.setStyleSheet("background: #000; border: 2px solid #333;")
        right_side.addWidget(self.pic_box)
        layout.addLayout(right_side, 3)

    def add_param(self, g, label, val, row):
        g.addWidget(QLabel(label), row, 0)
        e = QLineEdit(val)
        e.setFixedWidth(80)
        g.addWidget(e, row, 1)
        return e

    def log(self, msg):
        self.log_txt.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def init_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
            self.log("相机驱动已加载成功")
        else:
            self.log("未发现可用相机")

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.cur_frame = cv2.imread(path)
            self.is_cam_source = False
            self.show_img(self.cur_frame)
            self.log(f"已加载文件: {os.path.basename(path)}")

    def snap_camera(self):
        if self.cap and self.cap.isOpened():
            for _ in range(5): self.cap.grab()  # 清空缓存
            ret, frame = self.cap.read()
            if ret:
                self.cur_frame = frame
                self.is_cam_source = True
                self.show_img(self.cur_frame)
                self.log("相机抓拍成功")
        else:
            QMessageBox.warning(self, "错误", "相机未连接")

    def start_detection(self):
        if self.cur_frame is None:
            return

        try:
            params = {
                'conf': float(self.txt_conf.text()),
                'tg': int(self.txt_tg.text()), 'ta': int(self.txt_ta.text()),
                'tg1': int(self.txt_tg1.text()), 'ta1': int(self.txt_ta1.text())
            }
        except:
            QMessageBox.critical(self, "错误", "参数必须为有效数值")
            return

        self.btn_run.setEnabled(False)
        self.res_lbl.setText("TESTING")
        self.res_lbl.setStyleSheet("background: #555; color: #FFF; font-size: 22pt;")

        self.worker = WorkerThread(self.cur_frame, "best-seg.onnx", params, self.is_cam_source)
        self.worker.update_image.connect(self.show_img)
        self.worker.update_log.connect(self.log)
        self.worker.finished_res.connect(self.update_ui_result)
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.start()

    def show_img(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.pic_box.setPixmap(
            QPixmap.fromImage(qimg).scaled(self.pic_box.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_ui_result(self, ai, bd):
        ok = ai and bd
        self.res_lbl.setText("PASS" if ok else "FAIL")
        color = "#28A745" if ok else "#DC3545"
        self.res_lbl.setStyleSheet(f"background: {color}; color: white; font-weight: bold; font-size: 26pt;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec_())
