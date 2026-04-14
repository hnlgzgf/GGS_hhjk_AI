import sys
import cv2
import numpy as np
import onnxruntime as ort
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QListWidget,
    QFileDialog, QDoubleSpinBox, QSpinBox, QStyle
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os


class DetectionWorker(QThread):
    """后台检测线程：支持相机 or 本地图像"""
    finished_signal = pyqtSignal(bool, str, np.ndarray)

    def __init__(self, camera_id=0, onnx_path="best-seg.onnx", image_path=None):
        super().__init__()
        self.camera_id = camera_id
        self.onnx_path = onnx_path
        self.image_path = image_path
        self.cap = None
        self.running = False

        # 检查 ONNX 模型是否存在
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 模型未找到: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)

    def run(self):
        self.running = True

        # === 数据源选择：相机 或 本地图像 ===
        if self.image_path:
            frame = cv2.imread(self.image_path)
            if frame is None:
                self.finished_signal.emit(False, "❌ 无法读取本地图像", None)
                return
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2448)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)

            if not self.cap.isOpened():
                self.finished_signal.emit(False, "❌ 无法打开相机", None)
                return

            # 丢弃缓存帧
            for _ in range(10):
                self.cap.grab()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.finished_signal.emit(False, "❌ 相机读取失败", None)
                return

        # 1. 图像预处理：旋转180° + 裁剪ROI
        processed_img = self.rotate_and_crop(frame)

        # 2. AI 分割推理（ONNX）
        ai_result, result_img = self.run_yolo_inference(processed_img)

        # 3. 传统视觉检测
        bd_result = self.run_traditional_vision(processed_img)

        # 4. 综合判断
        final_result = ai_result and bd_result
        status = "OK" if final_result else "NG"

        self.finished_signal.emit(final_result, f"检测完成: {status}", result_img)

        if self.cap:
            self.cap.release()
        self.running = False

    def rotate_and_crop(self, src):
        """旋转180度并裁剪固定ROI"""
        image_rotate = cv2.rotate(src, cv2.ROTATE_180)
        row1, col1 = 684, 808
        row2, col2 = 1076, 1582
        h, w = image_rotate.shape[:2]
        col1 = max(0, col1)
        row1 = max(0, row1)
        width = min(col2 - col1, w - col1)
        height = min(row2 - row1, h - row1)
        roi = image_rotate[row1:row1 + height, col1:col1 + width]
        return roi

    def run_yolo_inference(self, img):
        """YOLO ONNX 推理（简化版：仅返回原图）"""
        try:
            # 注意：此处需根据你的 best-seg.onnx 实际输入输出调整！
            # 以下为通用占位逻辑，实际项目中应解析分割掩码
            input_size = (640, 640)  # 假设模型输入为640x640
            resized = cv2.resize(img, input_size)
            input_tensor = resized.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)

            outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})

            # TODO: 根据模型输出解析 mask 或 bbox，并叠加到 img 上
            # 此处为演示，直接返回原图
            return True, img.copy()
        except Exception as e:
            print(f"[AI推理错误] {e}")
            return False, img.copy()

    def run_traditional_vision(self, src_gray):
        """传统视觉检测逻辑（复刻原C# action函数）"""
        try:
            if len(src_gray.shape) == 3:
                gray = cv2.cvtColor(src_gray, cv2.COLOR_BGR2GRAY)
            else:
                gray = src_gray.copy()

            # Step 1: 初始二值化 + 闭运算
            _, bin_img = cv2.threshold(gray, 0, 140, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel_close)

            # Step 2: 连通域筛选（目标区域）
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            region_mask = np.zeros_like(bin_img)
            for c in contours:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                if 20000 <= area <= 300000 and 200 <= h <= 800:
                    cv2.drawContours(region_mask, [c], -1, 255, -1)

            # Step 3: 开运算去噪
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel_open)

            # Step 4: ReduceDomain（掩码应用）
            reduced = cv2.bitwise_and(gray, gray, mask=region_mask)

            # Step 5: 特别亮的小白点
            Tgray, Tarea = 254, 45
            _, bright_small = cv2.threshold(reduced, Tgray, 255, cv2.THRESH_BINARY)
            contours_small, _ = cv2.findContours(bright_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sel1 = np.zeros_like(gray)
            for c in contours_small:
                if cv2.contourArea(c) >= Tarea:
                    cv2.drawContours(sel1, [c], -1, 255, -1)

            # Step 6: 较暗的大白点
            Tgray1, Tarea1 = 240, 200
            _, bright_large = cv2.threshold(reduced, Tgray1, 255, cv2.THRESH_BINARY)
            contours_large, _ = cv2.findContours(bright_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sel2 = np.zeros_like(gray)
            for c in contours_large:
                if cv2.contourArea(c) >= Tarea1:
                    cv2.drawContours(sel2, [c], -1, 255, -1)

            # Step 7: 合并 + 膨胀
            union = cv2.bitwise_or(sel1, sel2)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            union = cv2.dilate(union, kernel_dilate)

            # Step 8: 最终判断（无缺陷轮廓 = OK）
            final_contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            color = (0, 255, 0) if len(final_contours) == 0 else (0, 0, 255)
            cv2.drawContours(display_img, final_contours, -1, color, 2)

            return len(final_contours) == 0

        except Exception as e:
            print(f"[传统视觉错误] {e}")
            return False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高更盛 04T11 安装检测系统 (Python + ONNX)")
        self.setGeometry(100, 100, 1200, 800)
        self.current_image_path = None

        # 全局样式
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QLabel { font-family: "Microsoft YaHei"; }
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:disabled { background-color: #cccccc; }
            QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 14px;
            }
            QListWidget {
                background: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ===== 左侧面板 =====
        left_layout = QVBoxLayout()

        # 检测结果
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("等待检测")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #333;")
        self.result_label.setFixedSize(220, 100)
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)

        # 模式提示
        self.mode_label = QLabel("模式: 相机")
        self.mode_label.setStyleSheet("color: #666; font-style: italic; font-size: 13px;")

        # AI参数
        ai_group = QGroupBox("AI参数")
        ai_layout = QVBoxLayout()
        ai_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setValue(0.7)
        self.conf_spin.setSingleStep(0.05)
        ai_layout.addWidget(self.conf_spin)
        ai_group.setLayout(ai_layout)

        # 传统视觉参数
        bd1_group = QGroupBox("较亮白点 (小面积)")
        bd1_layout = QVBoxLayout()
        bd1_layout.addWidget(QLabel("阈值:"));
        self.bd1_thresh = QSpinBox();
        self.bd1_thresh.setValue(254)
        bd1_layout.addWidget(self.bd1_thresh)
        bd1_layout.addWidget(QLabel("最小面积:"));
        self.bd1_area = QSpinBox();
        self.bd1_area.setValue(45)
        bd1_layout.addWidget(self.bd1_area)
        bd1_group.setLayout(bd1_layout)

        bd2_group = QGroupBox("大面积白点 (低亮度)")
        bd2_layout = QVBoxLayout()
        bd2_layout.addWidget(QLabel("阈值:"));
        self.bd2_thresh = QSpinBox();
        self.bd2_thresh.setValue(240)
        bd2_layout.addWidget(self.bd2_thresh)
        bd2_layout.addWidget(QLabel("最小面积:"));
        self.bd2_area = QSpinBox();
        self.bd2_area.setValue(200)
        bd2_layout.addWidget(self.bd2_area)
        bd2_group.setLayout(bd2_layout)

        # 按钮
        self.btn_open_image = QPushButton("📁 打开图像")
        self.btn_open_image.clicked.connect(self.open_local_image)
        self.btn_detect = QPushButton("🚀 一键检测")
        self.btn_detect.clicked.connect(self.start_detection)

        # 日志
        self.log_list = QListWidget()
        self.log_list.setMaximumHeight(150)

        left_layout.addWidget(result_group)
        left_layout.addWidget(self.mode_label)
        left_layout.addWidget(ai_group)
        left_layout.addWidget(bd1_group)
        left_layout.addWidget(bd2_group)
        left_layout.addWidget(self.btn_open_image)
        left_layout.addWidget(self.btn_detect)
        left_layout.addWidget(QLabel("日志:"))
        left_layout.addWidget(self.log_list)
        left_layout.addStretch()

        # ===== 右侧面板 =====
        right_layout = QVBoxLayout()
        self.display_label = QLabel("画面显示区")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background: #222; color: #aaa; border-radius: 8px; font-size: 16px;")
        self.display_label.setMinimumSize(640, 480)
        right_layout.addWidget(QLabel("检测画面"))
        right_layout.addWidget(self.display_label)

        # ===== 布局组合 =====
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 2)

    def open_local_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if file_path:
            self.current_image_path = file_path
            self.mode_label.setText("模式: 本地图像")
            self.mode_label.setStyleSheet("color: #0078d7; font-weight: bold;")

            img = cv2.imread(file_path)
            if img is not None:
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(q_img)
                self.display_label.setPixmap(
                    pixmap.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.log_list.addItem(f"✅ 已加载: {os.path.basename(file_path)}")
            else:
                self.log_list.addItem("❌ 图像加载失败！")
                self.current_image_path = None
                self.mode_label.setText("模式: 相机")
                self.mode_label.setStyleSheet("color: #666; font-style: italic;")
        else:
            pass  # 用户取消

    def start_detection(self):
        self.log_list.addItem("▶ 开始检测...")
        self.btn_detect.setEnabled(False)
        self.result_label.setText("检测中...")
        self.result_label.setStyleSheet("font-size: 36px; color: #ff8c00;")

        # 创建检测线程（自动判断模式）
        if self.current_image_path:
            self.detector = DetectionWorker(image_path=self.current_image_path)
        else:
            self.detector = DetectionWorker(camera_id=0)

        self.detector.finished_signal.connect(self.on_detection_finished)
        self.detector.start()

    def on_detection_finished(self, result, log_msg, result_img):
        self.log_list.addItem(log_msg)
        self.btn_detect.setEnabled(True)

        if result_img is not None:
            h, w, ch = result_img.shape
            bytes_per_line = ch * w
            q_img = QImage(result_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            self.display_label.setPixmap(
                pixmap.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

        status_text = "PASS" if result else "FAIL"
        color = "#4CAF50" if result else "#F44336"
        self.result_label.setText(status_text)
        self.result_label.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {color};")

    def closeEvent(self, event):
        if hasattr(self, 'detector') and self.detector.cap:
            self.detector.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())