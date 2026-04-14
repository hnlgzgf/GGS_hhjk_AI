import sys
import cv2
import numpy as np
import json
import os
import time  # 新增：用于测量耗时
from ctypes import c_ubyte, cast, POINTER
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QGridLayout, QSlider, QFormLayout, QFileDialog, QMessageBox
)
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from ultralytics import YOLO
from datetime import datetime
# ==================== 新增：PIL 用于中文绘制 ====================
from PIL import Image, ImageDraw, ImageFont
from FpPlc import FpPlc


class DetectionThread(QThread):
    result_ready = pyqtSignal(object)  # dict
    error = pyqtSignal(str)

    def __init__(self, window: "MainWindow", params: dict):
        super().__init__()
        self.window = window
        self.params = params

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            # 相机采图（不要在子线程触碰 UI 控件）
            if not self.window.camera:
                self.window.camera = HikCamera()
                self.window.camera.initialize()
            if not self.window.camera.is_opened:
                self.window.camera.open_device()

            frame = self.window.camera.get_frame()
            roi = self.window.rotate_and_crop(frame)
            if roi is None or roi.size == 0:
                raise Exception("ROI无效")

            ai_start = time.time()
            yolo_img, ai_ok = self.window.yolo_detect_with_params(
                roi.copy(),
                conf=self.params["conf"],
                ai_min_area=self.params["ai_min_area"],
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return

            trad_start = time.time()
            trad_img, trad_ok = self.window.traditional_detect_with_params(
                roi.copy(),
                tgray=self.params["tgray"],
                tarea=self.params["tarea"],
                tgray1=self.params["tgray1"],
                tarea1=self.params["tarea1"],
            )
            trad_time = time.time() - trad_start
            if self.isInterruptionRequested():
                return

            final_ok = bool(ai_ok and trad_ok)
            self.result_ready.emit(
                {
                    "yolo_img": yolo_img,
                    "ai_ok": ai_ok,
                    "ai_time": ai_time,
                    "trad_img": trad_img,
                    "trad_ok": trad_ok,
                    "trad_time": trad_time,
                    "final_ok": final_ok,
                }
            )
        except Exception as e:
            self.error.emit(str(e))


# ==================== 新增：侧面检测独立线程 ====================
class SideDetectionThread(QThread):
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            # 相机二独立采集（枚举 index=1）
            if not self.window.camera_side:
                self.window.camera_side = HikCamera()
                self.window.camera_side.initialize()
            if not self.window.camera_side.is_opened:
                self.window.camera_side.open_device(index=1)

            frame = self.window.camera_side.get_frame()
            roi = self.window.rotate_and_crop_side(frame)
            if roi is None or roi.size == 0:
                raise Exception("侧面ROI无效")

            start = time.time()
            side_img, side_ok = self.window.side_detect(roi.copy())
            side_time = time.time() - start
            if self.isInterruptionRequested():
                return

            self.result_ready.emit({
                "side_img": side_img,
                "side_ok": side_ok,
                "side_time": side_time,
            })
        except Exception as e:
            self.error.emit(str(e))


# ==================== 海康 SDK 导入 ====================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MvImport'))
try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    from MvErrorDefine_const import *
except ImportError:
    print("缺少海康SDK文件，请将 MvImport 目录下的文件复制到当前目录")
    sys.exit(1)

# ==================== 全局模型加载 ====================
CONFIG_PATH = "config.json"
MODEL_PATH = "best-seg.onnx"
SEG_MODEL_PATH = "best_seg_DW.onnx"
SIDE_MODEL_PATH = "best_CM.onnx"          # 新增：侧面检测模型

model = YOLO(MODEL_PATH)
seg_model = YOLO(SEG_MODEL_PATH, task='segment')
side_model = YOLO(SIDE_MODEL_PATH)        # 新增

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
    font_path = r"C:\Windows\Fonts\simfang.ttf"  # 如无此字体可改为 simhei.ttf

    cv_img = img.copy()
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"警告：未找到字体 {font_path}，使用默认字体（中文可能乱码）")

    rgb_color = color[::-1]  # BGR -> RGB
    draw.text(position, text, font=font, fill=rgb_color, stroke_width=thickness, stroke_fill=rgb_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==================== 海康相机采集类（参考 hik.py 实现，支持 index） ====================
class HikCamera:
    def __init__(self):
        self.cam = None
        self.is_opened = False
        self.device_list = MV_CC_DEVICE_INFO_LIST()  # 与 hik 一致：单例 device_list

    def initialize(self):
        print("正在初始化海康SDK...")
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            raise Exception(f"SDK初始化失败: 0x{ret:X} (请检查 MvCameraControl.dll)")
        print("SDK初始化成功")

    def enum_devices(self):
        """与 hik.py 一致：直接枚举 MV_GIGE_DEVICE | MV_USB_DEVICE"""
        print("正在枚举相机...")
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != MV_OK:
            raise Exception(f"枚举设备失败: 0x{ret:X}")
        if self.device_list.nDeviceNum == 0:
            raise Exception("未检测到海康相机，请检查连接与驱动")
        print(f"检测到 {self.device_list.nDeviceNum} 个相机")
        return self.device_list

    def open_device(self, index=0):
        if self.device_list.nDeviceNum == 0:
            self.enum_devices()
        
        # 与 hik.py 一致：用 cast 正确获取设备信息
        dev = cast(self.device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        print(f"尝试打开设备 {index}...")
        
        if self.cam:
            print("释放之前的相机资源...")
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
        
        self.cam = MvCamera()
        
        ret = self.cam.MV_CC_CreateHandle(dev)
        if ret != MV_OK:
            raise Exception(f"创建设备句柄失败: 0x{ret:X}")
        print("设备句柄创建成功")
        
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            self.cam.MV_CC_DestroyHandle()
            raise Exception(f"打开设备失败: 0x{ret:X} (可能被其他程序占用)")
        print("设备打开成功")
        
        # 与 hik 一致：不设置 AcquisitionMode，只设置 TriggerMode
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != MV_OK:
            print("警告：设置触发模式失败，尝试继续")
        
        # 与 hik 一致：优先 RGB8，失败再 BGR8，都失败用默认（不报错）
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
        if ret != MV_OK:
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BGR8_Packed)
        if ret != MV_OK:
            print("使用相机默认像素格式")
        
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            raise Exception(f"开始采集失败: 0x{ret:X}")
        print("开始采集成功")
        
        self.is_opened = True
        print(f"相机 {index} 打开成功")

    def get_frame(self, timeout=1000):
        if not self.is_opened or not self.cam:
            raise Exception("相机未打开")
        
        st_frame_info = MV_FRAME_OUT_INFO_EX()
        buffer_size = 50 * 1024 * 1024  # 50MB，与 hik 一致
        data_buf = (c_ubyte * buffer_size)()
        
        # 与 hik 一致：传 len(data_buf)
        ret = self.cam.MV_CC_GetOneFrameTimeout(data_buf, len(data_buf), st_frame_info, timeout)
        if ret != MV_OK:
            if ret in (0x80000002, 0x80000003):  # 超时/无数据，重试一次
                ret = self.cam.MV_CC_GetOneFrameTimeout(data_buf, len(data_buf), st_frame_info, timeout)
            if ret != MV_OK:
                raise Exception(f"获取图像失败: 0x{ret:X}")
        
        w = st_frame_info.nWidth
        h = st_frame_info.nHeight
        pt = st_frame_info.enPixelType
        
        # 与 hik 一致：彩色优先 (RGB8, BGR8, Mono8)
        if pt == PixelType_Gvsp_RGB8_Packed:
            img = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3)
            img = img.reshape((h, w, 3))
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pt == PixelType_Gvsp_BGR8_Packed:
            img = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3)
            img = img.reshape((h, w, 3))
            frame = img
        elif pt == PixelType_Gvsp_Mono8:
            img = np.frombuffer(data_buf, dtype=np.uint8, count=w * h)
            img = img.reshape((h, w))
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise Exception(f"不支持的像素格式: 0x{pt:X}")
        
        return frame

    def close(self):
        if self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        self.cam = None
        self.is_opened = False

    def __del__(self):
        self.close()

# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高更盛04T11安装检测 - 双相机版")
        icon_path = "1.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("警告：未找到 1.ico 文件，使用默认图标")
        self.resize(1800, 900)

        self.total_count = 0
        self.good_count = 0
        self.camera = None
        self.camera_side = None          # 新增：相机二
        self.plc = None
        self.plc_setting = ""
        self._plc_prev_trigger = False
        self._plc_busy = False
        self._plc_prev_trigger_side = False   # 新增
        self._plc_busy_side = False           # 新增
        self._detect_thread = None
        self._side_thread = None              # 新增

        self.init_ui()
        self.load_config()
        self._init_plc_trigger()

    def _init_plc_trigger(self):
        """参考 Form1.cs / FpPlc.py：R[0] 上升沿触发主检测；R[1] 上升沿触发侧面检测"""
        try:
            if not self.plc_setting:
                self.log_text.append("PLC未配置（config.json: plc_setting），跳过PLC触发")
                return

            self.plc = FpPlc()
            if not self.plc.open(self.plc_setting):
                self.log_text.append(f"PLC连接失败: {self.plc_setting}")
                self.plc = None
                return

            self.log_text.append(f"PLC已连接: {self.plc_setting}（监听 R[0] 主检测 + R[1] 侧面检测）")
            self._plc_timer = QTimer(self)
            self._plc_timer.timeout.connect(self._poll_plc_trigger)
            self._plc_timer.start(20)
        except Exception as e:
            self.log_text.append(f"PLC初始化异常: {e}")
            self.plc = None

    def _poll_plc_trigger(self):
        if not self.plc:
            return
        # 主检测
        if not self._plc_busy:
            trig = self.plc.get_r(0)
            if trig and not self._plc_prev_trigger:
                self.log_text.append("PLC触发：主检测一次")
                self._start_detection(source="PLC")
            self._plc_prev_trigger = trig
        # 侧面检测（独立）
        if not self._plc_busy_side:
            trig_side = self.plc.get_r(1)
            if trig_side and not self._plc_prev_trigger_side:
                self.log_text.append("PLC触发：侧面检测一次")
                self._start_side_detection(source="PLC")
            self._plc_prev_trigger_side = trig_side

    def _start_detection(self, source: str = "MANUAL"):
        """启动一次主检测。检测进行中会忽略新的触发"""
        if self._plc_busy:
            self.log_text.append(f"{source} 触发被忽略：正在检测中")
            return
        self._plc_busy = True

        # 从 UI 读取参数（只能在主线程读）
        try:
            params = {
                "conf": float(self.conf_edit.text()),
                "tgray": int(self.tgray_edit.text()),
                "tarea": int(self.tarea_edit.text()),
                "tgray1": int(self.tgray1_edit.text()),
                "tarea1": int(self.tarea1_edit.text()),
                "ai_min_area": int(self.ai_area_edit.text()),
            }
        except Exception as e:
            self.log_text.append(f"参数读取失败: {e}")
            self._plc_busy = False
            return

        self.btn_detect.setEnabled(False)
        self.log_text.append(f"{source} 启动主检测...")

        self._detect_thread = DetectionThread(self, params)
        self._detect_thread.result_ready.connect(self._on_detection_result)
        self._detect_thread.error.connect(self._on_detection_error)
        self._detect_thread.finished.connect(lambda: self.btn_detect.setEnabled(True))
        self._detect_thread.start()

    def _start_side_detection(self, source: str = "MANUAL"):
        """启动一次侧面检测（独立互斥）"""
        if self._plc_busy_side:
            self.log_text.append(f"{source} 侧面触发被忽略：正在检测中")
            return
        self._plc_busy_side = True

        self.btn_side_detect.setEnabled(False)
        self.log_text.append(f"{source} 启动侧面检测...")

        self._side_thread = SideDetectionThread(self)
        self._side_thread.result_ready.connect(self._on_side_detection_result)
        self._side_thread.error.connect(self._on_side_detection_error)
        self._side_thread.finished.connect(lambda: self.btn_side_detect.setEnabled(True))
        self._side_thread.start()

    def _on_detection_result(self, r: dict):
        try:
            yolo_img = draw_chinese_text(r["yolo_img"], "AI检测安装到位", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.update_result_text(self.text_ai, bool(r["ai_ok"]))

            trad_img = draw_chinese_text(r["trad_img"], "白点灰尘检测", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_trad, bool(r["trad_ok"]))

            final_ok = bool(r["final_ok"])
            self.update_result_text(self.text_final, final_ok)
            self.update_stats(final_ok)
            self._plc_report_result(final_ok)

            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 检测完成 | AI安装检测：{'OK' if r['ai_ok'] else 'NG'} | "
                f"白点灰尘检测：{'OK' if r['trad_ok'] else 'NG'} | 综合：{'OK' if final_ok else 'NG'} | "
                f"AI检测安装到位耗时：{r['ai_time']:.3f}s | 白点灰尘检测耗时：{r['trad_time']:.3f}s"
            )
        finally:
            self._plc_busy = False
            self._detect_thread = None

    def _on_detection_error(self, msg: str):
        try:
            self.log_text.append(f"主检测错误: {msg}")
        finally:
            self._plc_busy = False
            self._detect_thread = None

    def _on_side_detection_result(self, r: dict):
        try:
            side_img = draw_chinese_text(r["side_img"], "AI侧面检测", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)
            self.show_side.setPixmap(cv2_to_qpixmap(side_img))
            self.update_result_text(self.text_side, bool(r["side_ok"]))

            self._plc_report_side_result(r["side_ok"])

            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 侧面检测完成 | "
                f"AI侧面检测：{'OK' if r['side_ok'] else 'NG'} | 耗时：{r['side_time']:.3f}s"
            )
        finally:
            self._plc_busy_side = False
            self._side_thread = None

    def _on_side_detection_error(self, msg: str):
        try:
            self.log_text.append(f"侧面检测错误: {msg}")
        finally:
            self._plc_busy_side = False
            self._side_thread = None

    def _plc_report_result(self, final_ok: bool):
        """OK: R0502=0; NG: R0502=1; 完成信号 R0501 脉冲 100ms。"""
        if not self.plc:
            return
        try:
            # 先写结果位（与 Form1.cs 逻辑一致：NG=1, OK=0）
            self.plc.write_bit("R0502", (not final_ok))
            # 再给完成脉冲
            self.plc.write_bit("R0501", True)
            QTimer.singleShot(100, lambda: self.plc.write_bit("R0501", False) if self.plc else None)
        except Exception as e:
            self.log_text.append(f"PLC回写异常: {e}")

    # ==================== 新增：侧面PLC回写 ====================
    def _plc_report_side_result(self, side_ok: bool):
        """OK: R0504=0; NG: R0504=1; 完成信号 R0503 脉冲 100ms。"""
        if not self.plc:
            return
        try:
            self.plc.write_bit("R0504", (not side_ok))
            self.plc.write_bit("R0503", True)
            QTimer.singleShot(100, lambda: self.plc.write_bit("R0503", False) if self.plc else None)
        except Exception as e:
            self.log_text.append(f"PLC侧面回写异常: {e}")

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

        # ==================== 三栏图像显示 ====================
        image_layout = QHBoxLayout()
        self.show_window = QLabel()      # 主AI
        self.show_side = QLabel()        # 新增：AI侧面检测
        self.picture_box1 = QLabel()     # 白点

        for label in [self.show_window, self.show_side, self.picture_box1]:
            label.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
            label.setAlignment(Qt.AlignCenter)
            label.setScaledContents(True)

        image_layout.addWidget(self.show_window)
        image_layout.addWidget(self.show_side)
        image_layout.addWidget(self.picture_box1)
        main_layout.addLayout(image_layout, stretch=6)

        mid_layout = QHBoxLayout()

        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout()

        # AI安装检测
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

        # 白点检测
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

        # 综合
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

        # ==================== 新增：侧面检测结果 ====================
        side_layout = QHBoxLayout()
        side_label = QLabel("侧面检测:")
        side_label.setFont(QFont("Arial", 20, QFont.Bold))
        side_label.setAlignment(Qt.AlignCenter)
        side_label.setFixedWidth(80)
        self.text_side = QLabel("--")
        self.text_side.setAlignment(Qt.AlignCenter)
        self.text_side.setFixedHeight(60)
        self.text_side.setFixedWidth(120)
        self.text_side.setFont(QFont("Arial", 20, QFont.Bold))
        self.text_side.setStyleSheet("background-color: #f0f0f0; color: #333; border-radius: 8px;")
        side_layout.addWidget(side_label)
        side_layout.addWidget(self.text_side)
        result_layout.addLayout(side_layout)

        result_group.setLayout(result_layout)
        result_group.setFixedWidth(320)
        mid_layout.addWidget(result_group)

        # 参数调节组（完全保持原样）
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
        self.ai_area_edit = QLineEdit("3000")
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
        self.btn_side_detect = QPushButton("一键侧面检测")   # 新增
        self.btn_image_detect = QPushButton("打开图片检测")
        self.btn_save = QPushButton("保存参数")
        self.btn_load = QPushButton("载入参数")

        self.btn_detect.setFixedSize(160, 80)
        self.btn_side_detect.setFixedSize(160, 80)
        self.btn_image_detect.setFixedSize(160, 80)

        self.btn_detect.clicked.connect(lambda: self._start_detection(source="MANUAL"))
        self.btn_side_detect.clicked.connect(lambda: self._start_side_detection(source="MANUAL"))  # 新增
        self.btn_image_detect.clicked.connect(self.on_image_detect)
        self.btn_save.clicked.connect(self.save_config)
        self.btn_load.clicked.connect(self.load_config)

        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_side_detect)  # 新增按钮
        btn_layout.addWidget(self.btn_image_detect)
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
        space_shortcut.activated.connect(lambda: self._start_detection(source="MANUAL"))

        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(lambda: self._start_detection(source="MANUAL"))

        enter_shortcut2 = QShortcut(QKeySequence(Qt.Key_Enter), self)
        enter_shortcut2.activated.connect(lambda: self._start_detection(source="MANUAL"))

    def save_config(self):
        config = {
            "conf": self.conf_edit.text(),
            "tgray": self.tgray_edit.text(),
            "tarea": self.tarea_edit.text(),
            "tgray1": self.tgray1_edit.text(),
            "tarea1": self.tarea1_edit.text(),
            "ai_min_area": self.ai_area_edit.text(),
            "plc_setting": self.plc_setting,
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
            self.ai_area_edit.setText(config.get("ai_min_area", "3000"))
            self.plc_setting = config.get("plc_setting", "")

            self.conf_slider.setValue(int(float(config.get("conf", "0.7")) * 100))
            self.tgray_slider.setValue(int(config.get("tgray", "254")))
            self.tarea_slider.setValue(int(config.get("tarea", "45")))
            self.tgray1_slider.setValue(int(config.get("tgray1", "240")))
            self.tarea1_slider.setValue(int(config.get("tarea1", "200")))
            self.ai_area_slider.setValue(int(config.get("ai_min_area", "3000")))
            self.log_text.append("参数载入成功！")
        except Exception as e:
            self.log_text.append(f"载入失败: {str(e)}")

    def grab_two_frames(self):
        """使用海康相机采集图像"""
        try:
            if not self.camera:
                self.camera = HikCamera()
                self.camera.initialize()
            
            if not self.camera.is_opened:
                self.camera.open_device()

            self.log_text.append("采集图像...")
            frame = self.camera.get_frame()
            self.log_text.append("图像采集完成")
            return frame
        except Exception as e:
            self.log_text.append(f"采集错误: {str(e)}")
            # 尝试重新初始化相机
            if self.camera:
                self.camera.close()
                self.camera = None
            raise

    def rotate_and_crop(self, src: np.ndarray) -> np.ndarray:
        rotated = cv2.rotate(src, cv2.ROTATE_180)
        row1, col1, row2, col2 = 684, 808, 1076, 1582
        roi = rotated[row1:row2, col1:col2]
        return roi.copy()

    # ==================== 新增：侧面ROI裁剪 ====================
    def rotate_and_crop_side(self, src: np.ndarray) -> np.ndarray:
        rotated = cv2.rotate(src, cv2.ROTATE_180)
        row1, col1, row2, col2 = 680, 800, 1080, 1580   # 按需求
        roi = rotated[row1:row2, col1:col2]
        return roi.copy()

    def yolo_detect_with_params(self, frame: np.ndarray, conf: float, ai_min_area: int):
        results = model(frame, conf=conf, iou=0.9, verbose=False)

        plot_rgb = results[0].plot()
        plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)

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

    def yolo_detect(self, frame: np.ndarray):
        conf = float(self.conf_edit.text())
        ai_min_area = int(self.ai_area_edit.text())
        return self.yolo_detect_with_params(frame, conf=conf, ai_min_area=ai_min_area)

    def traditional_detect_with_params(self, src: np.ndarray, tgray: int, tarea: int, tgray1: int, tarea1: int):
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

    def traditional_detect(self, src: np.ndarray):
        tgray = int(self.tgray_edit.text())
        tarea = int(self.tarea_edit.text())
        tgray1 = int(self.tgray1_edit.text())
        tarea1 = int(self.tarea1_edit.text())
        return self.traditional_detect_with_params(src, tgray=tgray, tarea=tarea, tgray1=tgray1, tarea1=tarea1)

    # ==================== 新增：侧面检测（参考AI检测安装到位，仅判断掩膜是否为空） ====================
    def side_detect(self, frame: np.ndarray):
        results = side_model(frame, conf=0.5, iou=0.9, verbose=False)
        plot_rgb = results[0].plot()
        plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)

        # 掩膜为空 = OK，不为空 = NG
        masks = results[0].masks
        has_mask = masks is not None and len(masks.xy) > 0
        side_ok = not has_mask
        return plot_bgr, side_ok

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
            frame = self.grab_two_frames()
            roi = self.rotate_and_crop(frame)
            if roi.size == 0:
                self.log_text.append("ROI无效")
                return

            # ==================== AI检测耗时测量 ====================
            ai_start = time.time()
            yolo_img, ai_ok = self.yolo_detect(roi.copy())
            ai_time = time.time() - ai_start
            yolo_img = draw_chinese_text(yolo_img, "AI检测安装到位", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.update_result_text(self.text_ai, ai_ok)

            # ==================== 白点检测耗时测量 ====================
            trad_start = time.time()
            trad_img, trad_ok = self.traditional_detect(roi.copy())
            trad_time = time.time() - trad_start
            trad_img = draw_chinese_text(trad_img, "白点灰尘检测", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_trad, trad_ok)

            final_ok = ai_ok and trad_ok
            self.update_result_text(self.text_final, final_ok)
            self.update_stats(final_ok)
            self._plc_report_result(final_ok)
         
            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 检测完成 | AI安装检测：{'OK' if ai_ok else 'NG'} | "
                f"白点灰尘检测：{'OK' if trad_ok else 'NG'} | 综合：{'OK' if final_ok else 'NG'} | "
                f"AI检测安装到位耗时：{ai_time:.3f}s | 白点灰尘检测耗时：{trad_time:.3f}s"
            )
            self._plc_busy = False
        except Exception as e:
            self.log_text.append(f"错误: {str(e)}")
            self._plc_busy = False

    # ==================== 新增：打开本地图片进行检测（保持原样，仅支持主流程） ====================
    def on_image_detect(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            self.log_text.append("无法加载图片文件")
            return

        # ==================== AI检测耗时测量 ====================
        ai_start = time.time()
        yolo_img, ai_ok = self.yolo_detect(frame.copy())
        ai_time = time.time() - ai_start
        yolo_img = draw_chinese_text(yolo_img, "AI检测安装到位", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)
        self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
        self.update_result_text(self.text_ai, ai_ok)

        # ==================== 白点检测耗时测量 ====================
        trad_start = time.time()
        trad_img, trad_ok = self.traditional_detect(frame.copy())
        trad_time = time.time() - trad_start
        trad_img = draw_chinese_text(trad_img, "白点灰尘检测", (10, 10), font_size=16, color=(255, 0, 0), thickness=1)
        self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
        self.update_result_text(self.text_trad, trad_ok)

        final_ok = ai_ok and trad_ok
        self.update_result_text(self.text_final, final_ok)
        self.update_stats(final_ok)

        self.log_text.append(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片检测完成 | 文件: {os.path.basename(file_path)} | "
            f"AI: {'OK' if ai_ok else 'NG'} | 白点: {'OK' if trad_ok else 'NG'} | 综合: {'OK' if final_ok else 'NG'} | "
            f"AI检测安装到位耗时：{ai_time:.3f}s | 白点灰尘检测耗时：{trad_time:.3f}s"
        )

    def clear_stats(self):
        self.total_count = 0
        self.good_count = 0
        self.label_total.setText("0")
        self.label_good.setText("0")
        self.label_rate.setText("0.00%")
        self.log_text.append("【统计】生产统计数据已清零")

    def closeEvent(self, event):
        """窗口关闭时释放相机资源"""
        if self._detect_thread and self._detect_thread.isRunning():
            self._detect_thread.requestInterruption()
            self._detect_thread.wait(1500)
        if self._side_thread and self._side_thread.isRunning():
            self._side_thread.requestInterruption()
            self._side_thread.wait(1500)
        if self.camera:
            self.camera.close()
        if self.camera_side:
            self.camera_side.close()
        if self.plc:
            try:
                self.plc.close()
            except Exception:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())