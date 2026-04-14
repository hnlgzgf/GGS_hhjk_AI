# -*- coding: utf-8 -*-
import sys
import multiprocessing
if getattr(sys, "frozen", False):
    multiprocessing.freeze_support()
import cv2
import numpy as np
import json
import os
import tempfile
import atexit
import time
import re
import torch
import onnxruntime as ort
from ctypes import c_ubyte, cast, POINTER
from typing import Optional
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QGridLayout, QSlider, QFormLayout, QFileDialog, QMessageBox, QTabWidget, QComboBox, QFrame, QLayout, QSizePolicy, QToolButton, QCheckBox)
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from ultralytics import YOLO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import serial
import threading
import queue
import msvcrt

def _is_pyinstaller_mp_child() -> bool:
    # Frozen multiprocessing workers relaunch this exe with special argv flags.
    # They should not initialize GUI.
    for arg in sys.argv[1:]:
        a = str(arg).lower()
        if a.startswith("--multiprocessing-fork"):
            return True
        if a.startswith("--multiprocessing-spawn"):
            return True
    return False

_instance_lock_file = None

def _acquire_single_instance_lock() -> bool:
    global _instance_lock_file
    lock_path = os.path.join(tempfile.gettempdir(), "main_new_hik_CC_beifen3.lock")
    _instance_lock_file = open(lock_path, "a+")
    try:
        msvcrt.locking(_instance_lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False

def _release_single_instance_lock():
    global _instance_lock_file
    if _instance_lock_file is None:
        return
    try:
        _instance_lock_file.seek(0)
        msvcrt.locking(_instance_lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    try:
        _instance_lock_file.close()
    except Exception:
        pass
    _instance_lock_file = None

# ================== MEWTOCOL 协议实现 =================
class MewtocolPLC:
    def __init__(self):
        self.serial = None
        self.lock = threading.Lock()
        self.is_connected = False
        self.timeout_count = 0
        self.crc_err_count = 0
        self.reconnect_count = 0
        self.poll_interval = 0.03
        self.port_settings = {}
        self.stop_event = threading.Event()
        self.read_cache = {}  # 缓存触点状态

    def _calc_bcc(self, cmd: str) -> str:
        bcc = 0
        for char in cmd:
            bcc ^= ord(char)
        return f"{bcc:02X}"

    def open(self, setting_str: str) -> bool:
        try:
            parts = [p.strip() for p in setting_str.split(",")]
            if len(parts) < 6: return False
            
            port = parts[0]
            baud = int(parts[1])
            bytesize = int(parts[2])
            stopbits = serial.STOPBITS_ONE if int(parts[3]) == 1 else serial.STOPBITS_TWO
            parity_map = {"0": serial.PARITY_NONE, "1": serial.PARITY_ODD, "2": serial.PARITY_EVEN}
            parity = parity_map.get(parts[4], serial.PARITY_NONE)
            self.poll_interval = int(parts[5]) / 1000.0

            self.port_settings = {
                "port": port, "baudrate": baud, "bytesize": bytesize,
                "parity": parity, "stopbits": stopbits, "timeout": 0.5
            }
            return self.connect()
        except Exception as e:
            print(f"PLC配置解析失败: {e}")
            return False

    def connect(self) -> bool:
        with self.lock:
            try:
                if self.serial and self.serial.is_open:
                    self.serial.close()
                self.serial = serial.Serial(**self.port_settings)
                self.is_connected = True
                return True
            except Exception as e:
                print(f"PLC连接失败: {e}")
                self.is_connected = False
                return False

    def _send_command(self, cmd_body: str) -> Optional[str]:
        if not self.is_connected or not self.serial: return None
        # 使用 EE 站号（全局站号），大部分松下 PLC 默认都会响应
        station = "EE" 
        full_cmd = f"%{station}#{cmd_body}{self._calc_bcc('%' + station + '#' + cmd_body)}\r"
        with self.lock:
            try:
                self.serial.flushInput()
                self.serial.write(full_cmd.encode('ascii'))
                
                # 读取响应
                response = self.serial.read_until(b'\r').decode('ascii', errors='ignore')
                
                if not response:
                    self.timeout_count += 1
                    return None
                
                # 打印接收到的数据，方便调试
                if "RCS" in cmd_body:
                    print(f"PLC 响应 ({cmd_body}): {response.strip()}")
                
                # 校验 BCC
                if len(response) < 4: return None
                res_body = response[:-3] 
                res_bcc = response[-3:-1]
                if self._calc_bcc(res_body) != res_bcc:
                    self.crc_err_count += 1
                    return None
                
                return response
            except Exception as e:
                print(f"PLC通讯异常: {e}")
                self.is_connected = False
                return None

    def read_bit(self, addr: str) -> bool:
        # Panasonic MEWTOCOL 读单个触点: %01#RCS[类型][地址][BCC]\r
        # 补全触点类型 'R'
        clean_addr = addr.upper()
        if not clean_addr.startswith("R"):
            clean_addr = "R" + clean_addr
        
        # 保持地址至少 4 位（例如 R1205）
        # 如果地址是 R0，补成 R0000
        prefix = clean_addr[0]
        digits = clean_addr[1:]
        if len(digits) < 4:
            digits = digits.zfill(4)
        cmd_addr = f"{prefix}{digits}"
        
        res = self._send_command(f"RCS{cmd_addr}")
        if res and "$" in res:
            # 响应格式: %01$RC<状态><BCC>\r
            # 状态位在 $RC 后面一位
            idx = res.find("$RC") + 3
            status = res[idx:idx+1]
            val = (status == "1")
            self.read_cache[addr] = val
            return val
        return self.read_cache.get(addr, False)

    def write_bit(self, addr: str, val: bool):
        # Panasonic MEWTOCOL 写单个触点: %01#WCS[类型][地址][状态][BCC]\r
        clean_addr = addr.upper()
        if not clean_addr.startswith("R"):
            clean_addr = "R" + clean_addr
        
        prefix = clean_addr[0]
        digits = clean_addr[1:]
        if len(digits) < 4:
            digits = digits.zfill(4)
        cmd_addr = f"{prefix}{digits}"
        
        status = "1" if val else "0"
        self._send_command(f"WCS{cmd_addr}{status}")

    def close(self):
        self.is_connected = False
        with self.lock:
            if self.serial:
                self.serial.close()
                self.serial = None

    def get_status(self) -> dict:
        return {
            "connected": self.is_connected,
            "timeout": self.timeout_count,
            "crc_err": self.crc_err_count,
            "reconnects": self.reconnect_count
        }


# 检测线程（相机一：正面检测）
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
            if not self.window.camera:
                self.window.camera = HiKCamera()
                self.window.camera.initialize()
            if not self.window.camera.is_opened:
                ip = self.window.get_front_cam_ip()
                self.window.camera.open_device_by_ip(ip)
            frame = self.window.camera.get_frame()
            roi = self.window.crop_front_roi(
                frame,
                self.params.get("front_roi"),
                rotate_angle=self.params.get("front_rotate_angle", 0.0),
                flip_horizontal=bool(self.params.get("front_flip_horizontal", False)),
                flip_vertical=bool(self.params.get("front_flip_vertical", False)),
            )
            if roi is None or roi.size == 0:
                raise Exception("ROI无效")
            save_raw = bool(self.params.get("save_raw_image", False))
            save_roi = bool(self.params.get("save_roi_image", False))
            saved_raw_path, saved_roi_path, save_error = self.window.try_save_detection_images(
                station="front",
                raw_img=frame,
                roi_img=roi,
                save_raw=save_raw,
                save_roi=save_roi,
            )
            gray_ng, mean_gray = self.window.check_mean_gray_ng(roi)
            if gray_ng:
                self.result_ready.emit(
                    {
                        "yolo_img": roi.copy(),
                        "ai_ok": False,
                        "ai_time": 0.0,
                        "trad_img": roi.copy(),
                        "trad_ok": False,
                        "trad_time": 0.0,
                        "final_ok": False,
                        "saved_raw_path": saved_raw_path,
                        "saved_roi_path": saved_roi_path,
                        "save_error": save_error,
                        "gray_ng": True,
                        "mean_gray": mean_gray,
                    }
                )
                return
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
                    "saved_raw_path": saved_raw_path,
                    "saved_roi_path": saved_roi_path,
                    "save_error": save_error,
                }
            )
        except Exception as e:
            self.error.emit(str(e))

# 侧面检测线程（相机二）
class SideDetectionThread(QThread):
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
            if not self.window.camera2:
                self.window.camera2 = HiKCamera()
                self.window.camera2.initialize()
            if not self.window.camera2.is_opened:
                ip = self.window.get_side_cam_ip()
                self.window.camera2.open_device_by_ip(ip)
            frame = self.window.camera2.get_frame()
            roi = self.window.crop_side_roi(
                frame,
                self.params.get("side_roi"),
                rotate_angle=self.params.get("side_rotate_angle", 0.0),
                flip_horizontal=bool(self.params.get("side_flip_horizontal", False)),
                flip_vertical=bool(self.params.get("side_flip_vertical", False)),
            )
            if roi is None or roi.size == 0:
                raise Exception("侧面ROI无效")
            save_raw = bool(self.params.get("save_raw_image", False))
            save_roi = bool(self.params.get("save_roi_image", False))
            saved_raw_path, saved_roi_path, save_error = self.window.try_save_detection_images(
                station="side",
                raw_img=frame,
                roi_img=roi,
                save_raw=save_raw,
                save_roi=save_roi,
            )
            gray_ng, mean_gray = self.window.check_mean_gray_ng(roi)
            if gray_ng:
                self.result_ready.emit(
                    {
                        "side_img": roi.copy(),
                        "side_ok": False,
                        "ai_time": 0.0,
                        "saved_raw_path": saved_raw_path,
                        "saved_roi_path": saved_roi_path,
                        "save_error": save_error,
                        "gray_ng": True,
                        "mean_gray": mean_gray,
                    }
                )
                return
            ai_start = time.time()
            side_img, side_ok = self.window.side_detect_with_params(
                roi.copy(),
                conf=self.params["side_conf"],
                min_area=self.params["side_min_area"],
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return
            self.result_ready.emit(
                {
                    "side_img": side_img,
                    "side_ok": side_ok,
                    "ai_time": ai_time,
                    "saved_raw_path": saved_raw_path,
                    "saved_roi_path": saved_roi_path,
                    "save_error": save_error,
                }
            )
        except Exception as e:
            self.error.emit(str(e))

# 图片正面检测线程（避免主线程阻塞）
class ImageDetectionThread(QThread):
    result_ready = pyqtSignal(object)  # dict
    error = pyqtSignal(str)

    def __init__(self, window: "MainWindow", file_path: str, params: dict):
        super().__init__()
        self.window = window
        self.file_path = file_path
        self.params = params

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            frame = cv2.imread(self.file_path)
            if frame is None:
                raise Exception("无法加载图片文件")
            gray_ng, mean_gray = self.window.check_mean_gray_ng(frame)
            if gray_ng:
                self.result_ready.emit(
                    {
                        "file_path": self.file_path,
                        "yolo_img": frame.copy(),
                        "ai_ok": False,
                        "ai_time": 0.0,
                        "trad_img": frame.copy(),
                        "trad_ok": False,
                        "trad_time": 0.0,
                        "final_ok": False,
                        "gray_ng": True,
                        "mean_gray": mean_gray,
                    }
                )
                return
            ai_start = time.time()
            yolo_img, ai_ok = self.window.yolo_detect_with_params(
                frame.copy(),
                conf=self.params["conf"],
                ai_min_area=self.params["ai_min_area"],
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return
            trad_start = time.time()
            trad_img, trad_ok = self.window.traditional_detect_with_params(
                frame.copy(),
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
                    "file_path": self.file_path,
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

# 图片侧面检测线程（避免主线程阻塞）
class ImageSideDetectionThread(QThread):
    result_ready = pyqtSignal(object)  # dict
    error = pyqtSignal(str)

    def __init__(self, window: "MainWindow", file_path: str, side_conf: float, side_min_area: int):
        super().__init__()
        self.window = window
        self.file_path = file_path
        self.side_conf = side_conf
        self.side_min_area = side_min_area

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            frame = cv2.imread(self.file_path)
            if frame is None:
                raise Exception("打开图片失败")
            if frame.size == 0:
                raise Exception("侧面图片无效")
            gray_ng, mean_gray = self.window.check_mean_gray_ng(frame)
            if gray_ng:
                self.result_ready.emit(
                    {
                        "file_path": self.file_path,
                        "side_img": frame.copy(),
                        "side_ok": False,
                        "ai_time": 0.0,
                        "gray_ng": True,
                        "mean_gray": mean_gray,
                    }
                )
                return
            ai_start = time.time()
            side_img, side_ok = self.window.side_detect_with_params(
                frame.copy(),
                conf=self.side_conf,
                min_area=self.side_min_area,
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return
            self.result_ready.emit(
                {
                    "file_path": self.file_path,
                    "side_img": side_img,
                    "side_ok": bool(side_ok),
                    "ai_time": ai_time,
                }
            )
        except Exception as e:
            self.error.emit(str(e))


class ModelWarmupThread(QThread):
    result_ready = pyqtSignal(object)  # dict
    error = pyqtSignal(str)

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            # Use a moderate tensor size so warmup is meaningful but not too expensive.
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            timings = {}

            t0 = time.time()
            self.window._run_yolo(self.window.model, dummy, conf=0.25, iou=0.7, verbose=False)
            timings["front_ai_warmup_s"] = time.time() - t0
            if self.isInterruptionRequested():
                return

            t0 = time.time()
            self.window._run_yolo(self.window.seg_model, dummy, conf=0.25, iou=0.7, verbose=False)
            timings["front_trad_warmup_s"] = time.time() - t0
            if self.isInterruptionRequested():
                return

            t0 = time.time()
            self.window._run_yolo(self.window.side_model, dummy, conf=0.25, iou=0.7, verbose=False)
            timings["side_ai_warmup_s"] = time.time() - t0

            self.result_ready.emit(timings)
        except Exception as e:
            self.error.emit(str(e))

# 海康SDK导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MvImport'))
try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    from MvErrorDefine_const import *
except ImportError:
    print("SDK missing")
    sys.exit(1)

# ================== 全局常量 =================
CONFIG_PATH = "config.json"
MODEL_PATH = "best-seg.onnx"
SEG_MODEL_PATH = "best_seg_DW.onnx"
SIDE_MODEL_PATH = "best_CM.onnx"
IMAGE_DIR = "image"
ROI_IMAGE_DIR = "imageROI"
YOLO_DEVICE = 0  # 强制使用第一张GPU
USE_ONNX_MODELS = any(str(p).lower().endswith(".onnx") for p in [MODEL_PATH, SEG_MODEL_PATH, SIDE_MODEL_PATH])

# 记录模型推理设备信息
def log_model_device_info():
    try:
        providers = ort.get_available_providers()
        device_info = f"ONNX Runtime Providers: {providers} | configured YOLO_DEVICE={YOLO_DEVICE}"
        print(device_info)
    except Exception as e:
        print(f"Error logging device info: {e}")

def get_configured_infer_device_label() -> str:
    if USE_ONNX_MODELS:
        return f"ONNX(device={YOLO_DEVICE})"
    return f"CUDA(device={YOLO_DEVICE})" if torch.cuda.is_available() else "CPU"

# ================== 工具函数 =================
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

def draw_chinese_text(img: np.ndarray, text: str, position: tuple, font_size: int = 20, color: tuple = (0, 255, 0), thickness: int = 2):
    font_path = r"C:\Windows\Fonts\simfang.ttf"
    cv_img = img.copy()
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"警告: 未找到字体 {font_path}，使用默认字体(中文可能乱码)")
    rgb_color = color[::-1]  # BGR -> RGB
    draw.text(position, text, font=font, fill=rgb_color, stroke_width=thickness)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 海康相机采集类
class HiKCamera:
    def __init__(self):
        self.cam = None
        self.is_opened = False
        self.device_list = MV_CC_DEVICE_INFO_LIST()

    @staticmethod
    def _ip_to_str(ip: int) -> str:
        return f"{(ip >> 24) & 0xFF}.{(ip >> 16) & 0xFF}.{(ip >> 8) & 0xFF}.{ip & 0xFF}"

    @classmethod
    def enum_gige_ip_list(cls):
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != MV_OK:
            raise Exception(f"枚举设备失败: 0x{ret:X}")
        ips = []
        for i in range(device_list.nDeviceNum):
            dev = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev.nTLayerType == MV_GIGE_DEVICE:
                try:
                    ip = dev.SpecialInfo.stGigEInfo.nCurrentIp
                except Exception:
                    ip = dev.stSpecialInfo.stGigEInfo.nCurrentIp
                ips.append(cls._ip_to_str(ip))
        return ips

    def initialize(self):
        print("正在初始化海康SDK...")
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            raise Exception(f"SDK初始化失败: 0x{ret:X} (请检查MvCameraControl.dll)")
        print("SDK初始化成功")

    def enum_devices(self):
        print("正在枚举相机...")
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != MV_OK:
            raise Exception(f"枚举设备失败: 0x{ret:X}")
        if self.device_list.nDeviceNum == 0:
            raise Exception("未检测到海康相机, 请检查连接与驱动")
        print(f"检测到 {self.device_list.nDeviceNum} 个相机")
        return self.device_list

    def _open_device_with_info(self, dev, label: str = ""):
        if self.cam:
            print("释放之前的相机资源..")
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.cam = None
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(dev)
        if ret != MV_OK:
            self.cam = None
            raise Exception(f"创建设备句柄失败: 0x{ret:X}")
        print("设备句柄创建成功")
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            self.cam.MV_CC_DestroyHandle()
            raise Exception(f"打开设备失败: 0x{ret:X} (可能被其他程序占用)")
        print("设备打开成功")
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != MV_OK:
            print("警告: 设置触发模式失败，尝试继续")
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
        if ret != MV_OK:
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BGR8_Packed)
        if ret != MV_OK:
            print("使用相机默认像素格式")
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            raise Exception(f"开始采集失败：0x{ret:X}")
        print("开始采集成功")
        self.is_opened = True
        if label:
            print(f"{label}相机打开成功")

    def open_device(self, index=0):
        if self.device_list.nDeviceNum == 0:
            self.enum_devices()
        if index >= self.device_list.nDeviceNum:
            raise Exception(f"设备索引 {index} 超出范围，检测到 {self.device_list.nDeviceNum} 个相机")
        dev = cast(self.device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        print(f"尝试打开设备 {index}...")
        self._open_device_with_info(dev, label=str(index))

    def open_device_by_ip(self, ip_str: str):
        if not ip_str:
            raise Exception("未设置相机IP")
        if self.device_list.nDeviceNum == 0:
            self.enum_devices()
        target = None
        for i in range(self.device_list.nDeviceNum):
            dev = cast(self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev.nTLayerType != MV_GIGE_DEVICE:
                continue
            try:
                ip = dev.SpecialInfo.stGigEInfo.nCurrentIp
            except Exception:
                ip = dev.stSpecialInfo.stGigEInfo.nCurrentIp
            if self._ip_to_str(ip) == ip_str:
                target = dev
                break
        if target is None:
            raise Exception(f"未找到指定IP相机: {ip_str}")
        print(f"尝试打开设备 IP: {ip_str}...")
        self._open_device_with_info(target, label=ip_str)

    def close(self):
        if self.cam:
            try:
                self.cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.cam = None
        self.is_opened = False

    def get_float(self, key: str) -> float:
        if not self.is_opened or not self.cam:
            raise Exception("相机未打开")
        st_val = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue(key, st_val)
        if ret != MV_OK:
            raise Exception(f"获取{key}失败: 0x{ret:X}")
        return float(st_val.fCurValue)

    def set_float(self, key: str, value: float):
        if not self.is_opened or not self.cam:
            raise Exception("相机未打开")
        ret = self.cam.MV_CC_SetFloatValue(key, float(value))
        if ret != MV_OK:
            raise Exception(f"设置{key}失败: 0x{ret:X}")

    def get_exposure(self) -> float:
        return self.get_float("ExposureTime")

    def set_exposure(self, value: float):
        self.set_float("ExposureTime", value)

    def get_gain(self) -> float:
        return self.get_float("Gain")

    def set_gain(self, value: float):
        self.set_float("Gain", value)
    def get_frame(self, timeout=1000):
        if not self.is_opened or not self.cam:
            raise Exception("相机未打开")
        st_frame_info = MV_FRAME_OUT_INFO_EX()
        buffer_size = 50 * 1024 * 1024
        data_buf = (c_ubyte * buffer_size)()
        ret = self.cam.MV_CC_GetOneFrameTimeout(data_buf, len(data_buf), st_frame_info, timeout)
        if ret != MV_OK:
            if ret in (0x80000002, 0x80000003):
                ret = self.cam.MV_CC_GetOneFrameTimeout(data_buf, len(data_buf), st_frame_info, timeout)
            if ret != MV_OK:
                raise Exception(f"获取图像失败: 0x{ret:X}")
        w = st_frame_info.nWidth
        h = st_frame_info.nHeight
        pt = st_frame_info.enPixelType
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

    #def close(self):
    #    if self.cam:
    #        self.cam.MV_CC_StopGrabbing()
    #        self.cam.MV_CC_CloseDevice()
    #        self.cam.MV_CC_DestroyHandle()
    #        self.cam = None
    #        self.is_opened = False

    def __del__(self):
        self.close()

# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高更盛04T11安装检测")
        icon_path = "1.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("警告: 未找到 1.ico 文件，使用默认图标")
        self.resize(1600, 900)
        self.total_count = 0
        self.good_count = 0
        self.side_total_count = 0
        self.side_good_count = 0
        self._rate_bar_width = 140
        self.camera = None
        self.camera2 = None  # 相机二（侧面检测）   
        self.plc = None
        self.plc_setting = ""
        self.plc_prev_trigger = False  # 相机一 R[0] 上升沿跟踪
        self.plc_prev_trigger2 = False  # 相机二 R[1] 上升沿跟踪
        self._plc_busy = False
        self._plc_busy2 = False
        self._detect_thread = None
        self._side_detect_thread = None
        self._image_detect_thread = None
        self._image_side_detect_thread = None
        self._warmup_thread = None
        self._warmup_done = False
        self._init_ui()
        self._update_rate_bar(self.good_count, self.total_count, self.front_ok_bar, self.front_ng_bar)
        self._update_rate_bar(self.side_good_count, self.side_total_count, self.side_ok_bar, self.side_ng_bar)
        self.load_config()
        self.refresh_plc_ports()
        self.refresh_camera_ip_list()
        self.timer = QTimer()
        self.timer.timeout.connect(self._plc_poll)
        self.timer.start(50)
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)
        self._update_status()
        
        # 加载 YOLO 模型
        self.log_text.append("正在加载YOLO模型...")
        try:
            self._ensure_gpu_runtime()
            self.model = YOLO(MODEL_PATH)
            self.seg_model = YOLO(SEG_MODEL_PATH, task='segment')
            self.side_model = YOLO(SIDE_MODEL_PATH, task='segment')
            self.log_text.append("YOLO模型加载成功")
            log_model_device_info()
            QTimer.singleShot(100, self._start_model_warmup)
        except Exception as e:
            self.log_text.append(f"YOLO模型加载失败: {e}")
            
        # 启动后自动连接
        QTimer.singleShot(1000, self._auto_connect_all)

    def _ensure_gpu_runtime(self):
        if USE_ONNX_MODELS:
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                raise RuntimeError(
                    "未检测到 ONNXRuntime CUDAExecutionProvider。"
                    "当前为 ONNX 模型，请安装 onnxruntime-gpu 并重启程序。"
                )
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("未检测到可用GPU(CUDA)，当前程序已配置为所有YOLO推理必须使用GPU。")

    def _run_yolo(self, yolo_model, frame: np.ndarray, **kwargs):
        # ONNX 后端由 onnxruntime provider 决定是否走 GPU；不要传 torch device，避免 cpu-only torch 报错
        if USE_ONNX_MODELS:
            return yolo_model(frame, device=YOLO_DEVICE, **kwargs)
        return yolo_model(frame, device=YOLO_DEVICE, **kwargs)

    def _start_model_warmup(self):
        if self._warmup_done:
            return
        if self._warmup_thread and self._warmup_thread.isRunning():
            return
        if not all(hasattr(self, name) for name in ("model", "seg_model", "side_model")):
            return
        self.log_text.append("开始后台预热模型（启动后首帧加速）...")
        self._warmup_thread = ModelWarmupThread(self)
        self._warmup_thread.result_ready.connect(self._on_model_warmup_done)
        self._warmup_thread.error.connect(self._on_model_warmup_error)
        self._warmup_thread.start()

    def _on_model_warmup_done(self, r: dict):
        self._warmup_done = True
        self._warmup_thread = None
        self.log_text.append(
            "模型预热完成 | "
            f"前面AI: {r.get('front_ai_warmup_s', 0.0):.3f}s | "
            f"白点分割: {r.get('front_trad_warmup_s', 0.0):.3f}s | "
            f"侧面AI: {r.get('side_ai_warmup_s', 0.0):.3f}s"
        )

    def _on_model_warmup_error(self, msg: str):
        self._warmup_thread = None
        self.log_text.append(f"模型预热失败: {msg}")

    def _auto_connect_all(self):
        """程序启动后自动连接所有设备"""
        self.log_text.append("--- 正在执行启动自动连接 ---")
        
        # 1. 连接 PLC
        try:
            self.connect_plc()
        except Exception as e:
            self.log_text.append(f"自动连接PLC失败: {e}")

        # 2. 连接正面相机
        try:
            self._ensure_camera_opened(side=False)
            self.log_text.append("自动连接正面相机成功")
        except Exception as e:
            self.log_text.append(f"自动连接正面相机失败: {e}")

        # 3. 连接侧面相机
        try:
            self._ensure_camera_opened(side=True)
            self.log_text.append("自动连接侧面相机成功")
        except Exception as e:
            self.log_text.append(f"自动连接侧面相机失败: {e}")
        
        self.log_text.append("--- 自动连接流程结束 ---")
        
    def _plc_poll(self):
        if not self.plc or not getattr(self.plc, "is_connected", False):
            return
        try:
            front_trigger_addr = self._get_plc_addr(self.plc_trigger_front_edit, "R1205")
            side_trigger_addr = self._get_plc_addr(self.plc_trigger_side_edit, "R1208")
            
            # 使用 MEWTOCOL 协议直接读取
            b0 = self.plc.read_bit(front_trigger_addr)
            b1 = self.plc.read_bit(side_trigger_addr)
            
            # 相机一触发 (上升沿)
            if b0 and not self.plc_prev_trigger and not self._plc_busy:
                self._start_detection(source="PLC")
            self.plc_prev_trigger = b0
            
            # 相机二触发 (上升沿)
            if b1 and not self.plc_prev_trigger2 and not self._plc_busy2:
                self._start_side_detection(source="PLC")
            self.plc_prev_trigger2 = b1
        except Exception as e:
            self.log_text.append(f"PLC轮询异常: {e}")

    def _set_status_label(self, label: QLabel, ok: bool):
        if ok:
            label.setText("在线")
            label.setStyleSheet("color: #2ecc71; font-weight: bold; font-size: 20px;")
        else:
            label.setText("离线")
            label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 20px;")

    def _update_status(self):
        cam_ok = bool(self.camera and getattr(self.camera, "is_opened", False))
        cam2_ok = bool(self.camera2 and getattr(self.camera2, "is_opened", False))
        plc_ok = bool(self.plc and getattr(self.plc, "is_connected", False))
        self._set_status_label(self.label_cam_status, cam_ok)
        self._set_status_label(self.label_cam2_status, cam2_ok)
        self._set_status_label(self.label_plc_status, plc_ok)

        # 更新 R1205 和 R1208 实时状态显示
        if hasattr(self, "label_r1205_status"):
            self.label_r1205_status.setText("ON" if self.plc_prev_trigger else "OFF")
            self.label_r1205_status.setStyleSheet(f"color: {'#2ecc71' if self.plc_prev_trigger else '#555'}; font-size: 11px; font-weight: bold;")
        if hasattr(self, "label_r1208_status"):
            self.label_r1208_status.setText("ON" if self.plc_prev_trigger2 else "OFF")
            self.label_r1208_status.setStyleSheet(f"color: {'#2ecc71' if self.plc_prev_trigger2 else '#555'}; font-size: 11px; font-weight: bold;")

        if self.plc and getattr(self.plc, "is_connected", False):
            st = self.plc.get_status()
            self.label_plc_timeout.setText(str(st.get("timeout", 0)))
            self.label_plc_reconnect.setText(str(st.get("reconnects", 0)))
            self.label_plc_queue.setText(str(st.get("queue_size", 0)))
            self.label_plc_crc.setText(str(st.get("crc_err", 0)))
        else:
            self.label_plc_timeout.setText("0")
            self.label_plc_reconnect.setText("0")
            self.label_plc_queue.setText("0")
            self.label_plc_crc.setText("0")

    def get_front_cam_ip(self) -> str:
        return self.front_cam_ip_combo.currentText().strip() if hasattr(self, "front_cam_ip_combo") else ""

    def get_side_cam_ip(self) -> str:
        return self.side_cam_ip_combo.currentText().strip() if hasattr(self, "side_cam_ip_combo") else ""

    def refresh_camera_ip_list(self):
        try:
            ips = HiKCamera.enum_gige_ip_list()
        except Exception as e:
            self.log_text.append(f"相机IP枚举失败: {e}")
            ips = []
        front_current = self.get_front_cam_ip()
        side_current = self.get_side_cam_ip()
        def _fill(combo, current):
            combo.blockSignals(True)
            combo.clear()
            if current and current not in ips:
                combo.addItem(current)
            for ip in ips:
                combo.addItem(ip)
            if current:
                idx = combo.findText(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)
        _fill(self.front_cam_ip_combo, front_current)
        _fill(self.side_cam_ip_combo, side_current)

    def _ensure_camera_opened(self, side: bool = False) -> HiKCamera:
        cam = self.camera2 if side else self.camera
        if not cam:
            cam = HiKCamera()
            cam.initialize()
            if side:
                self.camera2 = cam
            else:
                self.camera = cam
        if not cam.is_opened:
            ip = self.get_side_cam_ip() if side else self.get_front_cam_ip()
            cam.open_device_by_ip(ip)
        return cam

    def open_front_camera(self):
        try:
            self._ensure_camera_opened(side=False)
            self.log_text.append("正面相机已打开")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"打开正面相机失败: {e}")

    def close_front_camera(self):
        try:
            if self.camera:
                self.camera.close()
            self.log_text.append("正面相机已关闭")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"关闭正面相机失败: {e}")

    def open_side_camera(self):
        try:
            self._ensure_camera_opened(side=True)
            self.log_text.append("侧面相机已打开")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"打开侧面相机失败: {e}")

    def close_side_camera(self):
        try:
            if self.camera2:
                self.camera2.close()
            self.log_text.append("侧面相机已关闭")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"关闭侧面相机失败: {e}")

    def read_front_exposure(self):
        try:
            cam = self._ensure_camera_opened(side=False)
            self.front_exposure_edit.setText(f"{cam.get_exposure():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"读取正面曝光失败: {e}")

    def set_front_exposure(self):
        try:
            cam = self._ensure_camera_opened(side=False)
            cam.set_exposure(float(self.front_exposure_edit.text()))
        except Exception as e:
            QMessageBox.warning(self, "提示", f"设置正面曝光失败: {e}")

    def read_front_gain(self):
        try:
            cam = self._ensure_camera_opened(side=False)
            self.front_gain_edit.setText(f"{cam.get_gain():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"读取正面增益失败: {e}")

    def set_front_gain(self):
        try:
            cam = self._ensure_camera_opened(side=False)
            cam.set_gain(float(self.front_gain_edit.text()))
        except Exception as e:
            QMessageBox.warning(self, "提示", f"设置正面增益失败: {e}")

    def read_side_exposure(self):
        try:
            cam = self._ensure_camera_opened(side=True)
            self.side_exposure_edit.setText(f"{cam.get_exposure():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"读取侧面曝光失败: {e}")

    def set_side_exposure(self):
        try:
            cam = self._ensure_camera_opened(side=True)
            cam.set_exposure(float(self.side_exposure_edit.text()))
        except Exception as e:
            QMessageBox.warning(self, "提示", f"设置侧面曝光失败: {e}")

    def read_side_gain(self):
        try:
            cam = self._ensure_camera_opened(side=True)
            self.side_gain_edit.setText(f"{cam.get_gain():.2f}")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"读取侧面增益失败: {e}")

    def set_side_gain(self):
        try:
            cam = self._ensure_camera_opened(side=True)
            cam.set_gain(float(self.side_gain_edit.text()))
        except Exception as e:
            QMessageBox.warning(self, "提示", f"设置侧面增益失败: {e}")

    def refresh_plc_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        current = self.plc_port_combo.currentText().strip() if hasattr(self, "plc_port_combo") else ""
        self.plc_port_combo.blockSignals(True)
        self.plc_port_combo.clear()
        if current and current not in ports:
            self.plc_port_combo.addItem(current)
        for p in ports:
            self.plc_port_combo.addItem(p)
        if current:
            idx = self.plc_port_combo.findText(current)
            if idx >= 0:
                self.plc_port_combo.setCurrentIndex(idx)
        self.plc_port_combo.blockSignals(False)

    def build_plc_setting(self) -> str:
        port = self.plc_port_combo.currentText().strip()
        baud = self.plc_baud_combo.currentText().strip()
        databits = self.plc_databits_combo.currentText().strip()
        stopbits = self.plc_stopbits_combo.currentText().strip()
        parity_text = self.plc_parity_combo.currentText().strip().lower()
        parity_map = {"none": "0", "odd": "1", "even": "2"}
        parity = parity_map.get(parity_text, "0")
        poll = self.plc_poll_edit.text().strip() or "10"
        return f"{port},{baud},{databits},{stopbits},{parity},{poll}"

    def apply_plc_setting(self, setting_str: str):
        if not setting_str:
            return
        parts = [p.strip() for p in setting_str.split(",")]
        if len(parts) < 6:
            return
        port, baud, databits, stopbits, parity, poll = parts[:6]
        self.plc_port_combo.setCurrentText(port)
        self.plc_baud_combo.setCurrentText(baud)
        self.plc_databits_combo.setCurrentText(databits)
        self.plc_stopbits_combo.setCurrentText(stopbits)
        parity_map = {"0": "None", "1": "Odd", "2": "Even"}
        self.plc_parity_combo.setCurrentText(parity_map.get(parity, "None"))
        self.plc_poll_edit.setText(poll)

    def _get_plc_addr(self, edit: QLineEdit, default: str) -> str:
        addr = edit.text().strip().upper()
        return addr if addr else default

    def _r_index_from_addr(self, addr: str):
        m = re.match(r"^[Rr]?(\\d+)$", addr.strip())
        return int(m.group(1)) if m else None

    def connect_plc(self):
        self.plc_setting = self.build_plc_setting()
        if not self.plc_setting:
            self.log_text.append("PLC参数无效")
            return
        if self.plc:
            try:
                self.plc.close()
            except Exception:
                pass
        self.plc = MewtocolPLC()
        ok = self.plc.open(self.plc_setting)
        if ok:
            self.log_text.append("PLC连接成功 (MEWTOCOL)")
        else:
            self.log_text.append("PLC连接失败")

    def disconnect_plc(self):
        if self.plc:
            try:
                self.plc.close()
            except Exception:
                pass
        self.plc = None
        self.log_text.append("PLC已断开")

    def restore_defaults(self):
        self.conf_edit.setText("0.7")
        self.ai_area_edit.setText("3000")
        self.mean_gray_min_edit.setText("10")
        self.mean_gray_max_edit.setText("245")
        self.tgray_edit.setText("254")
        self.tarea_edit.setText("45")
        self.tgray1_edit.setText("240")
        self.tarea1_edit.setText("200")
        self.side_conf_edit.setText("0.25")
        self.side_area_edit.setText("500")
        self.front_roi_row1_edit.setText("684")
        self.front_roi_col1_edit.setText("808")
        self.front_roi_row2_edit.setText("1076")
        self.front_roi_col2_edit.setText("1582")
        self.side_roi_row1_edit.setText("680")
        self.side_roi_col1_edit.setText("800")
        self.side_roi_row2_edit.setText("1080")
        self.side_roi_col2_edit.setText("1580")
        self.front_roi_rotate_edit.setText("0")
        self.front_flip_horizontal_checkbox.setChecked(False)
        self.front_flip_vertical_checkbox.setChecked(False)
        self.side_roi_rotate_edit.setText("0")
        self.side_flip_horizontal_checkbox.setChecked(False)
        self.side_flip_vertical_checkbox.setChecked(False)
        self.save_raw_image_checkbox.setChecked(False)
        self.save_roi_image_checkbox.setChecked(False)
        self.plc_trigger_front_edit.setText("R0000")
        self.plc_trigger_side_edit.setText("R0001")
        self.plc_result_front_edit.setText("R0502")
        self.plc_done_front_edit.setText("R0501")
        self.plc_result_side_edit.setText("R0504")
        self.plc_done_side_edit.setText("R0503")
        self.conf_slider.setValue(70)
        self.ai_area_slider.setValue(3000)
        self.mean_gray_min_slider.setValue(10)
        self.mean_gray_max_slider.setValue(245)
        self.tgray_slider.setValue(254)
        self.tarea_slider.setValue(45)
        self.tgray1_slider.setValue(240)
        self.tarea1_slider.setValue(200)
        self.side_conf_slider.setValue(25)
        self.side_area_slider.setValue(500)
        QMessageBox.information(self, "提示", "参数已恢复默认")

    def toggle_log_view(self):
        if self.log_text.isVisible():
            self.log_text.setVisible(False)
            self.btn_toggle_log.setText("显示日志")
        else:
            self.log_text.setVisible(True)
            self.btn_toggle_log.setText("隐藏日志")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        page_display = QWidget()
        display_layout = QVBoxLayout(page_display)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(12)

        def _wrap_collapsible(title: str, content: QWidget, checked: bool = True) -> QWidget:
            wrapper = QWidget()
            wrap_layout = QVBoxLayout(wrapper)
            wrap_layout.setContentsMargins(0, 0, 0, 0)
            wrap_layout.setSpacing(6)
            toggle = QToolButton()
            toggle.setText(title)
            toggle.setCheckable(True)
            toggle.setChecked(checked)
            toggle.setToolButtonStyle(Qt.ToolButtonTextOnly)
            toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            toggle.setStyleSheet("QToolButton { font-size: 14px; font-weight: bold; }")
            content.setVisible(checked)
            def _on_toggle(state: bool):
                toggle.setArrowType(Qt.DownArrow if state else Qt.RightArrow)
                content.setVisible(state)
            toggle.toggled.connect(_on_toggle)
            wrap_layout.addWidget(toggle)
            wrap_layout.addWidget(content)
            return wrapper

        self.setStyleSheet("""
            QMainWindow { background-color: #f0f4f8; }
            QLabel { font-size: 14px; color: #333; }
            QGroupBox { font-weight: bold; font-size: 16px; border: 2px solid #a0c4ff; border-radius: 10px; margin-top: 10px; padding-top: 10px; background-color: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QPushButton { background-color: #4a90e2; color: white; font-size: 16px; border-radius: 8px; padding: 10px; min-width: 120px; }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2a6aaf; }
            QLineEdit { padding: 8px; border: 2px solid #bdc3c7; border-radius: 6px; font-size: 14px; }
            QSlider::groove:horizontal { height: 8px; background: #e0e0e0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #4a90e2; width: 20px; border-radius: 10px; margin: -6px 0; }
            QTextEdit { border: 2px solid #bdc3c7; border-radius: 8px; }
        """)

        status_group = QGroupBox("状态总览")
        status_group.setStyleSheet("QGroupBox { font-size: 14px; }")
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(8, 4, 8, 4)
        status_layout.setSpacing(10)

        self.label_cam_status = QLabel("--")
        self.label_cam_status.setStyleSheet("color: #c0392b; font-size: 11px;")
        self.label_cam2_status = QLabel("--")
        self.label_cam2_status.setStyleSheet("color: #c0392b; font-size: 11px;")
        self.label_plc_status = QLabel("--")
        self.label_plc_status.setStyleSheet("color: #c0392b; font-size: 11px;")

        cam_block = QHBoxLayout()
        cam_title = QLabel("正面相机")
        cam_title.setStyleSheet("color: #555; font-size: 11px;")
        cam_block.addWidget(cam_title)
        cam_block.addWidget(self.label_cam_status)
        cam_widget = QWidget()
        cam_widget.setLayout(cam_block)

        cam2_block = QHBoxLayout()
        cam2_title = QLabel("侧面相机")
        cam2_title.setStyleSheet("color: #555; font-size: 11px;")
        cam2_block.addWidget(cam2_title)
        cam2_block.addWidget(self.label_cam2_status)
        cam2_widget = QWidget()
        cam2_widget.setLayout(cam2_block)

        plc_block = QHBoxLayout()
        plc_title = QLabel("PLC通讯")
        plc_title.setStyleSheet("color: #555; font-size: 11px;")
        plc_block.addWidget(plc_title)
        plc_block.addWidget(self.label_plc_status)
        plc_widget = QWidget()
        plc_widget.setLayout(plc_block)

        r1205_block = QHBoxLayout()
        r1205_title = QLabel("正面触发(R1205)")
        r1205_title.setStyleSheet("color: #555; font-size: 11px;")
        self.label_r1205_status = QLabel("OFF")
        self.label_r1205_status.setStyleSheet("color: #555; font-size: 11px; font-weight: bold;")
        r1205_block.addWidget(r1205_title)
        r1205_block.addWidget(self.label_r1205_status)
        r1205_widget = QWidget()
        r1205_widget.setLayout(r1205_block)

        r1208_block = QHBoxLayout()
        r1208_title = QLabel("侧面触发(R1208)")
        r1208_title.setStyleSheet("color: #555; font-size: 11px;")
        self.label_r1208_status = QLabel("OFF")
        self.label_r1208_status.setStyleSheet("color: #555; font-size: 11px; font-weight: bold;")
        r1208_block.addWidget(r1208_title)
        r1208_block.addWidget(self.label_r1208_status)
        r1208_widget = QWidget()
        r1208_widget.setLayout(r1208_block)

        metrics_layout = QHBoxLayout()
        self.label_plc_timeout = QLabel("0")
        self.label_plc_reconnect = QLabel("0")
        self.label_plc_queue = QLabel("0")
        self.label_plc_crc = QLabel("0")
        for v in [self.label_plc_timeout, self.label_plc_reconnect, self.label_plc_queue, self.label_plc_crc]:
            v.setStyleSheet("color: #555; font-size: 11px;")
        metrics_layout.addWidget(QLabel("超时:"))
        metrics_layout.addWidget(self.label_plc_timeout)
        metrics_layout.addWidget(QLabel("重连:"))
        metrics_layout.addWidget(self.label_plc_reconnect)
        metrics_layout.addWidget(QLabel("队列:"))
        metrics_layout.addWidget(self.label_plc_queue)
        metrics_layout.addWidget(QLabel("CRC:"))
        metrics_layout.addWidget(self.label_plc_crc)
        metrics_widget = QWidget()
        metrics_widget.setLayout(metrics_layout)

        status_layout.addWidget(cam_widget)
        status_layout.addWidget(cam2_widget)
        status_layout.addWidget(plc_widget)
        status_layout.addWidget(r1205_widget)
        status_layout.addWidget(r1208_widget)
        status_layout.addStretch()
        status_layout.addWidget(metrics_widget)
        status_group.setLayout(status_layout)
        display_layout.addWidget(status_group)

        # 图像显示区（三个窗口：正面AI、白点、侧面）
        image_layout = QHBoxLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(16)

        front_window_group = QGroupBox("AI正面检测")
        front_window_layout = QVBoxLayout()
        front_tag = QLabel("正面")
        front_tag.setAlignment(Qt.AlignLeft)
        front_tag.setStyleSheet("color: #666; font-size: 12px;")
        front_window_layout.addWidget(front_tag)
        self.show_window = QLabel()
        self.show_window.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
        self.show_window.setAlignment(Qt.AlignCenter)
        self.show_window.setScaledContents(True)
        self.show_window.setMinimumSize(360, 360)
        self.show_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        front_window_layout.addWidget(self.show_window)
        front_window_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        front_window_group.setLayout(front_window_layout)
        front_window_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        trad_window_group = QGroupBox("白点检测")
        trad_window_layout = QVBoxLayout()
        trad_tag = QLabel("白点")
        trad_tag.setAlignment(Qt.AlignLeft)
        trad_tag.setStyleSheet("color: #666; font-size: 12px;")
        trad_window_layout.addWidget(trad_tag)
        self.picture_box1 = QLabel()
        self.picture_box1.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
        self.picture_box1.setAlignment(Qt.AlignCenter)
        self.picture_box1.setScaledContents(True)
        self.picture_box1.setMinimumSize(360, 360)
        self.picture_box1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        trad_window_layout.addWidget(self.picture_box1)
        trad_window_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        trad_window_group.setLayout(trad_window_layout)
        trad_window_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 侧面检测窗口
        side_window_group = QGroupBox("AI侧面检测")
        side_window_layout = QVBoxLayout()
        side_tag = QLabel("侧面")
        side_tag.setAlignment(Qt.AlignLeft)
        side_tag.setStyleSheet("color: #666; font-size: 12px;")
        side_window_layout.addWidget(side_tag)
        self.side_window = QLabel()
        self.side_window.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 10px;")
        self.side_window.setAlignment(Qt.AlignCenter)
        self.side_window.setScaledContents(True)
        self.side_window.setMinimumSize(360, 360)
        self.side_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        side_window_layout.addWidget(self.side_window)
        side_window_layout.setSizeConstraint(QLayout.SetDefaultConstraint)
        side_window_group.setLayout(side_window_layout)
        side_window_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        image_layout.addWidget(front_window_group)
        image_layout.addWidget(trad_window_group)
        image_layout.addWidget(side_window_group)
        display_layout.addLayout(image_layout, stretch=8)

        # 中间区：结果 + 参数
        mid_layout = QHBoxLayout()
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(16)

        # 检测结果显示  
        result_group = QGroupBox("检测结果")
        result_layout = QHBoxLayout()
        result_layout.setSpacing(24)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(12)
        right_col.setSpacing(12)

        ai_layout = QHBoxLayout()
        ai_label = QLabel("AI安装检测：")
        ai_label.setFont(QFont("Arial", 14, QFont.Bold))
        ai_label.setAlignment(Qt.AlignCenter)
        ai_label.setFixedWidth(100)
        self.text_ai = QLabel("--")
        self.text_ai.setAlignment(Qt.AlignCenter)
        self.text_ai.setFixedHeight(45)
        self.text_ai.setFixedWidth(100)
        self.text_ai.setFont(QFont("Arial", 18, QFont.Bold))
        self.text_ai.setStyleSheet("background-color: #f0f0f0; color: #333; border: 2px solid #a0c4ff; border-radius: 8px;")
        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.text_ai)
        right_col.addLayout(ai_layout)

        trad_layout = QHBoxLayout()
        trad_label = QLabel("白点检测：")
        trad_label.setFont(QFont("Arial", 14, QFont.Bold))
        trad_label.setAlignment(Qt.AlignCenter)
        trad_label.setFixedWidth(100)
        self.text_trad = QLabel("--")
        self.text_trad.setAlignment(Qt.AlignCenter)
        self.text_trad.setFixedHeight(45)
        self.text_trad.setFixedWidth(100)
        self.text_trad.setFont(QFont("Arial", 18, QFont.Bold))
        self.text_trad.setStyleSheet("background-color: #f0f0f0; color: #333; border: 2px solid #a0c4ff; border-radius: 8px;")
        trad_layout.addWidget(trad_label)
        trad_layout.addWidget(self.text_trad)
        right_col.addLayout(trad_layout)

        final_layout = QHBoxLayout()
        final_label = QLabel("综合判定:")
        final_label.setFont(QFont("Arial", 14, QFont.Bold))
        final_label.setAlignment(Qt.AlignCenter)
        final_label.setFixedWidth(100)
        self.text_final = QLabel("--")
        self.text_final.setAlignment(Qt.AlignCenter)
        self.text_final.setFixedHeight(45)
        self.text_final.setFixedWidth(100)
        self.text_final.setFont(QFont("Arial", 18, QFont.Bold))
        self.text_final.setStyleSheet("background-color: #f0f0f0; color: #333; border: 2px solid #a0c4ff; border-radius: 8px;")
        final_layout.addWidget(final_label)
        final_layout.addWidget(self.text_final)
        left_col.addLayout(final_layout)

        side_res_layout = QHBoxLayout()
        side_res_label = QLabel("侧面检测：")
        side_res_label.setFont(QFont("Arial", 14, QFont.Bold))
        side_res_label.setAlignment(Qt.AlignCenter)
        side_res_label.setFixedWidth(100)
        self.text_side = QLabel("--")
        self.text_side.setAlignment(Qt.AlignCenter)
        self.text_side.setFixedHeight(45)
        self.text_side.setFixedWidth(100)
        self.text_side.setFont(QFont("Arial", 18, QFont.Bold))
        self.text_side.setStyleSheet("background-color: #f0f0f0; color: #333; border: 2px solid #a0c4ff; border-radius: 8px;")
        side_res_layout.addWidget(side_res_label)
        side_res_layout.addWidget(self.text_side)
        left_col.addLayout(side_res_layout)

        result_layout.addLayout(left_col)
        result_layout.addLayout(right_col)
        result_group.setLayout(result_layout)
        result_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mid_layout.addWidget(result_group)

        # 参数调节
        params_group = QGroupBox("参数调节")
        params_layout = QGridLayout()
        params_layout.setHorizontalSpacing(12)
        params_layout.setVerticalSpacing(8)

        labels = [
            "正面AI置信度(0-1)", "正面AI最小面积(px)",
            "平均灰度下限(0-255)", "平均灰度上限(0-255)",
            "白点灰度阈值1(0-255)", "白点面积阈值1(px)",
            "白点灰度阈值2(0-255)", "白点面积阈值2(px)",
            "侧面AI置信度(0-1)", "侧面AI最小面积(px)"
        ]
        self.conf_edit = QLineEdit("0.7")
        self.ai_area_edit = QLineEdit("3000")
        self.mean_gray_min_edit = QLineEdit("10")
        self.mean_gray_max_edit = QLineEdit("245")
        self.tgray_edit = QLineEdit("254")
        self.tarea_edit = QLineEdit("45")
        self.tgray1_edit = QLineEdit("240")
        self.tarea1_edit = QLineEdit("200")
        self.side_conf_edit = QLineEdit("0.25")
        self.side_area_edit = QLineEdit("500")
        edits = [self.conf_edit, self.ai_area_edit, self.mean_gray_min_edit, self.mean_gray_max_edit, self.tgray_edit, self.tarea_edit, self.tgray1_edit, self.tarea1_edit, self.side_conf_edit, self.side_area_edit]

        roi_group = QGroupBox("ROI设置(行1, 列1, 行2, 列2)")
        roi_layout = QFormLayout()
        self.front_roi_row1_edit = QLineEdit("684")
        self.front_roi_col1_edit = QLineEdit("808")
        self.front_roi_row2_edit = QLineEdit("1076")
        self.front_roi_col2_edit = QLineEdit("1582")
        self.side_roi_row1_edit = QLineEdit("680")
        self.side_roi_col1_edit = QLineEdit("800")
        self.side_roi_row2_edit = QLineEdit("1080")
        self.side_roi_col2_edit = QLineEdit("1580")
        roi_layout.addRow("正面ROI row1:", self.front_roi_row1_edit)
        roi_layout.addRow("正面ROI col1:", self.front_roi_col1_edit)
        roi_layout.addRow("正面ROI row2:", self.front_roi_row2_edit)
        roi_layout.addRow("正面ROI col2:", self.front_roi_col2_edit)
        roi_layout.addRow("侧面ROI row1:", self.side_roi_row1_edit)
        roi_layout.addRow("侧面ROI col1:", self.side_roi_col1_edit)
        roi_layout.addRow("侧面ROI row2:", self.side_roi_row2_edit)
        roi_layout.addRow("侧面ROI col2:", self.side_roi_col2_edit)
        roi_group.setLayout(roi_layout)

        roi_transform_group = QGroupBox("ROI截图后处理")
        roi_transform_layout = QFormLayout()
        self.front_roi_rotate_edit = QLineEdit("0")
        self.side_roi_rotate_edit = QLineEdit("0")
        self.front_flip_horizontal_checkbox = QCheckBox("水平镜像")
        self.front_flip_vertical_checkbox = QCheckBox("垂直镜像")
        self.side_flip_horizontal_checkbox = QCheckBox("水平镜像")
        self.side_flip_vertical_checkbox = QCheckBox("垂直镜像")
        front_flip_box = QWidget()
        front_flip_layout = QHBoxLayout(front_flip_box)
        front_flip_layout.setContentsMargins(0, 0, 0, 0)
        front_flip_layout.addWidget(self.front_flip_horizontal_checkbox)
        front_flip_layout.addWidget(self.front_flip_vertical_checkbox)
        front_flip_layout.addStretch()
        side_flip_box = QWidget()
        side_flip_layout = QHBoxLayout(side_flip_box)
        side_flip_layout.setContentsMargins(0, 0, 0, 0)
        side_flip_layout.addWidget(self.side_flip_horizontal_checkbox)
        side_flip_layout.addWidget(self.side_flip_vertical_checkbox)
        side_flip_layout.addStretch()
        roi_transform_layout.addRow("正面旋转角度(°):", self.front_roi_rotate_edit)
        roi_transform_layout.addRow("正面镜像:", front_flip_box)
        roi_transform_layout.addRow("侧面旋转角度(°):", self.side_roi_rotate_edit)
        roi_transform_layout.addRow("侧面镜像:", side_flip_box)
        roi_transform_group.setLayout(roi_transform_layout)

        image_save_group = QGroupBox("检测图像保存")
        image_save_layout = QVBoxLayout()
        self.save_raw_image_checkbox = QCheckBox("保存相机原图到根目录 image 文件夹")
        self.save_roi_image_checkbox = QCheckBox("保存ROI截图到根目录 imageROI 文件夹")
        self.save_raw_image_checkbox.setChecked(False)
        self.save_roi_image_checkbox.setChecked(False)
        image_save_layout.addWidget(self.save_raw_image_checkbox)
        image_save_layout.addWidget(self.save_roi_image_checkbox)
        image_save_group.setLayout(image_save_layout)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.ai_area_slider = QSlider(Qt.Horizontal)
        self.mean_gray_min_slider = QSlider(Qt.Horizontal)
        self.mean_gray_max_slider = QSlider(Qt.Horizontal)
        self.tgray_slider = QSlider(Qt.Horizontal)
        self.tarea_slider = QSlider(Qt.Horizontal)
        self.tgray1_slider = QSlider(Qt.Horizontal)
        self.tarea1_slider = QSlider(Qt.Horizontal)
        self.side_conf_slider = QSlider(Qt.Horizontal)
        self.side_area_slider = QSlider(Qt.Horizontal)
        sliders = [self.conf_slider, self.ai_area_slider, self.mean_gray_min_slider, self.mean_gray_max_slider, self.tgray_slider, self.tarea_slider, self.tgray1_slider, self.tarea1_slider, self.side_conf_slider, self.side_area_slider]

        ranges = [(0, 100), (0, 10000), (0, 255), (0, 255), (0, 255), (0, 500), (0, 255), (0, 500), (0, 100), (0, 5000)]
        range_texts = ["0-1", "0-10000", "0-255", "0-255", "0-255", "0-500", "0-255", "0-500", "0-1", "0-5000"]
        factors = [100, 1, 1, 1, 1, 1, 1, 1, 100, 1]
        for i, (slider, edit, r, factor) in enumerate(zip(sliders, edits, ranges, factors)):
            slider.setRange(r[0], r[1])
            slider.setValue(int(float(edit.text()) * factor))
            slider.valueChanged.connect(lambda v, e=edit, f=factor: e.setText(str(v / f)))
            edit.textChanged.connect(lambda t, s=slider, f=factor: s.setValue(int(float(t) * f)) if t.replace('.', '', 1).isdigit() else None)
            params_layout.addWidget(QLabel(labels[i]), i // 2, (i % 2) * 4)
            params_layout.addWidget(edit, i // 2, (i % 2) * 4 + 1)
            params_layout.addWidget(slider, i // 2, (i % 2) * 4 + 2)
            range_label = QLabel(f"建议:{range_texts[i]}")
            range_label.setStyleSheet("color: #777; font-size: 12px;")
            params_layout.addWidget(range_label, i // 2, (i % 2) * 4 + 3)

        params_group.setLayout(params_layout)

        params_page = QWidget()
        params_page_layout = QVBoxLayout(params_page)
        params_page_layout.setSpacing(12)
        camera_ip_group = QGroupBox("相机IP设置")
        camera_ip_layout = QFormLayout()
        self.front_cam_ip_combo = QComboBox()
        self.front_cam_ip_combo.setEditable(True)
        self.side_cam_ip_combo = QComboBox()
        self.side_cam_ip_combo.setEditable(True)
        camera_ip_layout.addRow("正面相机IP:", self.front_cam_ip_combo)
        camera_ip_layout.addRow("侧面相机IP:", self.side_cam_ip_combo)
        camera_ip_group.setLayout(camera_ip_layout)
        self.btn_refresh_ip = QPushButton("刷新相机IP")
        self.btn_refresh_ip.clicked.connect(self.refresh_camera_ip_list)
        camera_btn_layout = QHBoxLayout()
        camera_btn_layout.addWidget(self.btn_refresh_ip)
        camera_btn_layout.addStretch()

        camera_ctrl_group = QGroupBox("相机控制")
        camera_ctrl_layout = QGridLayout()
        camera_ctrl_layout.setHorizontalSpacing(10)
        camera_ctrl_layout.setVerticalSpacing(8)

        self.btn_open_front_cam = QPushButton("打开正面相机")
        self.btn_close_front_cam = QPushButton("关闭正面相机")
        self.front_exposure_edit = QLineEdit()
        self.front_exposure_edit.setPlaceholderText("单位: us")
        self.front_gain_edit = QLineEdit()
        self.front_gain_edit.setPlaceholderText("单位: dB")
        self.btn_read_front_exposure = QPushButton("读取曝光")
        self.btn_set_front_exposure = QPushButton("设置曝光")
        self.btn_read_front_gain = QPushButton("读取增益")
        self.btn_set_front_gain = QPushButton("设置增益")

        self.btn_open_side_cam = QPushButton("打开侧面相机")
        self.btn_close_side_cam = QPushButton("关闭侧面相机")
        self.side_exposure_edit = QLineEdit()
        self.side_exposure_edit.setPlaceholderText("单位: us")
        self.side_gain_edit = QLineEdit()
        self.side_gain_edit.setPlaceholderText("单位: dB")
        self.btn_read_side_exposure = QPushButton("读取曝光")
        self.btn_set_side_exposure = QPushButton("设置曝光")
        self.btn_read_side_gain = QPushButton("读取增益")
        self.btn_set_side_gain = QPushButton("设置增益")

        self.btn_open_front_cam.clicked.connect(self.open_front_camera)
        self.btn_close_front_cam.clicked.connect(self.close_front_camera)
        self.btn_read_front_exposure.clicked.connect(self.read_front_exposure)
        self.btn_set_front_exposure.clicked.connect(self.set_front_exposure)
        self.btn_read_front_gain.clicked.connect(self.read_front_gain)
        self.btn_set_front_gain.clicked.connect(self.set_front_gain)

        self.btn_open_side_cam.clicked.connect(self.open_side_camera)
        self.btn_close_side_cam.clicked.connect(self.close_side_camera)
        self.btn_read_side_exposure.clicked.connect(self.read_side_exposure)
        self.btn_set_side_exposure.clicked.connect(self.set_side_exposure)
        self.btn_read_side_gain.clicked.connect(self.read_side_gain)
        self.btn_set_side_gain.clicked.connect(self.set_side_gain)

        camera_ctrl_layout.addWidget(QLabel("正面相机"), 0, 0)
        camera_ctrl_layout.addWidget(self.btn_open_front_cam, 0, 1)
        camera_ctrl_layout.addWidget(self.btn_close_front_cam, 0, 2)
        camera_ctrl_layout.addWidget(QLabel("曝光"), 1, 0)
        camera_ctrl_layout.addWidget(self.front_exposure_edit, 1, 1)
        camera_ctrl_layout.addWidget(self.btn_read_front_exposure, 1, 2)
        camera_ctrl_layout.addWidget(self.btn_set_front_exposure, 1, 3)
        camera_ctrl_layout.addWidget(QLabel("增益"), 2, 0)
        camera_ctrl_layout.addWidget(self.front_gain_edit, 2, 1)
        camera_ctrl_layout.addWidget(self.btn_read_front_gain, 2, 2)
        camera_ctrl_layout.addWidget(self.btn_set_front_gain, 2, 3)

        camera_ctrl_layout.addWidget(QLabel("侧面相机"), 3, 0)
        camera_ctrl_layout.addWidget(self.btn_open_side_cam, 3, 1)
        camera_ctrl_layout.addWidget(self.btn_close_side_cam, 3, 2)
        camera_ctrl_layout.addWidget(QLabel("曝光"), 4, 0)
        camera_ctrl_layout.addWidget(self.side_exposure_edit, 4, 1)
        camera_ctrl_layout.addWidget(self.btn_read_side_exposure, 4, 2)
        camera_ctrl_layout.addWidget(self.btn_set_side_exposure, 4, 3)
        camera_ctrl_layout.addWidget(QLabel("增益"), 5, 0)
        camera_ctrl_layout.addWidget(self.side_gain_edit, 5, 1)
        camera_ctrl_layout.addWidget(self.btn_read_side_gain, 5, 2)
        camera_ctrl_layout.addWidget(self.btn_set_side_gain, 5, 3)

        camera_ctrl_group.setLayout(camera_ctrl_layout)
        camera_ctrl_collapsible = _wrap_collapsible("相机控制", camera_ctrl_group, checked=False)

        plc_group = QGroupBox("PLC通讯设置")
        plc_group.setStyleSheet("QGroupBox { font-size: 14px; } QLabel { font-size: 12px; } QLineEdit { font-size: 12px; }")
        plc_layout = QFormLayout()
        self.plc_port_combo = QComboBox()
        self.plc_port_combo.setEditable(True)
        self.plc_baud_combo = QComboBox()
        self.plc_baud_combo.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.plc_databits_combo = QComboBox()
        self.plc_databits_combo.addItems(["7", "8"])
        self.plc_stopbits_combo = QComboBox()
        self.plc_stopbits_combo.addItems(["1", "2"])
        self.plc_parity_combo = QComboBox()
        self.plc_parity_combo.addItems(["None", "Odd", "Even"])
        self.plc_poll_edit = QLineEdit("10")
        self.plc_poll_edit.setPlaceholderText("单位: ms")
        self.plc_trigger_front_edit = QLineEdit("R0000")
        self.plc_trigger_side_edit = QLineEdit("R0001")
        self.plc_result_front_edit = QLineEdit("R0502")
        self.plc_done_front_edit = QLineEdit("R0501")
        self.plc_result_side_edit = QLineEdit("R0504")
        self.plc_done_side_edit = QLineEdit("R0503")
        plc_layout.addRow("串口:", self.plc_port_combo)
        plc_layout.addRow("波特率:", self.plc_baud_combo)
        plc_layout.addRow("数据位:", self.plc_databits_combo)
        plc_layout.addRow("停止位:", self.plc_stopbits_combo)
        plc_layout.addRow("校验位:", self.plc_parity_combo)
        plc_layout.addRow("轮询(ms):", self.plc_poll_edit)
        
        plc_layout.addRow("正面触发拍照:", self._create_plc_test_row(self.plc_trigger_front_edit))
        plc_layout.addRow("侧面触发拍照:", self._create_plc_test_row(self.plc_trigger_side_edit))
        plc_layout.addRow("正面结果信号:", self._create_plc_test_row(self.plc_result_front_edit))
        plc_layout.addRow("正面完成信号:", self._create_plc_test_row(self.plc_done_front_edit))
        plc_layout.addRow("侧面结果信号:", self._create_plc_test_row(self.plc_result_side_edit))
        plc_layout.addRow("侧面完成信号:", self._create_plc_test_row(self.plc_done_side_edit))
        
        plc_group.setLayout(plc_layout)
        plc_collapsible = _wrap_collapsible("PLC通讯设置", plc_group, checked=False)

        plc_btn_layout = QHBoxLayout()
        self.btn_refresh_plc = QPushButton("刷新串口")
        self.btn_refresh_plc.clicked.connect(self.refresh_plc_ports)
        self.btn_connect_plc = QPushButton("连接PLC")
        self.btn_connect_plc.clicked.connect(self.connect_plc)
        self.btn_disconnect_plc = QPushButton("断开PLC")
        self.btn_disconnect_plc.clicked.connect(self.disconnect_plc)
        plc_btn_layout.addWidget(self.btn_refresh_plc)
        plc_btn_layout.addWidget(self.btn_connect_plc)
        plc_btn_layout.addWidget(self.btn_disconnect_plc)
        plc_btn_layout.addStretch()
        params_top_layout = QHBoxLayout()
        params_top_layout.setSpacing(16)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(12)
        right_col.setSpacing(12)

        left_col.addWidget(params_group)
        left_col.addWidget(roi_group)
        left_col.addWidget(roi_transform_group)
        left_col.addWidget(image_save_group)
        left_col.addStretch()

        right_col.addWidget(camera_ip_group)
        right_col.addLayout(camera_btn_layout)
        right_col.addWidget(camera_ctrl_collapsible)
        right_col.addWidget(plc_collapsible)
        right_col.addLayout(plc_btn_layout)
        right_col.addStretch()

        params_top_layout.addLayout(left_col, 3)
        params_top_layout.addLayout(right_col, 2)
        params_page_layout.addLayout(params_top_layout)
        params_btn_layout = QHBoxLayout()
        self.btn_defaults = QPushButton("恢复默认")
        self.btn_defaults.setFixedSize(160, 45)
        self.btn_defaults.clicked.connect(self.restore_defaults)
        self.btn_save = QPushButton("保存参数")
        self.btn_save.setFixedSize(160, 45)
        self.btn_save.clicked.connect(self.save_config)
        self.btn_load = QPushButton("加载参数")
        self.btn_load.setFixedSize(160, 45)
        self.btn_load.clicked.connect(self.load_config)
        params_btn_layout.addStretch()
        params_btn_layout.addWidget(self.btn_defaults)
        params_btn_layout.addWidget(self.btn_save)
        params_btn_layout.addWidget(self.btn_load)
        params_page_layout.addLayout(params_btn_layout)
        params_page_layout.addStretch()
        result_stats_group = QGroupBox("结果与统计")
        result_stats_layout = QHBoxLayout()
        result_stats_layout.addLayout(mid_layout)
        result_stats_group.setLayout(result_stats_layout)
        result_stats_group.setMaximumHeight(170)
        display_layout.addWidget(result_stats_group)

        # 底部区：按钮 + 日志 + 统计
        bottom_layout = QHBoxLayout()

        btn_layout = QVBoxLayout()
        self.btn_detect = QPushButton("一键检测")
        self.btn_detect.setFixedSize(180, 70)
        self.btn_detect.clicked.connect(lambda: self._start_detection(source="MANUAL"))
        self.btn_side_detect = QPushButton("一键侧面检测")
        self.btn_side_detect.setFixedSize(180, 70)
        self.btn_side_detect.clicked.connect(lambda: self._start_side_detection(source="MANUAL"))
        self.btn_image_detect = QPushButton("打开图片检测")
        self.btn_image_detect.setFixedSize(160, 50)
        self.btn_image_detect.clicked.connect(self.on_image_detect)
        self.btn_image_side_detect = QPushButton("打开图片侧面检测")
        self.btn_image_side_detect.setFixedSize(160, 50)
        self.btn_image_side_detect.clicked.connect(self.on_image_side_detect)
        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_side_detect)
        btn_layout.addWidget(self.btn_image_detect)
        btn_layout.addWidget(self.btn_image_side_detect)
        btn_layout.addStretch()
        bottom_layout.addLayout(btn_layout)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.btn_toggle_log = QPushButton("隐藏日志")
        self.btn_toggle_log.setFixedHeight(35)
        self.btn_toggle_log.setText("显示日志")
        self.btn_toggle_log.clicked.connect(self.toggle_log_view)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setVisible(False)
        log_layout.addWidget(self.btn_toggle_log, alignment=Qt.AlignRight)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        bottom_layout.addWidget(log_group, stretch=3)

        front_stats_group = QGroupBox("正面工位生产统计")
        front_stats_layout = QHBoxLayout()
        front_stats_layout.setContentsMargins(8, 8, 8, 8)
        front_stats_layout.setSpacing(10)
        front_stats_form = QFormLayout()
        self.label_total = QLabel("0")
        self.label_good = QLabel("0")
        self.label_rate = QLabel("0.00%")
        self.label_rate.setFont(QFont("Arial", 16, QFont.Bold))
        front_stats_form.addRow("生产总数:", self.label_total)
        front_stats_form.addRow("良品总数:", self.label_good)
        front_stats_form.addRow("良率：", self.label_rate)
        front_stats_layout.addLayout(front_stats_form)
        self.front_ok_bar = QFrame()
        self.front_ok_bar.setStyleSheet("background-color: #2ecc71;")
        self.front_ok_bar.setFixedHeight(14)
        self.front_ng_bar = QFrame()
        self.front_ng_bar.setStyleSheet("background-color: #f1c40f;")
        self.front_ng_bar.setFixedHeight(14)
        front_bar_layout = QHBoxLayout()
        front_bar_layout.setSpacing(0)
        front_bar_layout.setContentsMargins(0, 0, 0, 0)
        front_bar_layout.addWidget(self.front_ok_bar)
        front_bar_layout.addWidget(self.front_ng_bar)
        front_bar_container = QWidget()
        front_bar_container.setLayout(front_bar_layout)
        front_bar_container.setFixedWidth(140)
        front_stats_layout.addWidget(front_bar_container)

        self.btn_clear_front_stats = QPushButton("清零统计")
        self.btn_clear_front_stats.setFixedHeight(36)
        self.btn_clear_front_stats.clicked.connect(self.clear_front_stats)
        front_stats_layout.addWidget(self.btn_clear_front_stats)
        front_stats_layout.addStretch()
        front_stats_group.setLayout(front_stats_layout)
        front_stats_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        side_stats_group = QGroupBox("侧面工位生产统计")
        side_stats_layout = QHBoxLayout()
        side_stats_layout.setContentsMargins(8, 8, 8, 8)
        side_stats_layout.setSpacing(10)
        side_stats_form = QFormLayout()
        self.side_label_total = QLabel("0")
        self.side_label_good = QLabel("0")
        self.side_label_rate = QLabel("0.00%")
        self.side_label_rate.setFont(QFont("Arial", 16, QFont.Bold))
        side_stats_form.addRow("生产总数:", self.side_label_total)
        side_stats_form.addRow("良品总数:", self.side_label_good)
        side_stats_form.addRow("良率：", self.side_label_rate)
        side_stats_layout.addLayout(side_stats_form)
        self.side_ok_bar = QFrame()
        self.side_ok_bar.setStyleSheet("background-color: #2ecc71;")
        self.side_ok_bar.setFixedHeight(14)
        self.side_ng_bar = QFrame()
        self.side_ng_bar.setStyleSheet("background-color: #f1c40f;")
        self.side_ng_bar.setFixedHeight(14)
        side_bar_layout = QHBoxLayout()
        side_bar_layout.setSpacing(0)
        side_bar_layout.setContentsMargins(0, 0, 0, 0)
        side_bar_layout.addWidget(self.side_ok_bar)
        side_bar_layout.addWidget(self.side_ng_bar)
        side_bar_container = QWidget()
        side_bar_container.setLayout(side_bar_layout)
        side_bar_container.setFixedWidth(140)
        side_stats_layout.addWidget(side_bar_container)
        self.btn_clear_side_stats = QPushButton("清零统计")
        self.btn_clear_side_stats.setFixedHeight(36)
        self.btn_clear_side_stats.clicked.connect(self.clear_side_stats)
        side_stats_layout.addWidget(self.btn_clear_side_stats)
        side_stats_layout.addStretch()
        side_stats_group.setLayout(side_stats_layout)
        side_stats_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mid_layout.addWidget(front_stats_group)
        mid_layout.addWidget(side_stats_group)

        display_layout.addLayout(bottom_layout, stretch=1)

        tab_widget.addTab(page_display, "显示")
        tab_widget.addTab(params_page, "设置参数")


        ## 快捷键设置：空格/回车触发检测    
        space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        space_shortcut.activated.connect(lambda: self._start_detection(source="MANUAL"))
        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(lambda: self._start_detection(source="MANUAL"))
        enter_shortcut2 = QShortcut(QKeySequence(Qt.Key_Enter), self)
        enter_shortcut2.activated.connect(lambda: self._start_detection(source="MANUAL"))

    ## 配置保存/载入
    def save_config(self):
        self.plc_setting = self.build_plc_setting()
        config = {
            "conf": self.conf_edit.text(),
            "mean_gray_min": self.mean_gray_min_edit.text(),
            "mean_gray_max": self.mean_gray_max_edit.text(),
            "tgray": self.tgray_edit.text(),
            "tarea": self.tarea_edit.text(),
            "tgray1": self.tgray1_edit.text(),
            "tarea1": self.tarea1_edit.text(),
            "ai_min_area": self.ai_area_edit.text(),
            "side_conf": self.side_conf_edit.text(),
            "side_min_area": self.side_area_edit.text(),
            "front_roi_row1": self.front_roi_row1_edit.text(),
            "front_roi_col1": self.front_roi_col1_edit.text(),
            "front_roi_row2": self.front_roi_row2_edit.text(),
            "front_roi_col2": self.front_roi_col2_edit.text(),
            "side_roi_row1": self.side_roi_row1_edit.text(),
            "side_roi_col1": self.side_roi_col1_edit.text(),
            "side_roi_row2": self.side_roi_row2_edit.text(),
            "side_roi_col2": self.side_roi_col2_edit.text(),
            "front_roi_rotate_angle": self.front_roi_rotate_edit.text(),
            "front_flip_horizontal": self.front_flip_horizontal_checkbox.isChecked(),
            "front_flip_vertical": self.front_flip_vertical_checkbox.isChecked(),
            "side_roi_rotate_angle": self.side_roi_rotate_edit.text(),
            "side_flip_horizontal": self.side_flip_horizontal_checkbox.isChecked(),
            "side_flip_vertical": self.side_flip_vertical_checkbox.isChecked(),
            "save_raw_image": self.save_raw_image_checkbox.isChecked(),
            "save_roi_image": self.save_roi_image_checkbox.isChecked(),
            "plc_trigger_front": self.plc_trigger_front_edit.text(),
            "plc_trigger_side": self.plc_trigger_side_edit.text(),
            "plc_result_front": self.plc_result_front_edit.text(),
            "plc_done_front": self.plc_done_front_edit.text(),
            "plc_result_side": self.plc_result_side_edit.text(),
            "plc_done_side": self.plc_done_side_edit.text(),
            "plc_setting": self.plc_setting,
            "front_cam_ip": self.get_front_cam_ip(),
            "side_cam_ip": self.get_side_cam_ip(),
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.log_text.append("参数保存成功")
            QMessageBox.information(self, "提示", "参数保存成功")
        except Exception as e:
            self.log_text.append(f"保存失败：{str(e)}")

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            self.log_text.append("未找到配置文件，使用默认参数")
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.conf_edit.setText(config.get("conf", "0.7"))
            self.mean_gray_min_edit.setText(config.get("mean_gray_min", "10"))
            self.mean_gray_max_edit.setText(config.get("mean_gray_max", "245"))
            self.tgray_edit.setText(config.get("tgray", "254"))
            self.tarea_edit.setText(config.get("tarea", "45"))
            self.tgray1_edit.setText(config.get("tgray1", "240"))
            self.tarea1_edit.setText(config.get("tarea1", "200"))
            self.ai_area_edit.setText(config.get("ai_min_area", "3000"))
            self.side_conf_edit.setText(config.get("side_conf", "0.25"))
            self.side_area_edit.setText(config.get("side_min_area", "500"))
            self.front_roi_row1_edit.setText(config.get("front_roi_row1", "684"))
            self.front_roi_col1_edit.setText(config.get("front_roi_col1", "808"))
            self.front_roi_row2_edit.setText(config.get("front_roi_row2", "1076"))
            self.front_roi_col2_edit.setText(config.get("front_roi_col2", "1582"))
            self.side_roi_row1_edit.setText(config.get("side_roi_row1", "680"))
            self.side_roi_col1_edit.setText(config.get("side_roi_col1", "800"))
            self.side_roi_row2_edit.setText(config.get("side_roi_row2", "1080"))
            self.side_roi_col2_edit.setText(config.get("side_roi_col2", "1580"))
            self.front_roi_rotate_edit.setText(str(config.get("front_roi_rotate_angle", "0")))
            self.front_flip_horizontal_checkbox.setChecked(self._safe_bool_from_config(config.get("front_flip_horizontal", False)))
            self.front_flip_vertical_checkbox.setChecked(self._safe_bool_from_config(config.get("front_flip_vertical", False)))
            self.side_roi_rotate_edit.setText(str(config.get("side_roi_rotate_angle", "0")))
            self.side_flip_horizontal_checkbox.setChecked(self._safe_bool_from_config(config.get("side_flip_horizontal", False)))
            self.side_flip_vertical_checkbox.setChecked(self._safe_bool_from_config(config.get("side_flip_vertical", False)))
            self.save_raw_image_checkbox.setChecked(self._safe_bool_from_config(config.get("save_raw_image", False)))
            self.save_roi_image_checkbox.setChecked(self._safe_bool_from_config(config.get("save_roi_image", False)))
            self.plc_trigger_front_edit.setText(config.get("plc_trigger_front", "R0000"))
            self.plc_trigger_side_edit.setText(config.get("plc_trigger_side", "R0001"))
            self.plc_result_front_edit.setText(config.get("plc_result_front", "R0502"))
            self.plc_done_front_edit.setText(config.get("plc_done_front", "R0501"))
            self.plc_result_side_edit.setText(config.get("plc_result_side", "R0504"))
            self.plc_done_side_edit.setText(config.get("plc_done_side", "R0503"))
            self.plc_setting = config.get("plc_setting", "")
            self.front_cam_ip_combo.setCurrentText(config.get("front_cam_ip", ""))
            self.side_cam_ip_combo.setCurrentText(config.get("side_cam_ip", ""))
            self.apply_plc_setting(self.plc_setting)
            self.conf_slider.setValue(int(float(config.get("conf", "0.7")) * 100))
            self.mean_gray_min_slider.setValue(int(float(config.get("mean_gray_min", "10"))))
            self.mean_gray_max_slider.setValue(int(float(config.get("mean_gray_max", "245"))))
            self.tgray_slider.setValue(int(float(config.get("tgray", "254"))))
            self.tarea_slider.setValue(int(float(config.get("tarea", "45"))))
            self.tgray1_slider.setValue(int(float(config.get("tgray1", "240"))))
            self.tarea1_slider.setValue(int(float(config.get("tarea1", "200"))))
            self.ai_area_slider.setValue(int(float(config.get("ai_min_area", "3000"))))
            self.side_conf_slider.setValue(int(float(config.get("side_conf", "0.25")) * 100))
            self.side_area_slider.setValue(int(float(config.get("side_min_area", "500"))))
            self.log_text.append("参数加载成功")
        except Exception as e:
            self.log_text.append(f"载入失败：{str(e)}")

    # 图像处理方法
    def _safe_int_from_edit(self, edit: QLineEdit, default: int) -> int:
        try:
            return int(float(edit.text().strip()))
        except Exception:
            return default

    def _safe_float_from_edit(self, edit: QLineEdit, default: float) -> float:
        try:
            return float(edit.text().strip())
        except Exception:
            return default

    def _safe_bool_from_config(self, value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return default

    def get_front_roi_values(self):
        return (
            self._safe_int_from_edit(self.front_roi_row1_edit, 684),
            self._safe_int_from_edit(self.front_roi_col1_edit, 808),
            self._safe_int_from_edit(self.front_roi_row2_edit, 1076),
            self._safe_int_from_edit(self.front_roi_col2_edit, 1582),
        )

    def get_side_roi_values(self):
        return (
            self._safe_int_from_edit(self.side_roi_row1_edit, 680),
            self._safe_int_from_edit(self.side_roi_col1_edit, 800),
            self._safe_int_from_edit(self.side_roi_row2_edit, 1080),
            self._safe_int_from_edit(self.side_roi_col2_edit, 1580),
        )

    def get_front_roi_transform(self):
        return {
            "rotate_angle": self._safe_float_from_edit(self.front_roi_rotate_edit, 0.0),
            "flip_horizontal": self.front_flip_horizontal_checkbox.isChecked(),
            "flip_vertical": self.front_flip_vertical_checkbox.isChecked(),
        }

    def get_side_roi_transform(self):
        return {
            "rotate_angle": self._safe_float_from_edit(self.side_roi_rotate_edit, 0.0),
            "flip_horizontal": self.side_flip_horizontal_checkbox.isChecked(),
            "flip_vertical": self.side_flip_vertical_checkbox.isChecked(),
        }

    def _apply_post_roi_transform(
        self,
        src: np.ndarray,
        rotate_angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> np.ndarray:
        if src is None or src.size == 0:
            return np.empty((0, 0), dtype=np.uint8)
        out = src.copy()
        angle = float(rotate_angle or 0.0)
        if abs(angle) % 360.0 > 1e-6:
            h, w = out.shape[:2]
            center = (w / 2.0, h / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos_v = abs(rot_mat[0, 0])
            sin_v = abs(rot_mat[0, 1])
            new_w = int((h * sin_v) + (w * cos_v))
            new_h = int((h * cos_v) + (w * sin_v))
            rot_mat[0, 2] += (new_w / 2.0) - center[0]
            rot_mat[1, 2] += (new_h / 2.0) - center[1]
            border = 0 if out.ndim == 2 else (0, 0, 0)
            out = cv2.warpAffine(out, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        if flip_horizontal and flip_vertical:
            out = cv2.flip(out, -1)
        elif flip_horizontal:
            out = cv2.flip(out, 1)
        elif flip_vertical:
            out = cv2.flip(out, 0)
        return out

    def _crop_by_roi(
        self,
        src: np.ndarray,
        roi_values,
        rotate_code=None,
        rotate_angle: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> np.ndarray:
        if src is None or src.size == 0:
            return np.empty((0, 0), dtype=np.uint8)
        work_img = cv2.rotate(src, rotate_code) if rotate_code is not None else src
        if not roi_values or len(roi_values) != 4:
            return np.empty((0, 0, src.shape[2] if src.ndim == 3 else 1), dtype=src.dtype)
        row1, col1, row2, col2 = [int(v) for v in roi_values]
        h, w = work_img.shape[:2]
        row1 = max(0, min(row1, h))
        row2 = max(0, min(row2, h))
        col1 = max(0, min(col1, w))
        col2 = max(0, min(col2, w))
        if row2 <= row1 or col2 <= col1:
            return np.empty((0, 0, src.shape[2] if src.ndim == 3 else 1), dtype=src.dtype)
        roi = work_img[row1:row2, col1:col2]
        return self._apply_post_roi_transform(
            roi,
            rotate_angle=rotate_angle,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )

    def _save_image(self, folder_name: str, file_name: str, image: np.ndarray, station: str = "") -> str:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(root_dir, folder_name, station) if station else os.path.join(root_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        ok = cv2.imwrite(save_path, image)
        if not ok:
            raise Exception(f"图像保存失败: {save_path}")
        return save_path

    def save_detection_images(self, station: str, raw_img: np.ndarray, roi_img: np.ndarray, save_raw: bool, save_roi: bool):
        station = station.strip().lower()
        if station not in ("front", "side"):
            station = "front"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_path = None
        roi_path = None
        if save_raw and raw_img is not None and raw_img.size > 0:
            raw_name = f"{station}_raw_{ts}.bmp"
            raw_path = self._save_image(IMAGE_DIR, raw_name, raw_img, station=station)
        if save_roi and roi_img is not None and roi_img.size > 0:
            roi_name = f"{station}_roi_{ts}.bmp"
            roi_path = self._save_image(ROI_IMAGE_DIR, roi_name, roi_img, station=station)
        return raw_path, roi_path

    def try_save_detection_images(self, station: str, raw_img: np.ndarray, roi_img: np.ndarray, save_raw: bool, save_roi: bool):
        try:
            raw_path, roi_path = self.save_detection_images(station, raw_img, roi_img, save_raw, save_roi)
            return raw_path, roi_path, None
        except Exception as e:
            return None, None, str(e)

    def rotate_and_crop(self, src: np.ndarray) -> np.ndarray:
        transform = self.get_front_roi_transform()
        return self.crop_front_roi(
            src,
            self.get_front_roi_values(),
            rotate_angle=transform["rotate_angle"],
            flip_horizontal=transform["flip_horizontal"],
            flip_vertical=transform["flip_vertical"],
        )

    def rotate_and_crop_side(self, src: np.ndarray) -> np.ndarray:
        transform = self.get_side_roi_transform()
        return self.crop_side_roi(
            src,
            self.get_side_roi_values(),
            rotate_angle=transform["rotate_angle"],
            flip_horizontal=transform["flip_horizontal"],
            flip_vertical=transform["flip_vertical"],
        )

    def crop_front_roi(
        self,
        src: np.ndarray,
        roi_values=None,
        rotate_angle: float = None,
        flip_horizontal: bool = None,
        flip_vertical: bool = None,
    ) -> np.ndarray:
        transform = self.get_front_roi_transform() if (rotate_angle is None or flip_horizontal is None or flip_vertical is None) else None
        return self._crop_by_roi(
            src,
            roi_values or self.get_front_roi_values(),
            rotate_code=cv2.ROTATE_180,
            rotate_angle=transform["rotate_angle"] if rotate_angle is None else rotate_angle,
            flip_horizontal=transform["flip_horizontal"] if flip_horizontal is None else flip_horizontal,
            flip_vertical=transform["flip_vertical"] if flip_vertical is None else flip_vertical,
        )

    def crop_side_roi(
        self,
        src: np.ndarray,
        roi_values=None,
        rotate_angle: float = None,
        flip_horizontal: bool = None,
        flip_vertical: bool = None,
    ) -> np.ndarray:
        transform = self.get_side_roi_transform() if (rotate_angle is None or flip_horizontal is None or flip_vertical is None) else None
        return self._crop_by_roi(
            src,
            roi_values or self.get_side_roi_values(),
            rotate_code=None,
            rotate_angle=transform["rotate_angle"] if rotate_angle is None else rotate_angle,
            flip_horizontal=transform["flip_horizontal"] if flip_horizontal is None else flip_horizontal,
            flip_vertical=transform["flip_vertical"] if flip_vertical is None else flip_vertical,
        )

    def check_mean_gray_ng(self, image: np.ndarray):
        if image is None or image.size == 0:
            raise ValueError("待检测图像无效")
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        mean_gray = float(np.mean(gray))
        mean_gray_min = self._safe_int_from_edit(self.mean_gray_min_edit, 10)
        mean_gray_max = self._safe_int_from_edit(self.mean_gray_max_edit, 245)
        if mean_gray_min > mean_gray_max:
            mean_gray_min, mean_gray_max = mean_gray_max, mean_gray_min
        is_ng = mean_gray < mean_gray_min or mean_gray > mean_gray_max
        return is_ng, mean_gray

    def yolo_detect_with_params(self, frame: np.ndarray, conf: float, ai_min_area: int):
        results = self._run_yolo(self.model, frame, conf=conf, iou=0.9, verbose=False)
        plot_rgb = results[0].plot()
        plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)
        significant = False
        masks = results[0].masks
        if masks is not None and len(masks.xy) > 0:
            for xy in masks.xy:
                if len(xy) < 3:
                    continue
                contour = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
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
        ai_min_area = int(float(self.ai_area_edit.text()))
        return self.yolo_detect_with_params(frame, conf=conf, ai_min_area=ai_min_area)

    def side_detect_with_params(self, frame: np.ndarray, conf: float, min_area: int):
        """侧面检测: 掩膜为空=OK，掩膜非空=NG(参考yolo_detect_with_params)"""
        results = self._run_yolo(self.side_model, frame, conf=conf, iou=0.9, verbose=False)
        plot_rgb = results[0].plot()
        plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)
        has_mask = False
        masks = results[0].masks
        if masks is not None and len(masks.xy) > 0:
            for xy in masks.xy:
                if len(xy) < 3:
                    continue
                contour = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
                area = cv2.contourArea(contour)
                if area >= min_area:
                    has_mask = True
                    break
        if has_mask:
            display_img = plot_bgr
            side_ok = False
        else:
            display_img = frame.copy()
            side_ok = True
        return display_img, side_ok

    def traditional_detect_with_params(self, src: np.ndarray, tgray: int, tarea: int, tgray1: int, tarea1: int):
        if len(src.shape) == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src.copy()
        results = self._run_yolo(self.seg_model, src, conf=0.25, iou=0.7, verbose=False)[0]
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
        sell = np.zeros(src.shape[:2], dtype=np.uint8)
        for c in contours:
            if cv2.contourArea(c) >= tarea:
                cv2.drawContours(sell, [c], -1, 255, -1)
        _, bright_small1 = cv2.threshold(reduced, tgray1, 255, cv2.THRESH_BINARY)
        contours1, _ = cv2.findContours(bright_small1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c1 in contours1:
            if cv2.contourArea(c1) >= tarea1:
                cv2.drawContours(sell, [c1], -1, 255, -1)
        contours_final, _ = cv2.findContours(sell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = src.copy()
        for c in contours_final:
            cv2.drawContours(overlay, [c], -1, (0, 0, 255), 2)
        ok = (len(contours_final) == 0)
        return overlay, ok

    def traditional_detect(self, frame: np.ndarray):
        tgray = int(self.tgray_edit.text())
        tarea = int(self.tarea_edit.text())
        tgray1 = int(self.tgray1_edit.text())
        tarea1 = int(self.tarea1_edit.text())
        return self.traditional_detect_with_params(frame, tgray, tarea, tgray1, tarea1)

    def update_result_text(self, label: QLabel, ok: bool):
        if ok:
            label.setText("OK")
            label.setStyleSheet("background-color: #00ff00; color: black; border: 2px solid #a0c4ff; border-radius: 8px;")
        else:
            label.setText("NG")
            label.setStyleSheet("background-color: #ff0000; color: white; border: 2px solid #a0c4ff; border-radius: 8px;")

    # ??????
    def _start_detection(self, source: str = "MANUAL"):
        if self._plc_busy:
            self.log_text.append(f"{source}触发被忽略，正面检测进行中")
            return
        self._plc_busy = True
        try:
            front_transform = self.get_front_roi_transform()
            params = {
                "conf": float(self.conf_edit.text()),
                "ai_min_area": int(float(self.ai_area_edit.text())),
                "tgray": int(float(self.tgray_edit.text())),
                "tarea": int(float(self.tarea_edit.text())),
                "tgray1": int(float(self.tgray1_edit.text())),
                "tarea1": int(float(self.tarea1_edit.text())),
                "front_roi": self.get_front_roi_values(),
                "front_rotate_angle": float(front_transform["rotate_angle"]),
                "front_flip_horizontal": bool(front_transform["flip_horizontal"]),
                "front_flip_vertical": bool(front_transform["flip_vertical"]),
                "save_raw_image": self.save_raw_image_checkbox.isChecked(),
                "save_roi_image": self.save_roi_image_checkbox.isChecked(),
            }
        except Exception as e:
            self.log_text.append(f"正面参数读取失败: {e}")
            self._plc_busy = False
            return
        self.btn_detect.setEnabled(False)
        self.log_text.append(f"{source} 启动正面检测..")
        self._detect_thread = DetectionThread(self, params)
        self._detect_thread.result_ready.connect(self._on_detection_result)
        self._detect_thread.error.connect(self._on_detection_error)
        self._detect_thread.finished.connect(lambda: self.btn_detect.setEnabled(True))
        self._detect_thread.start()

    def _on_detection_result(self, r: dict):
        try:
            final_ok = r["final_ok"]
            yolo_img = draw_chinese_text(r["yolo_img"], "AI检测安装到位", (10, 10), 20, (0, 255, 0), 2)
            trad_img = draw_chinese_text(r["trad_img"], "白点灰尘检测", (10, 10), 20, (255, 0, 0), 2)
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_ai, r["ai_ok"])
            self.update_result_text(self.text_trad, r["trad_ok"])
            self.update_result_text(self.text_final, final_ok)
            self.update_stats(final_ok)
            self._plc_report_result(final_ok)
            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 正面检测完成 | "
                f"AI安装检测: {'OK' if r['ai_ok'] else 'NG'} | "
                f"白点灰尘检测: {'OK' if r['trad_ok'] else 'NG'} | "
                f"综合: {'OK' if final_ok else 'NG'} | "
                f"AI耗时: {r['ai_time']:.3f}s | 白点耗时: {r['trad_time']:.3f}s"
            )
            if r.get("gray_ng"):
                self.log_text.append(f"正面检测灰度预检NG，平均灰度: {r['mean_gray']:.2f}")
            if r.get("save_error"):
                self.log_text.append(f"图像保存失败(front): {r['save_error']}")
            if r.get("saved_raw_path"):
                self.log_text.append(f"已保存原图: {r['saved_raw_path']}")
            if r.get("saved_roi_path"):
                self.log_text.append(f"已保存ROI: {r['saved_roi_path']}")
        finally:
            self._plc_busy = False
            self._detect_thread = None

    def _on_detection_error(self, msg: str):
        try:
            self.log_text.append(f"正面检测错误: {msg}")
        finally:
            self._plc_busy = False
            self._detect_thread = None

    def _plc_report_result(self, final_ok: bool):
        """正面检测结果回写"""
        if not self.plc or not self.plc.is_connected:
            return
        try:
            result_addr = self._get_plc_addr(self.plc_result_front_edit, "R1207")
            done_addr = self._get_plc_addr(self.plc_done_front_edit, "R1206")
            
            # 写结果: OK=0, NG=1
            self.plc.write_bit(result_addr, not final_ok)
            # 写完成脉冲
            self.plc.write_bit(done_addr, True)
            #QTimer.singleShot(150, lambda: self.plc.write_bit(done_addr, False) if self.plc else None)
        except Exception as e:
            self.log_text.append(f"PLC回写异常(正面): {e}")

    # 侧面检测结果回写
    def _plc_report_side_result(self, side_ok: bool):
        """侧面检测结果回写"""
        if not self.plc or not self.plc.is_connected:
            return
        try:
            result_addr = self._get_plc_addr(self.plc_result_side_edit, "R120A")
            done_addr = self._get_plc_addr(self.plc_done_side_edit, "R1209")
            
            # 写结果: OK=0, NG=1
            self.plc.write_bit(result_addr, not side_ok)
            # 写完成脉冲
            self.plc.write_bit(done_addr, True)
            #QTimer.singleShot(150, lambda: self.plc.write_bit(done_addr, False) if self.plc else None)
        except Exception as e:
            self.log_text.append(f"PLC回写异常(侧面): {e}")

    def _test_plc_read(self, line_edit):
        """PLC地址手动读取测试"""
        if not self.plc or not self.plc.is_connected:
            QMessageBox.warning(self, "错误", "PLC未连接")
            return
        addr = line_edit.text().strip()
        if not addr:
            QMessageBox.warning(self, "错误", "请输入有效的PLC地址")
            return
        val = self.plc.read_bit(addr)
        QMessageBox.information(self, "读取结果", f"地址 {addr} 当前状态: {'ON (1)' if val else 'OFF (0)'}")

    def _test_plc_write(self, line_edit, val):
        """PLC地址手动写入测试"""
        if not self.plc or not self.plc.is_connected:
            QMessageBox.warning(self, "错误", "PLC未连接")
            return
        addr = line_edit.text().strip()
        if not addr:
            QMessageBox.warning(self, "错误", "请输入有效的PLC地址")
            return
        self.plc.write_bit(addr, val)
        self.log_text.append(f"手动测试写入 PLC: {addr} -> {'ON' if val else 'OFF'}")

    def _create_plc_test_row(self, line_edit):
        """为PLC地址输入框创建带有读写按钮的布局"""
        layout = QHBoxLayout()
        layout.addWidget(line_edit)
        
        btn_read = QPushButton("读取")
        btn_read.setFixedWidth(50)
        btn_read.clicked.connect(lambda: self._test_plc_read(line_edit))
        
        btn_on = QPushButton("置1")
        btn_on.setFixedWidth(50)
        btn_on.clicked.connect(lambda: self._test_plc_write(line_edit, True))
        
        btn_off = QPushButton("置0")
        btn_off.setFixedWidth(50)
        btn_off.clicked.connect(lambda: self._test_plc_write(line_edit, False))
        
        layout.addWidget(btn_read)
        layout.addWidget(btn_on)
        layout.addWidget(btn_off)
        return layout

    # 侧面检测启动
    def _start_side_detection(self, source: str = "MANUAL"):
        if self._plc_busy2:
            self.log_text.append(f"{source}触发被忽略，侧面检测进行中")
            return
        self._plc_busy2 = True
        try:
            side_transform = self.get_side_roi_transform()
            params = {
                "side_conf": float(self.side_conf_edit.text()),
                "side_min_area": int(float(self.side_area_edit.text())),
                "side_roi": self.get_side_roi_values(),
                "side_rotate_angle": float(side_transform["rotate_angle"]),
                "side_flip_horizontal": bool(side_transform["flip_horizontal"]),
                "side_flip_vertical": bool(side_transform["flip_vertical"]),
                "save_raw_image": self.save_raw_image_checkbox.isChecked(),
                "save_roi_image": self.save_roi_image_checkbox.isChecked(),
            }
        except Exception as e:
            self.log_text.append(f"侧面参数读取失败: {e}")
            self._plc_busy2 = False
            return
        self.btn_side_detect.setEnabled(False)
        self.log_text.append(f"{source} 启动侧面检测..")
        self._side_detect_thread = SideDetectionThread(self, params)
        self._side_detect_thread.result_ready.connect(self._on_side_detection_result)
        self._side_detect_thread.error.connect(self._on_side_detection_error)
        self._side_detect_thread.finished.connect(lambda: self.btn_side_detect.setEnabled(True))
        self._side_detect_thread.start()

    def _on_side_detection_result(self, r: dict):
        try:
            side_img = draw_chinese_text(r["side_img"], "AI侧面检测", (10, 10), 20, (0, 255, 0), 2)
            self.side_window.setPixmap(cv2_to_qpixmap(side_img))
            self.update_result_text(self.text_side, bool(r["side_ok"]))
            self.update_side_stats(bool(r["side_ok"]))
            self._plc_report_side_result(bool(r["side_ok"]))
            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 侧面检测完成 | "
                f"结果: {'OK' if r['side_ok'] else 'NG'} | 耗时: {r['ai_time']:.3f}s"
            )
            if r.get("gray_ng"):
                self.log_text.append(f"侧面检测灰度预检NG，平均灰度: {r['mean_gray']:.2f}")
            if r.get("save_error"):
                self.log_text.append(f"图像保存失败(side): {r['save_error']}")
            if r.get("saved_raw_path"):
                self.log_text.append(f"已保存原图: {r['saved_raw_path']}")
            if r.get("saved_roi_path"):
                self.log_text.append(f"已保存ROI: {r['saved_roi_path']}")
        finally:
            self._plc_busy2 = False
            self._side_detect_thread = None

    def _on_side_detection_error(self, msg: str):
        try:
            self.log_text.append(f"侧面检测错误: {msg}")
        finally:
            self._plc_busy2 = False
            self._side_detect_thread = None

    def _update_rate_bar(self, ok_count: int, total_count: int, ok_bar: QFrame, ng_bar: QFrame):
        total = total_count
        if total <= 0:
            ok_w = self._rate_bar_width
            ng_w = 0
        else:
            ok_w = int(self._rate_bar_width * ok_count / total)
            ng_w = max(self._rate_bar_width - ok_w, 0)
        ok_bar.setFixedWidth(ok_w)
        ng_bar.setFixedWidth(ng_w)

    def update_stats(self, final_ok: bool):
        self.total_count += 1
        if final_ok:
            self.good_count += 1
        self.label_total.setText(str(self.total_count))
        self.label_good.setText(str(self.good_count))
        rate = (self.good_count / self.total_count * 100) if self.total_count > 0 else 0.0
        self.label_rate.setText(f"{rate:.2f}%")
        self._update_rate_bar(self.good_count, self.total_count, self.front_ok_bar, self.front_ng_bar)

    def update_side_stats(self, side_ok: bool):
        self.side_total_count += 1
        if side_ok:
            self.side_good_count += 1
        self.side_label_total.setText(str(self.side_total_count))
        self.side_label_good.setText(str(self.side_good_count))
        rate = (self.side_good_count / self.side_total_count * 100) if self.side_total_count > 0 else 0.0
        self.side_label_rate.setText(f"{rate:.2f}%")
        self._update_rate_bar(self.side_good_count, self.side_total_count, self.side_ok_bar, self.side_ng_bar)

    def grab_two_frames(self):
        try:
            if not self.camera:
                self.camera = HiKCamera()
                self.camera.initialize()
            if not self.camera.is_opened:
                ip = self.get_front_cam_ip()
                self.camera.open_device_by_ip(ip)
            self.log_text.append("采集图像...")
            frame = self.camera.get_frame()
            self.log_text.append("图像采集完成")
            return frame
        except Exception as e:
            self.log_text.append(f"采集错误：{str(e)}")
            if self.camera:
                self.camera.close()
                self.camera = None
            raise

    def on_detect(self):
        try:
            frame = self.grab_two_frames()
            front_transform = self.get_front_roi_transform()
            roi = self.crop_front_roi(
                frame,
                rotate_angle=front_transform["rotate_angle"],
                flip_horizontal=front_transform["flip_horizontal"],
                flip_vertical=front_transform["flip_vertical"],
            )
            if roi.size == 0:
                self.log_text.append("ROI无效")
                return
            saved_raw_path, saved_roi_path, save_error = self.try_save_detection_images(
                station="front",
                raw_img=frame,
                roi_img=roi,
                save_raw=self.save_raw_image_checkbox.isChecked(),
                save_roi=self.save_roi_image_checkbox.isChecked(),
            )
            gray_ng, mean_gray = self.check_mean_gray_ng(roi)
            if gray_ng:
                self.show_window.setPixmap(cv2_to_qpixmap(roi))
                self.picture_box1.setPixmap(cv2_to_qpixmap(roi))
                self.update_result_text(self.text_ai, False)
                self.update_result_text(self.text_trad, False)
                self.update_result_text(self.text_final, False)
                self.update_stats(False)
                self._plc_report_result(False)
                self.log_text.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 检测完成 | 综合: NG | "
                    f"原因: 平均灰度 {mean_gray:.2f} 不在允许范围内"
                )
                if save_error:
                    self.log_text.append(f"图像保存失败(front): {save_error}")
                if saved_raw_path:
                    self.log_text.append(f"已保存原图: {saved_raw_path}")
                if saved_roi_path:
                    self.log_text.append(f"已保存ROI: {saved_roi_path}")
                self._plc_busy = False
                return
            ai_start = time.time()
            yolo_img, ai_ok = self.yolo_detect(roi.copy())
            ai_time = time.time() - ai_start
            yolo_img = draw_chinese_text(yolo_img, "AI检测安装到位", (10, 10), 20, (0, 255, 0), 2)
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.update_result_text(self.text_ai, ai_ok)
            trad_start = time.time()
            trad_img, trad_ok = self.traditional_detect(roi.copy())
            trad_time = time.time() - trad_start
            trad_img = draw_chinese_text(trad_img, "白点灰尘检测", (10, 10), 20, (255, 0, 0), 2)
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_trad, trad_ok)
            final_ok = ai_ok and trad_ok
            self.update_result_text(self.text_final, final_ok)
            self.update_stats(final_ok)
            self._plc_report_result(final_ok)
            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 检测完成 | AI安装检测: {'OK' if ai_ok else 'NG'} | "
                f"白点灰尘检测: {'OK' if trad_ok else 'NG'} | 综合: {'OK' if final_ok else 'NG'} | "
                f"AI检测安装到位耗时: {ai_time:.3f}s | 白点灰尘检测耗时: {trad_time:.3f}s"
            )
            if save_error:
                self.log_text.append(f"图像保存失败(front): {save_error}")
            if saved_raw_path:
                self.log_text.append(f"已保存原图: {saved_raw_path}")
            if saved_roi_path:
                self.log_text.append(f"已保存ROI: {saved_roi_path}")
            self._plc_busy = False
        except Exception as e:
            self.log_text.append(f"错误: {str(e)}")
            self._plc_busy = False

    def on_image_detect(self):
        if self._image_detect_thread and self._image_detect_thread.isRunning():
            self.log_text.append("图片检测正在进行，请稍候。")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片检测图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not file_path:
            return
        try:
            params = {
                "conf": float(self.conf_edit.text()),
                "ai_min_area": int(float(self.ai_area_edit.text())),
                "tgray": int(float(self.tgray_edit.text())),
                "tarea": int(float(self.tarea_edit.text())),
                "tgray1": int(float(self.tgray1_edit.text())),
                "tarea1": int(float(self.tarea1_edit.text())),
            }
        except Exception as e:
            self.log_text.append(f"图片检测参数错误: {e}")
            return
        device_used = get_configured_infer_device_label()
        self.log_text.append(f"图片检测 - YOLO推理设备: {device_used}")
        self.btn_image_detect.setEnabled(False)
        self._image_detect_thread = ImageDetectionThread(self, file_path, params)
        self._image_detect_thread.result_ready.connect(self._on_image_detect_result)
        self._image_detect_thread.error.connect(self._on_image_detect_error)
        self._image_detect_thread.finished.connect(lambda: self.btn_image_detect.setEnabled(True))
        self._image_detect_thread.start()

    def _on_image_detect_result(self, r: dict):
        try:
            file_name = os.path.basename(r.get("file_path", ""))
            if r.get("gray_ng"):
                self.show_window.setPixmap(cv2_to_qpixmap(r["yolo_img"]))
                self.picture_box1.setPixmap(cv2_to_qpixmap(r["trad_img"]))
                self.update_result_text(self.text_ai, False)
                self.update_result_text(self.text_trad, False)
                self.update_result_text(self.text_final, False)
                self.update_stats(False)
                self.log_text.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片检测完成| 文件: {file_name} | 综合: NG | 均值灰度越界: {r['mean_gray']:.2f}"
                )
                return

            yolo_img = draw_chinese_text(r["yolo_img"], "AI安装检测", (10, 10), 20, (0, 255, 0), 2)
            trad_img = draw_chinese_text(r["trad_img"], "白点灰尘检测", (10, 10), 20, (255, 0, 0), 2)
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_ai, bool(r["ai_ok"]))
            self.update_result_text(self.text_trad, bool(r["trad_ok"]))
            self.update_result_text(self.text_final, bool(r["final_ok"]))
            self.update_stats(bool(r["final_ok"]))
            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片检测完成| 文件: {file_name} | "
                f"AI: {'OK' if r['ai_ok'] else 'NG'} | 白点: {'OK' if r['trad_ok'] else 'NG'} | "
                f"综合: {'OK' if r['final_ok'] else 'NG'} | "
                f"AI检测安装到位耗时：{r['ai_time']:.3f}s | 白点灰尘检测耗时：{r['trad_time']:.3f}s"
            )
        finally:
            self._image_detect_thread = None

    def _on_image_detect_error(self, msg: str):
        try:
            self.log_text.append(f"图片检测失败: {msg}")
        finally:
            self._image_detect_thread = None

    def on_image_side_detect(self):
        if self._image_side_detect_thread and self._image_side_detect_thread.isRunning():
            self.log_text.append("图片侧面检测正在进行，请稍候。")
            return
        if self._plc_busy2:
            self.log_text.append("图片侧面检测忙碌：有侧面检测任务正在执行")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择侧面检测图片", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not file_path:
            return
        try:
            params = {
                "side_conf": float(self.side_conf_edit.text()),
                "side_min_area": int(float(self.side_area_edit.text())),
            }
        except Exception as e:
            self.log_text.append(f"侧面检测参数错误: {e}")
            return
        device_used = get_configured_infer_device_label()
        self.log_text.append(f"图片侧面检测 - YOLO推理设备: {device_used}")
        self._plc_busy2 = True
        self.btn_image_side_detect.setEnabled(False)
        self._image_side_detect_thread = ImageSideDetectionThread(
            self,
            file_path,
            side_conf=params["side_conf"],
            side_min_area=params["side_min_area"],
        )
        self._image_side_detect_thread.result_ready.connect(self._on_image_side_detect_result)
        self._image_side_detect_thread.error.connect(self._on_image_side_detect_error)
        self._image_side_detect_thread.finished.connect(lambda: self.btn_image_side_detect.setEnabled(True))
        self._image_side_detect_thread.start()

    def _on_image_side_detect_result(self, r: dict):
        try:
            file_name = os.path.basename(r.get("file_path", ""))
            if r.get("gray_ng"):
                self.side_window.setPixmap(cv2_to_qpixmap(r["side_img"]))
                self.update_result_text(self.text_side, False)
                self.update_side_stats(False)
                self._plc_report_side_result(False)
                self.log_text.append(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片侧面检测完成 | 图片: {file_name} | 结果: NG | 均值灰度越界: {r['mean_gray']:.2f}"
                )
                return

            side_img = draw_chinese_text(r["side_img"], "AI侧面检测", (10, 10), 20, (0, 255, 0), 2)
            side_ok = bool(r["side_ok"])
            self.side_window.setPixmap(cv2_to_qpixmap(side_img))
            self.update_result_text(self.text_side, side_ok)
            self.update_side_stats(side_ok)
            self._plc_report_side_result(side_ok)
            self.log_text.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片侧面检测完成 | 图片: {file_name} | "
                f"结果: {'OK' if side_ok else 'NG'} | 耗时: {r['ai_time']:.3f}s"
            )
        finally:
            self._plc_busy2 = False
            self._image_side_detect_thread = None

    def _on_image_side_detect_error(self, msg: str):
        try:
            self.log_text.append(f"图片侧面检测失败: {msg}")
        finally:
            self._plc_busy2 = False
            self._image_side_detect_thread = None

    def clear_front_stats(self):
        self.total_count = 0
        self.good_count = 0
        self.label_total.setText("0")
        self.label_good.setText("0")
        self.label_rate.setText("0.00%")
        self._update_rate_bar(self.good_count, self.total_count, self.front_ok_bar, self.front_ng_bar)
        self.log_text.append("正面工位统计已清零")

    def clear_side_stats(self):
        self.side_total_count = 0
        self.side_good_count = 0
        self.side_label_total.setText("0")
        self.side_label_good.setText("0")
        self.side_label_rate.setText("0.00%")
        self._update_rate_bar(self.side_good_count, self.side_total_count, self.side_ok_bar, self.side_ng_bar)
        self.log_text.append("侧面工位统计已清零")

    def closeEvent(self, event):
        if self._detect_thread and self._detect_thread.isRunning():
            self._detect_thread.requestInterruption()
            self._detect_thread.wait(1500)
        if self._side_detect_thread and self._side_detect_thread.isRunning():
            self._side_detect_thread.requestInterruption()
            self._side_detect_thread.wait(1500)
        if self._image_detect_thread and self._image_detect_thread.isRunning():
            self._image_detect_thread.requestInterruption()
            self._image_detect_thread.wait(1500)
        if self._image_side_detect_thread and self._image_side_detect_thread.isRunning():
            self._image_side_detect_thread.requestInterruption()
            self._image_side_detect_thread.wait(1500)
        if self._warmup_thread and self._warmup_thread.isRunning():
            self._warmup_thread.requestInterruption()
            self._warmup_thread.wait(1500)
        if self.camera:
            self.camera.close()
        if self.camera2:
            self.camera2.close()
        if self.plc:
            try:
                self.plc.close()
            except Exception:
                pass
        event.accept()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if getattr(sys, "frozen", False) and _is_pyinstaller_mp_child():
        sys.exit(0)
    if getattr(sys, "frozen", False) and (not _acquire_single_instance_lock()):
        sys.exit(0)
    atexit.register(_release_single_instance_lock)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

