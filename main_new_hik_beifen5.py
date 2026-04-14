#修复版本说明（相对原始 main_new_hik_CC_beifen3. py）
# [FIX-1] HiKCamera 有两个 close() 定义，第二个覆盖第一个且无异常处理 → 合并为一个带完整异常处理的版本
# [FIX-2] _r_index_from_addr 正则 "\\d+" 在普通字符串里无法匹配 → 改为 r"\\d+"
# [FIX-3] _create_plc_test_row 返回 QHBoxLayout, QFormLayout.addRow() 需要 QWidget
# [FIX-4] read_bit/write_bit 仅支持纯数字地址，不支持十六进制地址（如 R120A）→ 改为直接透传地址
# [FIX-5] _plc_poll 在主线程 QTimer 里做同步串口读写，串口超时 0.5s 会冻结 UI → 改为独立线程轮询
# [FIX-6] on_detect() 是已废弃的同步主线检测方法，与 _start_detection() 逻辑重复 → 移除
# [FIX-7] _run_yolo() if/else 两个分支代码完全相同 → 合并为一行
# [FIX-8] crop_front_roi/crop_side_roi 用 None 判断导致合法 False 传参被覆盖 → 改用哨兵值 _UNSET
# [FIX-9] log_text 无限追加，长时间运行内存泄漏 → 限制最大 500 行
# [FIX-10] 每次 _start_detection 重新 connect 信号，旧线程若未 GC 会多次回调 → 先 disconnect 再连接
# [FIX-11] _ensure_gpu_runtime() 无 GPU 直接 raise，程序无法使用 → 改为警告 → 降级 CPU
# [FIX-12] 状态栏 R1205/R1208 标签写死，与实际触发地址输入框不同步 → 改为动态读取输入框内容

# ========== import ==========
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
from dataclasses import dataclass
import torch
import onnxruntime as ort
from ctypes import c_ubyte, cast, POINTER
from typing import Optional
import serial.tools.list_ports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QLineEdit,
    QGroupBox, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QGridLayout, QSlider, QFormLayout, QFileDialog, QMessageBox,
    QTabWidget, QComboBox, QFrame, QLayout, QSizePolicy, QToolButton, QCheckBox)
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
#启动侧面检测
#RCSR1208
#RCSR1207
# 哨兵值，用于区分"调用方未传参"与"调用方明确传入 False"
_UNSET = object()  # [FIX-8]

def _is_pyinstaller_mp_child() -> bool:
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
    lock_path = os.path.join(tempfile.gettempdir(), "main_new_hik_CC_fixed.lock")
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

# ==================== MEWTOCOL协议实现 ====================
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
        self.read_cache = {}

    def _calc_bcc(self, cmd: str) -> str:
        bcc = 0
        for char in cmd:
            bcc ^= ord(char)
        return f"{bcc:02x}"

    def open(self, setting_str: str) -> bool:
        try:
            parts = [p.strip() for p in setting_str.split(",")]
            if len(parts) < 6:
                return False
            port = parts[0]
            baud = int(parts[1])
            bytesize = int(parts[2])
            stopbits = serial.STOPBITS_ONE if int(parts[3]) == 1 else serial.STOPBITS_TWO
            parity_map = {"0": serial.PARITY_NONE, "1": serial.PARITY_ODD, "2": serial.PARITY_EVEN}
            parity = parity_map.get(parts[4], serial.PARITY_NONE)
            self.poll_interval = int(parts[5]) / 1000.0
            self.port_settings = {
                "port": port,
                "baudrate": baud,
                "bytesize": bytesize,
                "parity": parity,
                "stopbits": stopbits,
                "timeout": 0.5
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
        if not self.is_connected or not self.serial:
            return None
        station = "EE"
        full_cmd = f"%{station}#{cmd_body}{self._calc_bcc('%' + station + '#' + cmd_body)}\r"
        with self.lock:
            try:
                self.serial.flushInput()
                self.serial.write(full_cmd.encode('ascii'))
                response = self.serial.read_until(b'\r').decode('ascii', errors='ignore')
                if not response:
                    self.timeout_count += 1
                    return None
                if "RCS" in cmd_body:
                    print(f"PLC 响应 ({cmd_body}): {response.strip()}")
                if len(response) < 4:
                    return None
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

    # [FIX-4] 地址直接透传，不强制纯数字 zfill，支持十六进制地址（如 R120A）
    def _normalize_addr(self, addr: str) -> str:
        """
        将用户输入地址规范化为 MEWTOCOL 格式。
        支持纯数字地址（R0000~R9999）和含字母的十六进制地址（R120A 等）。
        若地址不以 R 开头则自动补 R 前缀。
        """
        clean = addr.strip().upper()
        if not clean.startswith("R"):
            clean = "R" + clean
        prefix = clean[0]  # 'R'
        suffix = clean[1:]  # 地址部分，可含字母
        # 只对纯数字部分补零至 4 位；含字母时保持原样（如 120A）
        if suffix.isdigit():
            suffix = suffix.zfill(4)
        return f"{prefix}{suffix}"

    def read_bit(self, addr: str) -> bool:
        cmd_addr = self._normalize_addr(addr)
        res = self._send_command(f"RCS{cmd_addr}")
        if res and "$" in res:
            idx = res.find("$RC") + 3
            status = res[idx:idx + 1]
            val = (status == "1")
            self.read_cache[addr] = val
            return val
        return self.read_cache.get(addr, False)

    def write_bit(self, addr: str, val: bool):
        cmd_addr = self._normalize_addr(addr)
        status = "1" if val else "0"
        self._send_command(f"WCS{cmd_addr}{status}")

    def close(self):
        self.is_connected = False
        with self.lock:
            if self.serial:
                try:
                    self.serial.close()
                except Exception:
                    pass
                self.serial = None

    def get_status(self) -> dict:
        return {
            "connected": self.is_connected,
            "timeout": self.timeout_count,
            "crc_err": self.crc_err_count,
            "reconnects": self.reconnect_count,
        }

# ==================== 将PLC串口读取移出主线程QTimer，避免串口超时冻结UI。通过信号将触发状态传回主线程。 ====================
class PLCPollerThread(QThread):
    trigger_front = pyqtSignal(bool)  # 正面触发位当前电平
    trigger_side = pyqtSignal(bool)   # 侧面触发位当前电平

    def __init__(self, plc: MewtocolPLC, get_front_addr, get_side_addr, parent=None):
        super().__init__(parent)
        self.plc = plc
        self.get_front_addr = get_front_addr  # callable → str
        self.get_side_addr = get_side_addr    # callable → str
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            if self.plc and self.plc.is_connected:
                try:
                    b0 = self.plc.read_bit(self.get_front_addr())
                    b1 = self.plc.read_bit(self.get_side_addr())
                    self.trigger_front.emit(b0)
                    self.trigger_side.emit(b1)
                except Exception as e:
                    print(f"PLCPollerThread 异常: {e}")
            # 轮询间隔由PLC配置决定（默认30ms）
            interval = self.plc.poll_interval if self.plc else 0.05
            self._stop_event.wait(timeout=interval)

# ==================== 检测线程 ====================
class DetectionThread(QThread):
    result_ready = pyqtSignal(object)
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
                flip_horizontal=self.params.get("front_flip_horizontal", False),
                flip_vertical=self.params.get("front_flip_vertical", False),
            )
            if roi is None or roi.size == 0:
                raise Exception("ROI无效")
            save_raw = bool(self.params.get("save_raw_image", False))
            save_roi = bool(self.params.get("save_roi_image", False))
            saved_raw_path, saved_roi_path, save_error = self.window.try_save_detection_images(
                station="front", raw_img=frame, roi_img=roi,
                save_raw=save_raw, save_roi=save_roi,
            )
            gray_ng, mean_gray = self.window.check_mean_gray_ng(roi)
            if gray_ng:
                self.result_ready.emit({
                    "yolo_img": roi.copy(), "ai_ok": False, "ai_time": 0.0,
                    "trad_img": roi.copy(), "trad_ok": False, "trad_time": 0.0,
                    "final_ok": False, "saved_raw_path": saved_raw_path,
                    "saved_roi_path": saved_roi_path, "save_error": save_error,
                    "gray_ng": True, "mean_gray": mean_gray,
                })
                return
            ai_start = time.time()
            yolo_img, ai_ok = self.window.yolo_detect_with_params(
                roi.copy(), conf=self.params["conf"], ai_min_area=self.params["ai_min_area"],
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return
            trad_start = time.time()
            trad_img, trad_ok = self.window.traditional_detect_with_params(
                roi.copy(),
                tgray=self.params["tgray"], tarea=self.params["tarea"],
                tgray1=self.params["tgray1"], tarea1=self.params["tarea1"],
            )
            trad_time = time.time() - trad_start
            if self.isInterruptionRequested():
                return
            final_ok = bool(ai_ok and trad_ok)
            self.result_ready.emit({
                "yolo_img": yolo_img, "ai_ok": ai_ok, "ai_time": ai_time,
                "trad_img": trad_img, "trad_ok": trad_ok, "trad_time": trad_time,
                "final_ok": final_ok, "saved_raw_path": saved_raw_path,
                "saved_roi_path": saved_roi_path, "save_error": save_error,
            })
        except Exception as e:
            self.error.emit(str(e))

class SideDetectionThread(QThread):
    result_ready = pyqtSignal(object)
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
                flip_horizontal=self.params.get("side_flip_horizontal", False),
                flip_vertical=self.params.get("side_flip_vertical", False),
            )
            if roi is None or roi.size == 0:
                raise Exception("侧面ROI无效")
            save_raw = bool(self.params.get("save_raw_image", False))
            save_roi = bool(self.params.get("save_roi_image", False))
            saved_raw_path, saved_roi_path, save_error = self.window.try_save_detection_images(
                station="side", raw_img=frame, roi_img=roi,
                save_raw=save_raw, save_roi=save_roi,
            )
            gray_ng, mean_gray = self.window.check_mean_gray_ng(roi)
            if gray_ng:
                self.result_ready.emit({
                    "side_img": roi.copy(), "side_ok": False, "ai_time": 0.0,
                    "saved_raw_path": saved_raw_path, "saved_roi_path": saved_roi_path,
                    "save_error": save_error, "gray_ng": True, "mean_gray": mean_gray,
                })
                return
            ai_start = time.time()
            side_img, side_ok = self.window.side_detect_with_params(
                roi.copy(), conf=self.params["side_conf"], min_area=self.params["side_min_area"],
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return
            self.result_ready.emit({
                "side_img": side_img, "side_ok": side_ok, "ai_time": ai_time,
                "saved_raw_path": saved_raw_path, "saved_roi_path": saved_roi_path,
                "save_error": save_error,
            })
        except Exception as e:
            self.error.emit(str(e))

class ImageDetectionThread(QThread):
    result_ready = pyqtSignal(object)
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
                self.result_ready.emit({
                    "file_path": self.file_path,
                    "yolo_img": frame.copy(), "ai_ok": False, "ai_time": 0.0,
                    "trad_img": frame.copy(), "trad_ok": False, "trad_time": 0.0,
                    "final_ok": False, "gray_ng": True, "mean_gray": mean_gray,
                })
                return
            ai_start = time.time()
            yolo_img, ai_ok = self.window.yolo_detect_with_params(
                frame.copy(), conf=self.params["conf"], ai_min_area=self.params["ai_min_area"],
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return
            trad_start = time.time()
            trad_img, trad_ok = self.window.traditional_detect_with_params(
                frame.copy(),
                tgray=self.params["tgray"], tarea=self.params["tarea"],
                tgray1=self.params["tgray1"], tarea1=self.params["tarea1"],
            )
            trad_time = time.time() - trad_start
            if self.isInterruptionRequested():
                return
            final_ok = bool(ai_ok and trad_ok)
            self.result_ready.emit({
                "file_path": self.file_path,
                "yolo_img": yolo_img, "ai_ok": ai_ok, "ai_time": ai_time,
                "trad_img": trad_img, "trad_ok": trad_ok, "trad_time": trad_time,
                "final_ok": final_ok,
            })
        except Exception as e:
            self.error.emit(str(e))

class ImageSideDetectionThread(QThread):
    result_ready = pyqtSignal(object)
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
                self.result_ready.emit({
                    "file_path": self.file_path,
                    "side_img": frame.copy(), "side_ok": False, "ai_time": 0.0,
                    "gray_ng": True, "mean_gray": mean_gray,
                })
                return
            ai_start = time.time()
            side_img, side_ok = self.window.side_detect_with_params(
                frame.copy(), conf=self.side_conf, min_area=self.side_min_area,
            )
            ai_time = time.time() - ai_start
            if self.isInterruptionRequested():
                return
            self.result_ready.emit({
                "file_path": self.file_path,
                "side_img": side_img, "side_ok": bool(side_ok), "ai_time": ai_time,
            })
        except Exception as e:
            self.error.emit(str(e))

class ModelWarmupThread(QThread):
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
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

# ==================== 海康SDK导入 ====================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MvImport'))
try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    from MvErrorDefine_const import *
except ImportError:
    print("SDK missing")
    sys.exit(1)

# ==================== 全局常量 ====================
CONFIG_PATH = "config.json"
MODEL_PATH = "best-seg.onnx"
SEG_MODEL_PATH = "best_seg_DW.onnx"
SIDE_MODEL_PATH = "best_CM.onnx"
IMAGE_DIR = "image"
ROI_IMAGE_DIR = "imageROI"
YOLO_DEVICE = 0
USE_ONNX_MODELS = any(str(p).lower().endswith(".onnx") for p in [MODEL_PATH, SEG_MODEL_PATH, SIDE_MODEL_PATH])
@dataclass(frozen=True)
class StationSpec:
    key: str
    display_name: str
    busy_attr: str
    prev_trigger_attr: str
    live_thread_attr: str
    image_thread_attr: str
    manual_button_attr: str
    image_button_attr: Optional[str]

LOG_MAX_LINES = 500  # [FIX-9] 日志最大行数


def resolve_runtime_infer_device():
    """
    Resolve runtime inference device for Ultralytics.
    Falls back to CPU automatically when CUDA runtime is unavailable.
    """
    if USE_ONNX_MODELS:
        try:
            providers = ort.get_available_providers()
        except Exception:
            providers = []
        if torch.cuda.is_available() and "CUDAExecutionProvider" in providers:
            return YOLO_DEVICE
        return "cpu"
    return YOLO_DEVICE if torch.cuda.is_available() else "cpu"

def log_model_device_info():
    try:
        providers = ort.get_available_providers()
        runtime_device = resolve_runtime_infer_device()
        print(
            f"ONNX Runtime Providers: {providers} | configured YOLO_DEVICE={YOLO_DEVICE} | "
            f"runtime_device={runtime_device}"
        )
    except Exception as e:
        print(f"Error logging device info: {e}")

def get_configured_infer_device_label() -> str:
    runtime_device = resolve_runtime_infer_device()
    if str(runtime_device).lower() == "cpu":
        return "CPU"
    if USE_ONNX_MODELS:
        return f"ONNX(device={runtime_device})"
    return f"CUDA(device={runtime_device})"

# ==================== 工具函数 ====================
def cv2_to_qpixmap(mat: np.ndarray) -> QPixmap:
    if mat is None or mat.size == 0:
        return QPixmap()
    if mat.ndim == 2:
        height, width = mat.shape
        q_img = QImage(mat.data, width, height, width, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        q_img = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)

def draw_chinese_text(img: np.ndarray, text: str, position: tuple, font_size: int = 20, color: tuple = (0, 255, 0), thickness: int = 1):
    font_path = r"C:\Windows\Fonts\simfang.ttf"
    cv_img = img.copy()
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"警告：未找到字体 {font_path}，使用默认字体（中文可能乱码）")
    rgb_color = color[::-1]
    draw.text(position, text, font=font, fill=rgb_color, stroke_width=thickness)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==================== 海康相机采集类 ====================
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
            raise Exception(f"枚举设备失败：0x{ret:X}")
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
            print("释放之前的相机资源...")
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
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(dev)
        if ret != MV_OK:
            self.cam = None
            raise Exception(f"创建设备句柄失败: 0x{ret:X}")
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            self.cam.MV_CC_DestroyHandle()
            raise Exception(f"打开设备失败: 0x{ret:X} (可能被其他程序占用)")
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != MV_OK:
            print("警告: 设置触发模式失败, 尝试继续")
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
        self.is_opened = True
        if label:
            print(f"{label} 相机打开成功")

    def open_device(self, index=0):
        if self.device_list.nDeviceNum == 0:
            self.enum_devices()
        if index >= self.device_list.nDeviceNum:
            raise Exception(f"设备索引 {index} 超出范围, 检测到 {self.device_list.nDeviceNum} 个设备")
        dev = cast(self.device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
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
        self._open_device_with_info(target, label=ip_str)

    # [FIX-1] 合并两个 close() 为一个带宽异常处理的版本
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
            img = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pt == PixelType_Gvsp_BGR8_Packed:
            img = np.frombuffer(data_buf, dtype=np.uint8, count=w * h * 3).reshape(h, w, 3)
            frame = img
        elif pt == PixelType_Gvsp_Mono8:
            img = np.frombuffer(data_buf, dtype=np.uint8, count=w * h).reshape(h, w)
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            raise Exception(f"不支持的像素格式: 0x{pt:X}")
        return frame

    def __del__(self):
        self.close()

# ==================== 主窗口类 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高更盛04T11安装检测")
        icon_path = "1.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print("警告: 未找到 1.ico 文件, 使用默认图标")
        self.resize(1600, 900)
        self.total_count = 0
        self.good_count = 0
        self.side_total_count = 0
        self.side_good_count = 0
        self._rate_bar_width = 140
        self.camera = None
        self.camera2 = None
        self.plc = None
        self.plc_setting = ""
        self.plc_prev_trigger = False
        self.plc_prev_trigger2 = False
        self._plc_busy = False
        self._plc_busy2 = False
        self._detect_thread = None
        self._side_detect_thread = None
        self._image_detect_thread = None
        self._image_side_detect_thread = None
        self._warmup_thread = None
        self._warmup_done = False
        self._plc_poller: Optional[PLCPollerThread] = None  # [FIX-5]
        self._auto_connect_started = False
        self._init_ui()
        self._update_rate_bar(self.good_count, self.total_count, self.front_ok_bar, self.front_ng_bar)
        self._update_rate_bar(self.side_good_count, self.side_total_count, self.side_ok_bar, self.side_ng_bar)
        self.load_config()
        self.refresh_plc_ports()
        self.refresh_camera_ip_list()
        # [FIX-5] 移除主线程 QTimer 轮询 PLC，改用状态刷新 timer（仅更新 UI）
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(500)
        self._update_status()
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
        QTimer.singleShot(1000, self._auto_connect_all)

    # ── [FIX-11] GPU 不可用时降级 CPU，不再直接崩溃 ────────────────────────
    def _get_station_spec(self, station: str) -> StationSpec:
        station = str(station).strip().lower()
        if station == "front":
            return StationSpec(
                key="front",
                display_name="启动正面检测",
                busy_attr="_plc_busy",
                prev_trigger_attr="plc_prev_trigger",
                live_thread_attr="_detect_thread",
                image_thread_attr="_image_detect_thread",
                manual_button_attr="btn_detect",
                image_button_attr="btn_image_detect",
            )
        if station == "side":
            return StationSpec(
                key="side",
                display_name="启动侧面检测",
                busy_attr="_plc_busy2",
                prev_trigger_attr="plc_prev_trigger2",
                live_thread_attr="_side_detect_thread",
                image_thread_attr="_image_side_detect_thread",
                manual_button_attr="btn_side_detect",
                image_button_attr="btn_image_side_detect",
            )
        raise ValueError(f"Unsupported station: {station}")

    def _get_station_busy(self, station: str) -> bool:
        spec = self._get_station_spec(station)
        return bool(getattr(self, spec.busy_attr, False))

    def _set_station_busy(self, station: str, busy: bool):
        spec = self._get_station_spec(station)
        setattr(self, spec.busy_attr, bool(busy))

    def _get_station_prev_trigger(self, station: str) -> bool:
        spec = self._get_station_spec(station)
        return bool(getattr(self, spec.prev_trigger_attr, False))

    def _set_station_prev_trigger(self, station: str, value: bool):
        spec = self._get_station_spec(station)
        setattr(self, spec.prev_trigger_attr, bool(value))

    def _get_task_thread(self, station: str, image_mode: bool = False):
        spec = self._get_station_spec(station)
        attr_name = spec.image_thread_attr if image_mode else spec.live_thread_attr
        return getattr(self, attr_name, None)

    def _set_task_thread(self, station: str, thread, image_mode: bool = False):
        spec = self._get_station_spec(station)
        attr_name = spec.image_thread_attr if image_mode else spec.live_thread_attr
        setattr(self, attr_name, thread)

    def _disconnect_thread_signals(self, thread):
        if not thread:
            return
        try:
            thread.result_ready.disconnect()
        except Exception:
            pass
        try:
            thread.error.disconnect()
        except Exception:
            pass

    def _bind_worker_thread(self, station: str, thread, result_handler, error_handler,
                            image_mode: bool = False):
        spec = self._get_station_spec(station)
        self._disconnect_thread_signals(self._get_task_thread(station, image_mode=image_mode))
        self._set_task_thread(station, thread, image_mode=image_mode)
        button_attr = spec.image_button_attr if image_mode else spec.manual_button_attr
        button = getattr(self, button_attr, None) if button_attr else None
        if button is not None:
            button.setEnabled(False)
            thread.finished.connect(lambda btn=button: btn.setEnabled(True))
        thread.result_ready.connect(result_handler)
        thread.error.connect(error_handler)
        thread.start()

    def _choose_image_file(self, title: str) -> str:
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        return file_path

    def _build_front_detection_params(self) -> dict:
        transform = self.get_front_roi_transform()
        return {
            "conf": float(self.conf_edit.text()),
            "ai_min_area": int(float(self.ai_area_edit.text())),
            "tgray": int(float(self.tgray_edit.text())),
            "tarea": int(float(self.tarea_edit.text())),
            "tgray1": int(float(self.tgray1_edit.text())),
            "tarea1": int(float(self.tarea1_edit.text())),
            "front_roi": self.get_front_roi_values(),
            "front_rotate_angle": float(transform["rotate_angle"]),
            "front_flip_horizontal": bool(transform["flip_horizontal"]),
            "front_flip_vertical": bool(transform["flip_vertical"]),
            "save_raw_image": self.save_raw_image_checkbox.isChecked(),
            "save_roi_image": self.save_roi_image_checkbox.isChecked(),
        }

    def _build_side_detection_params(self) -> dict:
        transform = self.get_side_roi_transform()
        return {
            "side_conf": float(self.side_conf_edit.text()),
            "side_min_area": int(float(self.side_area_edit.text())),
            "side_roi": self.get_side_roi_values(),
            "side_rotate_angle": float(transform["rotate_angle"]),
            "side_flip_horizontal": bool(transform["flip_horizontal"]),
            "side_flip_vertical": bool(transform["flip_vertical"]),
            "save_raw_image": self.save_raw_image_checkbox.isChecked(),
            "save_roi_image": self.save_roi_image_checkbox.isChecked(),
        }

    def _build_image_detect_params(self, station: str):
        station = str(station).strip().lower()
        if station == "front":
            return {
                "conf": float(self.conf_edit.text()),
                "ai_min_area": int(float(self.ai_area_edit.text())),
                "tgray": int(float(self.tgray_edit.text())),
                "tarea": int(float(self.tarea_edit.text())),
                "tgray1": int(float(self.tgray1_edit.text())),
                "tarea1": int(float(self.tarea1_edit.text())),
            }
        if station == "side":
            return {
                "side_conf": float(self.side_conf_edit.text()),
                "side_min_area": int(float(self.side_area_edit.text())),
            }
        raise ValueError(f"Unsupported station: {station}")

    def _start_live_detection(self, station: str, source: str, params: dict, thread_factory,
                              result_handler, error_handler):
        spec = self._get_station_spec(station)
        if self._get_station_busy(station):
            self._append_log(f"{source}触发被忽略，{spec.display_name}检测进行中")
            return
        self._set_station_busy(station, True)
        self._append_log(f"{source} 启动{spec.display_name}检测...")
        try:
            thread = thread_factory(params)
            self._bind_worker_thread(
                station, thread, result_handler, error_handler, image_mode=False
            )
        except Exception:
            self._set_station_busy(station, False)
            raise

    def _start_image_detection_task(self, station: str, dialog_title: str, busy_guard: bool,
                                    thread_factory, result_handler, error_handler):
        thread = self._get_task_thread(station, image_mode=True)
        if thread and thread.isRunning():
            self._append_log(f"图片{self._get_station_spec(station).display_name}检测正在进行，请稍候。")
            return
        if busy_guard and self._get_station_busy(station):
            self._append_log(f"图片{self._get_station_spec(station).display_name}检测忙碌：有{self._get_station_spec(station).display_name}检测任务正在执行")
            return
        file_path = self._choose_image_file(dialog_title)
        if not file_path:
            return
        params = self._build_image_detect_params(station)
        self._append_log(
            f"图片{self._get_station_spec(station).display_name}检测 - 推理设备: "
            f"{get_configured_infer_device_label()}"
        )
        if busy_guard:
            self._set_station_busy(station, True)
        try:
            thread = thread_factory(file_path, params)
            self._bind_worker_thread(
                station, thread, result_handler, error_handler, image_mode=True
            )
        except Exception:
            if busy_guard:
                self._set_station_busy(station, False)
            raise

    def _log_saved_paths(self, station: str, result: dict):
        if result.get("save_error"):
            self._append_log(f"图像保存失败({station}): {result['save_error']}")
        if result.get("saved_raw_path"):
            self._append_log(f"已保存原图: {result['saved_raw_path']}")
        if result.get("saved_roi_path"):
            self._append_log(f"已保存ROI: {result['saved_roi_path']}")

    def _ensure_gpu_runtime(self):
        if USE_ONNX_MODELS:
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                print(
                    "警告: 未检测到 ONNXRuntime CUDAExecutionProvider，"
                    "将使用 CPU 推理（速度较慢）。"
                    "如需 GPU 加速请安装 onnxruntime-gpu 并重启程序。"
                )
            return
        if not torch.cuda.is_available():
            print(
                "警告: 未检测到可用GPU(CUDA)，将使用 CPU 推理（速度较慢）。"
                "如需 GPU 加速请确认驱动及 CUDA 环境。"
            )

    # [FIX-7] 合并完全相同的两个分支
    def _run_yolo(self, yolo_model, frame: np.ndarray, **kwargs):
        runtime_device = resolve_runtime_infer_device()
        if str(runtime_device).lower() == "cpu":
            # Prevent stale CUDA env forcing CUDA path in some runtimes.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return yolo_model(frame, device=runtime_device, **kwargs)

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
        self._append_log(
            "模型预热完成 | "
            f"正面AI: {r.get('front_ai_warmup_s', 0.0):.3f}s | "
            f"白点分割: {r.get('front_trad_warmup_s', 0.0):.3f}s | "
            f"侧面AI: {r.get('side_ai_warmup_s', 0.0):.3f}s"
        )

    def _on_model_warmup_error(self, msg: str):
        self._warmup_thread = None
        self._append_log(f"模型预热失败: {msg}")

    def _auto_connect_all(self):
        if self._auto_connect_started:
            return
        self._auto_connect_started = True
        self._append_log("--- 正在执行启动自动连接 ---")
        try:
            self.connect_plc()
        except Exception as e:
            self._append_log(f"自动连接PLC失败: {e}")
        try:
            self._ensure_camera_opened(side=False)
            self._append_log("自动连接正面相机成功")
        except Exception as e:
            self._append_log(f"自动连接正面相机失败: {e}")
        try:
            self._ensure_camera_opened(side=True)
            self._append_log("自动连接侧面相机成功")
        except Exception as e:
            self._append_log(f"自动连接侧面相机失败: {e}")
        self._append_log("--- 自动连接流程结束 ---")

    # — [FIX-9] 统一日志追加入口，限制最大行数
    def _append_log(self, text: str):
        self.log_text.append(text)
        doc = self.log_text.document()
        while doc.blockCount() > LOG_MAX_LINES:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()  # 删除换行符

    # — [FIX-5] 启动/停止 PLC 轮询线程
    def _start_plc_poller(self):
        self._stop_plc_poller()
        self._plc_poller = PLCPollerThread(
            self.plc,
            get_front_addr=lambda: self._get_plc_addr(self.plc_trigger_front_edit, "R1205"),
            get_side_addr=lambda: self._get_plc_addr(self.plc_trigger_side_edit, "R1208"),
        )
        self._plc_poller.trigger_front.connect(self._on_plc_front_bit)
        self._plc_poller.trigger_side.connect(self._on_plc_side_bit)
        self._plc_poller.start()

    def _stop_plc_poller(self):
        if self._plc_poller:
            self._plc_poller.stop()
            self._plc_poller.wait(2000)
            self._plc_poller = None

    def _on_plc_front_bit(self, b0: bool):
        """接收来自轮询线程的正面触发电平，在主线程处理上升沿"""
        if b0 and not self._get_station_prev_trigger("front") and not self._get_station_busy("front"):
            self._start_detection(source="PLC")
        self._set_station_prev_trigger("front", b0)

    def _on_plc_side_bit(self, b1: bool):
        """接收来自轮询线程的侧面触发电平，在主线程处理上升沿"""
        if b1 and not self._get_station_prev_trigger("side") and not self._get_station_busy("side"):
            self._start_side_detection(source="PLC")
        self._set_station_prev_trigger("side", b1)

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
        # [FIX-12] 触发状态标签动态跟随输入框地址内容
        if hasattr(self, "label_r1205_status"):
            front_addr = self._get_plc_addr(self.plc_trigger_front_edit, "R1205")
            self.label_front_trigger_title.setText(f"正面触发({front_addr})")
            front_trigger = self._get_station_prev_trigger("front")
            self.label_r1205_status.setText("ON" if front_trigger else "OFF")
            self.label_r1205_status.setStyleSheet(
                f"color: {'#2ecc71' if front_trigger else '#555'}; font-size: 11px; font-weight: bold;"
            )
        if hasattr(self, "label_r1208_status"):
            side_addr = self._get_plc_addr(self.plc_trigger_side_edit, "R1208")
            self.label_side_trigger_title.setText(f"侧面触发({side_addr})")
            side_trigger = self._get_station_prev_trigger("side")
            self.label_r1208_status.setText("ON" if side_trigger else "OFF")
            self.label_r1208_status.setStyleSheet(
                f"color: {'#2ecc71' if side_trigger else '#555'}; font-size: 11px; font-weight: bold;"
            )
        if self.plc and getattr(self.plc, "is_connected", False):
            st = self.plc.get_status()
            self.label_plc_timeout.setText(str(st.get("timeout", 0)))
            self.label_plc_reconnect.setText(str(st.get("reconnects", 0)))
            self.label_plc_queue.setText(str(st.get("queue_size", 0)))
            self.label_plc_crc.setText(str(st.get("crc_err", 0)))
        else:
            for lbl in (self.label_plc_timeout, self.label_plc_reconnect, self.label_plc_queue, self.label_plc_crc):
                lbl.setText("0")

    def get_front_cam_ip(self) -> str:
        return self.front_cam_ip_combo.currentText().strip() if hasattr(self, "front_cam_ip_combo") else ""

    def get_side_cam_ip(self) -> str:
        return self.side_cam_ip_combo.currentText().strip() if hasattr(self, "side_cam_ip_combo") else ""

    def refresh_camera_ip_list(self):
        try:
            ips = HiKCamera.enum_gige_ip_list()
        except Exception as e:
            self._append_log(f"相机IP枚举失败: {e}")
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
            self._append_log("正面相机已打开")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"打开正面相机失败: {e}")

    def close_front_camera(self):
        try:
            if self.camera:
                self.camera.close()
            self._append_log("正面相机已关闭")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"关闭正面相机失败: {e}")

    def open_side_camera(self):
        try:
            self._ensure_camera_opened(side=True)
            self._append_log("侧面相机已打开")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"打开侧面相机失败: {e}")

    def close_side_camera(self):
        try:
            if self.camera2:
                self.camera2.close()
            self._append_log("侧面相机已关闭")
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

    def _test_plc_read(self, line_edit: QLineEdit):
        if not self.plc or not self.plc.is_connected:
            QMessageBox.warning(self, "提示", "PLC未连接")
            return
        addr = line_edit.text().strip()
        if not addr:
            QMessageBox.warning(self, "提示", "请输入要读取的 PLC 地址")
            return
        val = self.plc.read_bit(addr)
        QMessageBox.information(
            self,
            "读取结果",
            f"地址 {addr} 当前值: {'ON (1)' if val else 'OFF (0)'}"
        )

    def _test_plc_write(self, line_edit: QLineEdit, val: bool):
        if not self.plc or not self.plc.is_connected:
            QMessageBox.warning(self, "提示", "PLC未连接")
            return
        addr = line_edit.text().strip()
        if not addr:
            QMessageBox.warning(self, "提示", "请输入要写入的 PLC 地址")
            return
        self.plc.write_bit(addr, val)
        self._append_log(f"手动测试写入 PLC: {addr} -> {'ON' if val else 'OFF'}")

    def _create_plc_test_row(self, line_edit: QLineEdit) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        line_edit.setMinimumHeight(34)
        line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(line_edit)

        btn_read = QPushButton("读取")
        btn_read.setMinimumSize(54, 34)
        btn_read.clicked.connect(lambda: self._test_plc_read(line_edit))

        btn_on = QPushButton("ON")
        btn_on.setMinimumSize(54, 34)
        btn_on.clicked.connect(lambda: self._test_plc_write(line_edit, True))

        btn_off = QPushButton("OFF")
        btn_off.setMinimumSize(54, 34)
        btn_off.clicked.connect(lambda: self._test_plc_write(line_edit, False))

        layout.addWidget(btn_read)
        layout.addWidget(btn_on)
        layout.addWidget(btn_off)
        return container

    # [FIX-2] 正则使用 raw string, 修复 \\d+ 无法匹配数字的 bug
    def _r_index_from_addr(self, addr: str):
        m = re.match(r"^[Rr]?(\d+)", addr.strip())
        return int(m.group(1)) if m else None

    def connect_plc(self):
        self._stop_plc_poller()  # [FIX-5] 先停旧线程
        self.plc_setting = self.build_plc_setting()
        if not self.plc_setting:
            self._append_log("PLC参数无效")
            return
        if self.plc:
            try:
                self.plc.close()
            except Exception:
                pass
        self.plc = MewtocolPLC()
        ok = self.plc.open(self.plc_setting)
        if ok:
            self._append_log("PLC连接成功 (MEWTOCOL)")
            self._start_plc_poller()  # [FIX-5] 连接成功后启动轮询线程
        else:
            self._append_log("PLC连接失败")

    def disconnect_plc(self):
        self._stop_plc_poller()  # [FIX-5]
        if self.plc:
            try:
                self.plc.close()
            except Exception:
                pass
        self.plc = None
        self._append_log("PLC已断开")

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
        self.plc_trigger_front_edit.setText("R1205")
        self.plc_trigger_side_edit.setText("R1208")
        self.plc_result_front_edit.setText("R0502")
        self.plc_result_front_trad_edit.setText("R120B")
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

        def _wrap_collapsible(title: str, content: QWidget, checked: bool = True):
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
            QGroupBox { font-weight: bold; font-size: 16px; border: 2px solid #a0c4ff; border-radius: 8px; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 8px; }
            QPushButton { background-color: #4a90e2; color: white; font-size: 16px; border: none; border-radius: 6px; padding: 8px 16px; }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2a6aaf; }
            QLineEdit { padding: 8px; border: 2px solid #bdc3c7; border-radius: 6px; }
            QSlider::groove:horizontal { height: 8px; background: #e0e0e0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #4a90e2; width: 20px; border-radius: 10px; }
            QTextEdit { border: 2px solid #bdc3c7; border-radius: 8px; }
        """)

        # 状态总览
        status_group = QGroupBox("状态总览")
        status_group.setStyleSheet("QGroupBox { font-size: 14px; }")
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(8, 4, 8, 4)
        status_layout.setSpacing(10)

        self.label_cam_status = QLabel("--")
        self.label_cam2_status = QLabel("--")
        self.label_plc_status = QLabel("--")
        for lbl in (self.label_cam_status, self.label_cam2_status, self.label_plc_status):
            lbl.setStyleSheet("color: #c0392b; font-size: 11px;")

        def _status_block(title_str, status_lbl):
            block = QHBoxLayout()
            title = QLabel(title_str)
            title.setStyleSheet("color: #555; font-size: 11px;")
            block.addWidget(title)
            block.addWidget(status_lbl)
            w = QWidget()
            w.setLayout(block)
            return w

        status_layout.addWidget(_status_block("正面相机", self.label_cam_status))
        status_layout.addWidget(_status_block("侧面相机", self.label_cam2_status))
        status_layout.addWidget(_status_block("PLC通讯", self.label_plc_status))

        # [FIX-12] 触发状态动态标签，title 保存为成员变量方便后续更新
        r1205_block = QHBoxLayout()
        self.label_front_trigger_title = QLabel("正面触发(R1205)")
        self.label_front_trigger_title.setStyleSheet("color: #555; font-size: 11px;")
        self.label_r1205_status = QLabel("OFF")
        self.label_r1205_status.setStyleSheet("color: #555; font-size: 11px; font-weight: bold;")
        r1205_block.addWidget(self.label_front_trigger_title)
        r1205_block.addWidget(self.label_r1205_status)
        r1205_widget = QWidget()
        r1205_widget.setLayout(r1205_block)

        r1208_block = QHBoxLayout()
        self.label_side_trigger_title = QLabel("侧面触发(R1208)")
        self.label_side_trigger_title.setStyleSheet("color: #555; font-size: 11px;")
        self.label_r1208_status = QLabel("OFF")
        self.label_r1208_status.setStyleSheet("color: #555; font-size: 11px; font-weight: bold;")
        r1208_block.addWidget(self.label_side_trigger_title)
        r1208_block.addWidget(self.label_r1208_status)
        r1208_widget = QWidget()
        r1208_widget.setLayout(r1208_block)

        status_layout.addWidget(r1205_widget)
        status_layout.addWidget(r1208_widget)
        status_layout.addStretch()

        metrics_layout = QHBoxLayout()
        self.label_plc_timeout = QLabel("0")
        self.label_plc_reconnect = QLabel("0")
        self.label_plc_queue = QLabel("0")
        self.label_plc_crc = QLabel("0")
        for v in (self.label_plc_timeout, self.label_plc_reconnect,
                  self.label_plc_queue, self.label_plc_crc):
            v.setStyleSheet("color: #555; font-size: 11px;")
        for caption, lbl in [("超时:", self.label_plc_timeout), ("重连:", self.label_plc_reconnect),
                              ("队列:", self.label_plc_queue), ("CRC:", self.label_plc_crc)]:
            metrics_layout.addWidget(QLabel(caption))
            metrics_layout.addWidget(lbl)
        metrics_widget = QWidget()
        metrics_widget.setLayout(metrics_layout)
        status_layout.addWidget(metrics_widget)

        status_group.setLayout(status_layout)
        display_layout.addWidget(status_group)

        # 图像显示区
        image_layout = QHBoxLayout()
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(16)

        def _image_group(title, tag_text):
            grp = QGroupBox(title)
            layout = QVBoxLayout()
            tag = QLabel(tag_text)
            tag.setAlignment(Qt.AlignLeft)
            tag.setStyleSheet("color: #666; font-size: 12px;")
            layout.addWidget(tag)
            lbl = QLabel()
            lbl.setStyleSheet("background-color: white; border: 2px solid #a0c4ff; border-radius: 8px;")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setScaledContents(True)
            lbl.setMinimumSize(360, 360)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(lbl)
            layout.setSizeConstraint(QLayout.SetDefaultConstraint)
            grp.setLayout(layout)
            grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            return grp, lbl

        front_grp, self.show_window = _image_group("AI正面检测", "正面")
        trad_grp, self.picture_box1 = _image_group("白点检测", "白点")
        side_grp, self.side_window = _image_group("AI侧面检测", "侧面")
        image_layout.addWidget(front_grp)
        image_layout.addWidget(trad_grp)
        image_layout.addWidget(side_grp)
        display_layout.addLayout(image_layout, stretch=8)

        # 结果 + 统计
        mid_layout = QHBoxLayout()
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(16)

        result_group = QGroupBox("检测结果")
        result_layout = QHBoxLayout()
        result_layout.setSpacing(24)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(12)
        right_col.setSpacing(12)

        def _result_row(label_text):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFont(QFont("Arial", 14, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedWidth(100)
            val = QLabel("--")
            val.setAlignment(Qt.AlignCenter)
            val.setFixedHeight(45)
            val.setFixedWidth(100)
            val.setFont(QFont("Arial", 18, QFont.Bold))
            val.setStyleSheet("background-color: #f0f0f0; color: #333; border: 2px solid #a0c4ff; border-radius: 8px;")
            row.addWidget(lbl)
            row.addWidget(val)
            return row, val

        ai_row, self.text_ai = _result_row("AI安装检测: ")
        trad_row, self.text_trad = _result_row("白点检测: ")
        final_row, self.text_final = _result_row("综合判定: ")
        side_row, self.text_side = _result_row("侧面检测: ")
        right_col.addLayout(ai_row)
        right_col.addLayout(trad_row)
        left_col.addLayout(final_row)
        left_col.addLayout(side_row)
        result_layout.addLayout(left_col)
        result_layout.addLayout(right_col)
        result_group.setLayout(result_layout)
        result_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mid_layout.addWidget(result_group)

        # 参数调节（左列）
        params_group = QGroupBox("参数调节")
        params_layout = QGridLayout()
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(10, 10, 10, 10)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_edit = QLineEdit("0.7")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_edit.setText(f"{v/100.0:.2f}"))
        self.conf_edit.textChanged.connect(lambda t: self.conf_slider.setValue(int(float(t) * 100) if t.replace('.', '', 1).isdigit() else 0))
        params_layout.addWidget(QLabel("AI置信度:"), 0, 0)
        params_layout.addWidget(self.conf_slider, 0, 1)
        params_layout.addWidget(self.conf_edit, 0, 2)

        self.ai_area_slider = QSlider(Qt.Horizontal)
        self.ai_area_slider.setRange(0, 10000)
        self.ai_area_edit = QLineEdit("3000")
        self.ai_area_slider.valueChanged.connect(lambda v: self.ai_area_edit.setText(str(v)))
        self.ai_area_edit.textChanged.connect(lambda t: self.ai_area_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("AI最小面积:"), 1, 0)
        params_layout.addWidget(self.ai_area_slider, 1, 1)
        params_layout.addWidget(self.ai_area_edit, 1, 2)

        self.mean_gray_min_slider = QSlider(Qt.Horizontal)
        self.mean_gray_min_slider.setRange(0, 255)
        self.mean_gray_min_edit = QLineEdit("10")
        self.mean_gray_min_slider.valueChanged.connect(lambda v: self.mean_gray_min_edit.setText(str(v)))
        self.mean_gray_min_edit.textChanged.connect(lambda t: self.mean_gray_min_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("最小灰度阈值:"), 2, 0)
        params_layout.addWidget(self.mean_gray_min_slider, 2, 1)
        params_layout.addWidget(self.mean_gray_min_edit, 2, 2)

        self.mean_gray_max_slider = QSlider(Qt.Horizontal)
        self.mean_gray_max_slider.setRange(0, 255)
        self.mean_gray_max_edit = QLineEdit("245")
        self.mean_gray_max_slider.valueChanged.connect(lambda v: self.mean_gray_max_edit.setText(str(v)))
        self.mean_gray_max_edit.textChanged.connect(lambda t: self.mean_gray_max_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("最大灰度阈值:"), 3, 0)
        params_layout.addWidget(self.mean_gray_max_slider, 3, 1)
        params_layout.addWidget(self.mean_gray_max_edit, 3, 2)

        self.tgray_slider = QSlider(Qt.Horizontal)
        self.tgray_slider.setRange(0, 255)
        self.tgray_edit = QLineEdit("254")
        self.tgray_slider.valueChanged.connect(lambda v: self.tgray_edit.setText(str(v)))
        self.tgray_edit.textChanged.connect(lambda t: self.tgray_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("白点检测阈值:"), 4, 0)
        params_layout.addWidget(self.tgray_slider, 4, 1)
        params_layout.addWidget(self.tgray_edit, 4, 2)

        self.tarea_slider = QSlider(Qt.Horizontal)
        self.tarea_slider.setRange(0, 500)
        self.tarea_edit = QLineEdit("45")
        self.tarea_slider.valueChanged.connect(lambda v: self.tarea_edit.setText(str(v)))
        self.tarea_edit.textChanged.connect(lambda t: self.tarea_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("白点检测面积:"), 5, 0)
        params_layout.addWidget(self.tarea_slider, 5, 1)
        params_layout.addWidget(self.tarea_edit, 5, 2)

        self.tgray1_slider = QSlider(Qt.Horizontal)
        self.tgray1_slider.setRange(0, 255)
        self.tgray1_edit = QLineEdit("240")
        self.tgray1_slider.valueChanged.connect(lambda v: self.tgray1_edit.setText(str(v)))
        self.tgray1_edit.textChanged.connect(lambda t: self.tgray1_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("灰尘检测阈值:"), 6, 0)
        params_layout.addWidget(self.tgray1_slider, 6, 1)
        params_layout.addWidget(self.tgray1_edit, 6, 2)

        self.tarea1_slider = QSlider(Qt.Horizontal)
        self.tarea1_slider.setRange(0, 500)
        self.tarea1_edit = QLineEdit("200")
        self.tarea1_slider.valueChanged.connect(lambda v: self.tarea1_edit.setText(str(v)))
        self.tarea1_edit.textChanged.connect(lambda t: self.tarea1_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("灰尘检测面积:"), 7, 0)
        params_layout.addWidget(self.tarea1_slider, 7, 1)
        params_layout.addWidget(self.tarea1_edit, 7, 2)

        self.side_conf_slider = QSlider(Qt.Horizontal)
        self.side_conf_slider.setRange(0, 100)
        self.side_conf_edit = QLineEdit("0.25")
        self.side_conf_slider.valueChanged.connect(lambda v: self.side_conf_edit.setText(f"{v/100.0:.2f}"))
        self.side_conf_edit.textChanged.connect(lambda t: self.side_conf_slider.setValue(int(float(t) * 100) if t.replace('.', '', 1).isdigit() else 0))
        params_layout.addWidget(QLabel("侧面AI置信度:"), 8, 0)
        params_layout.addWidget(self.side_conf_slider, 8, 1)
        params_layout.addWidget(self.side_conf_edit, 8, 2)

        self.side_area_slider = QSlider(Qt.Horizontal)
        self.side_area_slider.setRange(0, 10000)
        self.side_area_edit = QLineEdit("500")
        self.side_area_slider.valueChanged.connect(lambda v: self.side_area_edit.setText(str(v)))
        self.side_area_edit.textChanged.connect(lambda t: self.side_area_slider.setValue(int(t) if t.isdigit() else 0))
        params_layout.addWidget(QLabel("侧面最小面积:"), 9, 0)
        params_layout.addWidget(self.side_area_slider, 9, 1)
        params_layout.addWidget(self.side_area_edit, 9, 2)

        params_group.setLayout(params_layout)

        # ROI 设置
        roi_group = QGroupBox("ROI区域设置")
        roi_layout = QFormLayout()
        self.front_roi_row1_edit = QLineEdit("684")
        self.front_roi_col1_edit = QLineEdit("808")
        self.front_roi_row2_edit = QLineEdit("1076")
        self.front_roi_col2_edit = QLineEdit("1582")
        self.side_roi_row1_edit = QLineEdit("680")
        self.side_roi_col1_edit = QLineEdit("800")
        self.side_roi_row2_edit = QLineEdit("1080")
        self.side_roi_col2_edit = QLineEdit("1580")

        roi_layout.addRow("正面ROI(row1,col1,row2,col2):", QWidget())
        roi_row = QHBoxLayout()
        roi_row.addWidget(self.front_roi_row1_edit)
        roi_row.addWidget(self.front_roi_col1_edit)
        roi_row.addWidget(self.front_roi_row2_edit)
        roi_row.addWidget(self.front_roi_col2_edit)
        roi_layout.addRow("正面:", roi_row)
        roi_row2 = QHBoxLayout()
        roi_row2.addWidget(self.side_roi_row1_edit)
        roi_row2.addWidget(self.side_roi_col1_edit)
        roi_row2.addWidget(self.side_roi_row2_edit)
        roi_row2.addWidget(self.side_roi_col2_edit)
        roi_layout.addRow("侧面:", roi_row2)
        roi_group.setLayout(roi_layout)

        # ROI 变换设置
        roi_transform_group = QGroupBox("ROI图像变换")
        roi_transform_layout = QFormLayout()
        self.front_roi_rotate_edit = QLineEdit("0")
        self.front_flip_horizontal_checkbox = QCheckBox("水平镜像")
        self.front_flip_vertical_checkbox = QCheckBox("垂直镜像")
        self.side_roi_rotate_edit = QLineEdit("0")
        self.side_flip_horizontal_checkbox = QCheckBox("水平镜像")
        self.side_flip_vertical_checkbox = QCheckBox("垂直镜像")

        def _flip_box(*checkboxes):
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            for cb in checkboxes:
                lay.addWidget(cb)
            lay.addStretch()
            return w

        roi_transform_layout.addRow("正面旋转角度(°):", self.front_roi_rotate_edit)
        roi_transform_layout.addRow("正面镜像:", _flip_box(self.front_flip_horizontal_checkbox, self.front_flip_vertical_checkbox))
        roi_transform_layout.addRow("侧面旋转角度(°):", self.side_roi_rotate_edit)
        roi_transform_layout.addRow("侧面镜像:", _flip_box(self.side_flip_horizontal_checkbox, self.side_flip_vertical_checkbox))
        roi_transform_group.setLayout(roi_transform_layout)

        image_save_group = QGroupBox("检测图像保存")
        image_save_layout = QVBoxLayout()
        self.save_raw_image_checkbox = QCheckBox("保存相机原图到根目录image文件夹")
        self.save_roi_image_checkbox = QCheckBox("保存ROI截图到根目录imageROI文件夹")
        self.save_raw_image_checkbox.setChecked(False)
        self.save_roi_image_checkbox.setChecked(False)
        image_save_layout.addWidget(self.save_raw_image_checkbox)
        image_save_layout.addWidget(self.save_roi_image_checkbox)
        image_save_group.setLayout(image_save_layout)

        # 设置页面
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
        self.front_exposure_edit.setPlaceholderText("单位：us")
        self.front_gain_edit = QLineEdit()
        self.front_gain_edit.setPlaceholderText("单位：dB")
        self.btn_read_front_exposure = QPushButton("读取曝光")
        self.btn_set_front_exposure = QPushButton("设置曝光")
        self.btn_read_front_gain = QPushButton("读取增益")
        self.btn_set_front_gain = QPushButton("设置增益")

        self.btn_open_side_cam = QPushButton("打开侧面相机")
        self.btn_close_side_cam = QPushButton("关闭侧面相机")
        self.side_exposure_edit = QLineEdit()
        self.side_exposure_edit.setPlaceholderText("单位：us")
        self.side_gain_edit = QLineEdit()
        self.side_gain_edit.setPlaceholderText("单位：dB")
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

        for row, items in enumerate([
            ("正面相机", self.btn_open_front_cam, self.btn_close_front_cam, None),
            ("曝光", self.front_exposure_edit, self.btn_read_front_exposure, self.btn_set_front_exposure),
            ("增益", self.front_gain_edit, self.btn_read_front_gain, self.btn_set_front_gain),
            ("侧面相机", self.btn_open_side_cam, self.btn_close_side_cam, None),
            ("曝光", self.side_exposure_edit, self.btn_read_side_exposure, self.btn_set_side_exposure),
            ("增益", self.side_gain_edit, self.btn_read_side_gain, self.btn_set_side_gain),
        ]):
            camera_ctrl_layout.addWidget(QLabel(items[0]), row, 0)
            camera_ctrl_layout.addWidget(items[1], row, 1)
            camera_ctrl_layout.addWidget(items[2], row, 2)
            if items[3]:
                camera_ctrl_layout.addWidget(items[3], row, 3)

        camera_ctrl_group.setLayout(camera_ctrl_layout)
        camera_ctrl_collapsible = _wrap_collapsible("相机控制", camera_ctrl_group, checked=False)

        # PLC 设置
        plc_group = QGroupBox("PLC通讯设置")
        plc_group.setStyleSheet("""
            QGroupBox { font-size: 14px; }
            QLabel { font-size: 12px; }
            QLineEdit, QComboBox {
                font-size: 12px;
                min-height: 30px;
                padding: 4px 8px;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                background-color: white;
            }
            QPushButton {
                font-size: 12px;
                min-height: 30px;
                padding: 4px 8px;
            }
        """)
        plc_layout = QFormLayout()
        plc_layout.setVerticalSpacing(8)
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
        self.plc_trigger_front_edit = QLineEdit("R1205")
        self.plc_trigger_side_edit = QLineEdit("R1208")
        self.plc_result_front_edit = QLineEdit("R0502")
        self.plc_result_front_trad_edit = QLineEdit("R120B")
        self.plc_done_front_edit = QLineEdit("R0501")
        self.plc_result_side_edit = QLineEdit("R0504")
        self.plc_done_side_edit = QLineEdit("R0503")

        plc_layout.addRow("串口:", self.plc_port_combo)
        plc_layout.addRow("波特率:", self.plc_baud_combo)
        plc_layout.addRow("数据位:", self.plc_databits_combo)
        plc_layout.addRow("停止位:", self.plc_stopbits_combo)
        plc_layout.addRow("校验位:", self.plc_parity_combo)
        plc_layout.addRow("轮询(ms):", self.plc_poll_edit)

        # [FIX-3] _create_plc_test_row 现在返回 QWidget，可以安全传给 addRow
        plc_layout.addRow("正面触发拍照:", self._create_plc_test_row(self.plc_trigger_front_edit))
        plc_layout.addRow("侧面触发拍照:", self._create_plc_test_row(self.plc_trigger_side_edit))
        plc_layout.addRow("正面结果信号:", self._create_plc_test_row(self.plc_result_front_edit))
        plc_layout.addRow("正面传统结果信号:", self._create_plc_test_row(self.plc_result_front_trad_edit))
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
        p_left = QVBoxLayout()
        p_right = QVBoxLayout()
        p_left.setSpacing(12)
        p_right.setSpacing(12)
        p_left.addWidget(params_group)
        p_left.addWidget(roi_group)
        p_left.addWidget(roi_transform_group)
        p_left.addWidget(image_save_group)
        p_left.addStretch()
        p_right.addWidget(camera_ip_group)
        p_right.addLayout(camera_btn_layout)
        p_right.addWidget(camera_ctrl_collapsible)
        p_right.addWidget(plc_collapsible)
        p_right.addLayout(plc_btn_layout)
        p_right.addStretch()
        params_top_layout.addLayout(p_left, 3)
        params_top_layout.addLayout(p_right, 2)
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

        # 结果与统计
        result_stats_group = QGroupBox("结果与统计")
        result_stats_layout = QHBoxLayout()
        result_stats_layout.addLayout(mid_layout)
        result_stats_group.setLayout(result_stats_layout)
        result_stats_group.setMaximumHeight(170)
        display_layout.addWidget(result_stats_group)

        # 底部：按钮 + 日志 + 统计
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
        self.btn_toggle_log = QPushButton("显示日志")
        self.btn_toggle_log.setFixedHeight(35)
        self.btn_toggle_log.clicked.connect(self.toggle_log_view)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        self.log_text.setVisible(False)
        log_layout.addWidget(self.btn_toggle_log, alignment=Qt.AlignRight)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        bottom_layout.addWidget(log_group, stretch=3)

        # 统计
        def _stats_group(title, ok_bar_attr, ng_bar_attr, total_attr, good_attr, rate_attr, clear_fn):
            grp = QGroupBox(title)
            lay = QHBoxLayout()
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(10)
            form = QFormLayout()
            total_lbl = QLabel("0")
            good_lbl = QLabel("0")
            rate_lbl = QLabel("0.00%")
            rate_lbl.setFont(QFont("Arial", 16, QFont.Bold))
            setattr(self, total_attr, total_lbl)
            setattr(self, good_attr, good_lbl)
            setattr(self, rate_attr, rate_lbl)
            form.addRow("生产总数:", total_lbl)
            form.addRow("良品总数:", good_lbl)
            form.addRow("良率:", rate_lbl)
            lay.addLayout(form)
            ok_bar = QFrame()
            ok_bar.setStyleSheet("background-color: #2ecc71;")
            ok_bar.setFixedHeight(14)
            ng_bar = QFrame()
            ng_bar.setStyleSheet("background-color: #f1c40f;")
            ng_bar.setFixedHeight(14)
            setattr(self, ok_bar_attr, ok_bar)
            setattr(self, ng_bar_attr, ng_bar)
            bar_lay = QHBoxLayout()
            bar_lay.setSpacing(0)
            bar_lay.setContentsMargins(0, 0, 0, 0)
            bar_lay.addWidget(ok_bar)
            bar_lay.addWidget(ng_bar)
            bar_w = QWidget()
            bar_w.setLayout(bar_lay)
            bar_w.setFixedWidth(140)
            lay.addWidget(bar_w)
            btn_clear = QPushButton("清零统计")
            btn_clear.setFixedHeight(36)
            btn_clear.clicked.connect(clear_fn)
            lay.addWidget(btn_clear)
            lay.addStretch()
            grp.setLayout(lay)
            grp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            return grp

        front_stats_group = _stats_group("正面工位生产统计", "front_ok_bar", "front_ng_bar", "label_total", "label_good", "label_rate", self.clear_front_stats)
        side_stats_group = _stats_group("侧面工位生产统计", "side_ok_bar", "side_ng_bar", "side_label_total", "side_label_good", "side_label_rate", self.clear_side_stats)
        mid_layout.addWidget(front_stats_group)
        mid_layout.addWidget(side_stats_group)
        display_layout.addLayout(bottom_layout, stretch=1)

        tab_widget.addTab(page_display, "显示")
        tab_widget.addTab(params_page, "设置参数")

        # 快捷键
        for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda: self._start_detection(source="MANUAL"))

    # ── 配置 ────────────────────────────────────────────────────────────────
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
            "plc_result_front_trad": self.plc_result_front_trad_edit.text(),
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
            self._append_log("参数保存成功")
            QMessageBox.information(self, "提示", "参数保存成功")
        except Exception as e:
            self._append_log(f"保存失败：{e}")

    def load_config(self):
        if not os.path.exists(CONFIG_PATH):
            self._append_log("未找到配置文件，使用默认参数")
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
            self.plc_trigger_front_edit.setText(config.get("plc_trigger_front", "R1205"))
            self.plc_trigger_side_edit.setText(config.get("plc_trigger_side", "R1208"))
            self.plc_result_front_edit.setText(config.get("plc_result_front", "R0502"))
            self.plc_result_front_trad_edit.setText(config.get("plc_result_front_trad", "R120B"))
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
            self._append_log("参数加载成功")
        except Exception as e:
            self._append_log(f"载入失败: {e}")

    # ==================== 图像处理 ====================
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

    def apply_post_roi_transform(self, src: np.ndarray, rotate_angle: float = 0.0, flip_horizontal: bool = False, flip_vertical: bool = False) -> np.ndarray:
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
            out = cv2.warpAffine(out, rot_mat, (new_w, new_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        if flip_horizontal and flip_vertical:
            out = cv2.flip(out, -1)
        elif flip_horizontal:
            out = cv2.flip(out, 1)
        elif flip_vertical:
            out = cv2.flip(out, 0)
        return out

    def _crop_by_roi(self, src: np.ndarray, roi_values,
                     rotate_code=None, rotate_angle: float = 0.0,
                     flip_horizontal: bool = False, flip_vertical: bool = False) -> np.ndarray:
        if src is None or src.size == 0:
            return np.empty((0, 0), dtype=np.uint8)
        work_img = cv2.rotate(src, rotate_code) if rotate_code is not None else src
        if not roi_values or len(roi_values) != 4:
            return np.empty((0, 0, src.shape[2] if src.ndim == 3 else 1), dtype=src.dtype)
        row1, col1, row2, col2 = [int(v) for v in roi_values]
        h, w = work_img.shape[:2]
        row1, row2 = max(0, min(row1, h)), max(0, min(row2, h))
        col1, col2 = max(0, min(col1, w)), max(0, min(col2, w))
        if row2 <= row1 or col2 <= col1:
            return np.empty((0, 0, src.shape[2] if src.ndim == 3 else 1), dtype=src.dtype)
        roi = work_img[row1:row2, col1:col2]
        return self.apply_post_roi_transform(roi,
                                             rotate_angle=rotate_angle, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical)

    # [FIX-8] 使用哨兵值_UNSET区分"未传参"与"明确传False"
    def crop_front_roi(self, src: np.ndarray,
                       roi_values=None,
                       rotate_angle=_UNSET,
                       flip_horizontal=_UNSET,
                       flip_vertical=_UNSET) -> np.ndarray:
        transform = self.get_front_roi_transform()
        return self._crop_by_roi(
            src,
            roi_values or self.get_front_roi_values(),
            rotate_code=cv2.ROTATE_180,
            rotate_angle=transform["rotate_angle"] if rotate_angle is _UNSET else rotate_angle,
            flip_horizontal=transform["flip_horizontal"] if flip_horizontal is _UNSET else flip_horizontal,
            flip_vertical=transform["flip_vertical"] if flip_vertical is _UNSET else flip_vertical,
        )

    def crop_side_roi(self, src: np.ndarray,
                      roi_values=None,
                      rotate_angle=_UNSET,
                      flip_horizontal=_UNSET,
                      flip_vertical=_UNSET) -> np.ndarray:
        transform = self.get_side_roi_transform()
        return self._crop_by_roi(
            src,
            roi_values or self.get_side_roi_values(),
            rotate_code=None,
            rotate_angle=transform["rotate_angle"] if rotate_angle is _UNSET else rotate_angle,
            flip_horizontal=transform["flip_horizontal"] if flip_horizontal is _UNSET else flip_horizontal,
            flip_vertical=transform["flip_vertical"] if flip_vertical is _UNSET else flip_vertical,
        )

    def _save_image(self, folder_name: str, file_name: str, image: np.ndarray, station: str = ""):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = (os.path.join(root_dir, folder_name, station)
                    if station else os.path.join(root_dir, folder_name))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        if not cv2.imwrite(save_path, image):
            raise Exception(f"图像保存失败: {save_path}")
        return save_path

    def save_detection_images(self, station: str, raw_img: np.ndarray, roi_img: np.ndarray,
                               save_raw: bool, save_roi: bool):
        station = station.strip().lower()
        if station not in ("front", "side"):
            station = "front"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        raw_path = roi_path = None
        if save_raw and raw_img is not None and raw_img.size > 0:
            raw_path = self._save_image(IMAGE_DIR, f"{station}_raw_{ts}.bmp", raw_img, station)
        if save_roi and roi_img is not None and roi_img.size > 0:
            roi_path = self._save_image(ROI_IMAGE_DIR, f"{station}_roi_{ts}.bmp", roi_img, station)
        return raw_path, roi_path

    def try_save_detection_images(self, station, raw_img, roi_img, save_raw, save_roi):
        try:
            raw_path, roi_path = self.save_detection_images(station, raw_img, roi_img, save_raw, save_roi)
            return raw_path, roi_path, None
        except Exception as e:
            return None, None, str(e)

    def check_mean_gray_ng(self, image: np.ndarray):
        if image is None or image.size == 0:
            raise ValueError("待检测图像无效")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_gray = float(np.mean(gray))
        lo = self._safe_int_from_edit(self.mean_gray_min_edit, 10)
        hi = self._safe_int_from_edit(self.mean_gray_max_edit, 245)
        if lo > hi:
            lo, hi = hi, lo
        return mean_gray < lo or mean_gray > hi, mean_gray

    def yolo_detect_with_params(self, frame: np.ndarray, conf: float, ai_min_area: int):
        results = self._run_yolo(self.model, frame, conf=conf, iou=0.9, verbose=False)
        plot_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
        significant = False
        masks = results[0].masks
        if masks is not None:
            for xy in masks.xy:
                if len(xy) >= 3:
                    contour = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
                    if cv2.contourArea(contour) >= ai_min_area:
                        significant = True
                        break
        return (plot_bgr, False) if significant else (frame.copy(), True)

    def side_detect_with_params(self, frame: np.ndarray, conf: float, min_area: int):
        results = self._run_yolo(self.side_model, frame, conf=conf, iou=0.9, verbose=False)
        plot_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
        has_mask = False
        masks = results[0].masks
        if masks is not None:
            for xy in masks.xy:
                if len(xy) >= 3:
                    contour = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
                    if cv2.contourArea(contour) >= min_area:
                        has_mask = True
                        break
        return (plot_bgr, False) if has_mask else (frame.copy(), True)

    def traditional_detect_with_params(self, src: np.ndarray, tgray: int, tarea: int, tgray1: int, tarea1: int):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if len(src.shape) == 3 else src
        results = self._run_yolo(self.seg_model, src, conf=0.25, iou=0.7, verbose=False)
        region_mask = np.zeros(gray.shape, dtype=np.uint8)
        if results[0].masks is not None and len(results[0].masks.xy) > 0:
            max_area, max_poly = 0, None
            for poly in results[0].masks.xy:
                if len(poly) < 3:
                    continue
                poly_int = np.round(poly).astype(np.int32)
                area = cv2.contourArea(poly_int)
                if area > max_area:
                    max_area, max_poly = area, poly_int
            if max_poly is not None:
                cv2.fillPoly(region_mask, [max_poly], 255)
        reduced = cv2.bitwise_and(gray, region_mask)
        sell = np.zeros(src.shape[:2], dtype=np.uint8)
        for thresh, area_thresh in [(tgray, tarea), (tgray1, tarea1)]:
            _, bv = cv2.threshold(reduced, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(bv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) >= area_thresh:
                    cv2.drawContours(sell, [c], -1, 255, -1)
        contours_final, _ = cv2.findContours(sell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = src.copy()
        for c in contours_final:
            cv2.drawContours(overlay, [c], -1, (0, 0, 255), 2)
        return overlay, (len(contours_final) == 0)

    def update_result_text(self, label: QLabel, ok: bool):
        if ok:
            label.setText("OK")
            label.setStyleSheet("background-color: #00ff00; color: black; border: 2px solid #00aa00; border-radius: 8px;")
        else:
            label.setText("NG")
            label.setStyleSheet("background-color: #ff0000; color: white; border: 2px solid #aa0000; border-radius: 8px;")

    # 检测启动（正面）
    # [FIX-10] 每次启动前断开旧线程信号，避免多次回调
    def _legacy_old_start_detection(self, source: str = "MANUAL"):
        if self._plc_busy:
            self._append_log(f"{source}触发被忽略，正面检测进行中")
            return
        self._plc_busy = True
        try:
            t = self.get_front_roi_transform()
            params = {
                "conf": float(self.conf_edit.text()),
                "ai_min_area": int(float(self.ai_area_edit.text())),
                "tgray": int(float(self.tgray_edit.text())),
                "tarea": int(float(self.tarea_edit.text())),
                "tgray1": int(float(self.tgray1_edit.text())),
                "tarea1": int(float(self.tarea1_edit.text())),
                "front_roi": self.get_front_roi_values(),
                "front_rotate_angle": float(t["rotate_angle"]),
                "front_flip_horizontal": bool(t["flip_horizontal"]),
                "front_flip_vertical": bool(t["flip_vertical"]),
                "save_raw_image": self.save_raw_image_checkbox.isChecked(),
                "save_roi_image": self.save_roi_image_checkbox.isChecked(),
            }
        except Exception as e:
            self._append_log(f"正面参数读取失败: {e}")
            self._plc_busy = False
            return
        self.btn_detect.setEnabled(False)
        self._append_log(f"{source} 启动正面检测...")
        if self._detect_thread:
            try:
                self._detect_thread.result_ready.disconnect()
                self._detect_thread.error.disconnect()
            except Exception:
                pass
        self._detect_thread = DetectionThread(self, params)
        self._detect_thread.result_ready.connect(self._on_detection_result)
        self._detect_thread.error.connect(self._on_detection_error)
        self._detect_thread.finished.connect(lambda: self.btn_detect.setEnabled(True))
        self._detect_thread.start()

    # ── 统计 ─────────────────────────────────────────────────────────────────
    def _legacy_on_plc_front_bit(self, b0: bool):
        if b0 and not self._get_station_prev_trigger("front") and not self._get_station_busy("front"):
            self._start_detection(source="PLC")
        self._set_station_prev_trigger("front", b0)

    def _legacy_on_plc_side_bit(self, b1: bool):
        if b1 and not self._get_station_prev_trigger("side") and not self._get_station_busy("side"):
            self._start_side_detection(source="PLC")
        self._set_station_prev_trigger("side", b1)

    def _legacy_update_status(self):
        cam_ok = bool(self.camera and getattr(self.camera, "is_opened", False))
        cam2_ok = bool(self.camera2 and getattr(self.camera2, "is_opened", False))
        plc_ok = bool(self.plc and getattr(self.plc, "is_connected", False))
        self._set_status_label(self.label_cam_status, cam_ok)
        self._set_status_label(self.label_cam2_status, cam2_ok)
        self._set_status_label(self.label_plc_status, plc_ok)
        if hasattr(self, "label_r1205_status"):
            front_addr = self._get_plc_addr(self.plc_trigger_front_edit, "R1205")
            front_trigger = self._get_station_prev_trigger("front")
            self.label_front_trigger_title.setText(f"启动正面检测({front_addr})")
            self.label_r1205_status.setText("ON" if front_trigger else "OFF")
            self.label_r1205_status.setStyleSheet(
                f"color: {'#2ecc71' if front_trigger else '#555'}; font-size: 11px; font-weight: bold;"
            )
        if hasattr(self, "label_r1208_status"):
            side_addr = self._get_plc_addr(self.plc_trigger_side_edit, "R1208")
            side_trigger = self._get_station_prev_trigger("side")
            self.label_side_trigger_title.setText(f"启动侧面检测({side_addr})")
            self.label_r1208_status.setText("ON" if side_trigger else "OFF")
            self.label_r1208_status.setStyleSheet(
                f"color: {'#2ecc71' if side_trigger else '#555'}; font-size: 11px; font-weight: bold;"
            )
        if self.plc and getattr(self.plc, "is_connected", False):
            st = self.plc.get_status()
            self.label_plc_timeout.setText(str(st.get("timeout", 0)))
            self.label_plc_reconnect.setText(str(st.get("reconnects", 0)))
            self.label_plc_queue.setText(str(st.get("queue_size", 0)))
            self.label_plc_crc.setText(str(st.get("crc_err", 0)))
        else:
            for lbl in (self.label_plc_timeout, self.label_plc_reconnect, self.label_plc_queue, self.label_plc_crc):
                lbl.setText("0")

    def _start_detection(self, source: str = "MANUAL"):
        try:
            params = self._build_front_detection_params()
        except Exception as e:
            self._append_log(f"正面参数读取失败: {e}")
            return
        self._start_live_detection(
            "front",
            source,
            params,
            thread_factory=lambda p: DetectionThread(self, p),
            result_handler=self._on_detection_result,
            error_handler=self._on_detection_error,
        )

    def _on_detection_result(self, r: dict):
        try:
            final_ok = bool(r["final_ok"])
            yolo_img = draw_chinese_text(r["yolo_img"], "AI安装检测", (10, 10), 20, (0, 255, 0), 1)
            trad_img = draw_chinese_text(r["trad_img"], "传统算法检测", (10, 10), 20, (0, 255, 0), 1)
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_ai, bool(r["ai_ok"]))
            self.update_result_text(self.text_trad, bool(r["trad_ok"]))
            self.update_result_text(self.text_final, final_ok)
            self.update_stats(final_ok)
            self._plc_report_result(bool(r["ai_ok"]), bool(r["trad_ok"]))
            self._append_log(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 正面检测完成 | "
                f"AI: {'OK' if r['ai_ok'] else 'NG'} | 传统算法: {'OK' if r['trad_ok'] else 'NG'} | "
                f"最终结果: {'OK' if final_ok else 'NG'} | "
                f"AI耗时: {r['ai_time']:.3f}s | 传统算法耗时: {r['trad_time']:.3f}s"
            )
            if r.get("gray_ng"):
                self._append_log(f"正面灰度预检NG, 平均灰度: {r['mean_gray']:.2f}")
            self._log_saved_paths("front", r)
        finally:
            self._set_station_busy("front", False)
            self._set_task_thread("front", None, image_mode=False)

    def _on_detection_error(self, msg: str):
        try:
            self._append_log(f"正面检测错误: {msg}")
        finally:
            self._set_station_busy("front", False)
            self._set_task_thread("front", None, image_mode=False)

    def _plc_report_result(self, ai_ok: bool, trad_ok: bool):
        if not self.plc or not self.plc.is_connected:
            return
        try:
            result_addr = self._get_plc_addr(self.plc_result_front_edit, "R1207")
            result_trad_addr = self._get_plc_addr(self.plc_result_front_trad_edit, "R120B")
            done_addr = self._get_plc_addr(self.plc_done_front_edit, "R1206")
            self.plc.write_bit(result_addr, not ai_ok)
            self.plc.write_bit(result_trad_addr, not trad_ok)
            self.plc.write_bit(done_addr, True)
        except Exception as e:
            self._append_log(f"PLC回写异常(正面): {e}")

    def _plc_report_side_result(self, side_ok: bool):
        if not self.plc or not self.plc.is_connected:
            return
        try:
            result_addr = self._get_plc_addr(self.plc_result_side_edit, "R120A")
            done_addr = self._get_plc_addr(self.plc_done_side_edit, "R1209")
            self.plc.write_bit(result_addr, not side_ok)
            self.plc.write_bit(done_addr, True)
        except Exception as e:
            self._append_log(f"PLC回写异常(侧面): {e}")

    def _start_side_detection(self, source: str = "MANUAL"):
        try:
            params = self._build_side_detection_params()
        except Exception as e:
            self._append_log(f"侧面参数读取失败: {e}")
            return
        self._start_live_detection(
            "side",
            source,
            params,
            thread_factory=lambda p: SideDetectionThread(self, p),
            result_handler=self._on_side_detection_result,
            error_handler=self._on_side_detection_error,
        )

    def _on_side_detection_result(self, r: dict):
        try:
            side_ok = bool(r["side_ok"])
            side_img = draw_chinese_text(r["side_img"], "AI侧面检测", (10, 10), 20, (0, 255, 0), 1)
            self.side_window.setPixmap(cv2_to_qpixmap(side_img))
            self.update_result_text(self.text_side, side_ok)
            self.update_side_stats(side_ok)
            self._plc_report_side_result(side_ok)
            self._append_log(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 侧面检测完成 | "
                f"最终结果: {'OK' if side_ok else 'NG'} | AI耗时: {r['ai_time']:.3f}s"
            )
            if r.get("gray_ng"):
                self._append_log(f"侧面灰度预检NG, 平均灰度: {r['mean_gray']:.2f}")
            self._log_saved_paths("side", r)
        finally:
            self._set_station_busy("side", False)
            self._set_task_thread("side", None, image_mode=False)

    def _on_side_detection_error(self, msg: str):
        try:
            self._append_log(f"侧面检测错误: {msg}")
        finally:
            self._set_station_busy("side", False)
            self._set_task_thread("side", None, image_mode=False)

    def on_image_detect(self):
        try:
            self._start_image_detection_task(
                "front",
                "选择图片进行正面检测",
                busy_guard=False,
                thread_factory=lambda file_path, params: ImageDetectionThread(self, file_path, params),
                result_handler=self._on_image_detect_result,
                error_handler=self._on_image_detect_error,
            )
        except Exception as e:
            self._append_log(f"图片检测参数错误: {e}")

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
                self._append_log(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片检测完成 | 文件: {file_name} | "
                    f"最终结果: NG | 灰度预检均值: {r['mean_gray']:.2f}"
                )
                return
            yolo_img = draw_chinese_text(r["yolo_img"], "AI图片检测", (10, 10), 20, (0, 255, 0), 1)
            trad_img = draw_chinese_text(r["trad_img"], "传统算法检测", (10, 10), 20, (0, 255, 0), 1)
            self.show_window.setPixmap(cv2_to_qpixmap(yolo_img))
            self.picture_box1.setPixmap(cv2_to_qpixmap(trad_img))
            self.update_result_text(self.text_ai, bool(r["ai_ok"]))
            self.update_result_text(self.text_trad, bool(r["trad_ok"]))
            self.update_result_text(self.text_final, bool(r["final_ok"]))
            self.update_stats(bool(r["final_ok"]))
            self._append_log(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片检测完成 | 文件: {file_name} | "
                f"AI: {'OK' if r['ai_ok'] else 'NG'} | 传统算法: {'OK' if r['trad_ok'] else 'NG'} | "
                f"最终结果: {'OK' if r['final_ok'] else 'NG'} | "
                f"AI耗时: {r['ai_time']:.3f}s | 传统算法耗时: {r['trad_time']:.3f}s"
            )
        finally:
            self._set_task_thread("front", None, image_mode=True)

    def _on_image_detect_error(self, msg: str):
        try:
            self._append_log(f"图片检测失败: {msg}")
        finally:
            self._set_task_thread("front", None, image_mode=True)

    def on_image_side_detect(self):
        try:
            self._start_image_detection_task(
                "side",
                "选择图片进行侧面检测",
                busy_guard=True,
                thread_factory=lambda file_path, params: ImageSideDetectionThread(
                    self,
                    file_path,
                    side_conf=params["side_conf"],
                    side_min_area=params["side_min_area"],
                ),
                result_handler=self._on_image_side_detect_result,
                error_handler=self._on_image_side_detect_error,
            )
        except Exception as e:
            self._append_log(f"图片侧面检测参数错误: {e}")

    def _on_image_side_detect_result(self, r: dict):
        try:
            file_name = os.path.basename(r.get("file_path", ""))
            if r.get("gray_ng"):
                self.side_window.setPixmap(cv2_to_qpixmap(r["side_img"]))
                self.update_result_text(self.text_side, False)
                self.update_side_stats(False)
                self._plc_report_side_result(False)
                self._append_log(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片侧面检测完成 | 文件: {file_name} | "
                    f"最终结果: NG | 灰度预检均值: {r['mean_gray']:.2f}"
                )
                return
            side_ok = bool(r["side_ok"])
            side_img = draw_chinese_text(r["side_img"], "AI侧面检测", (10, 10), 20, (0, 255, 0), 1)
            self.side_window.setPixmap(cv2_to_qpixmap(side_img))
            self.update_result_text(self.text_side, side_ok)
            self.update_side_stats(side_ok)
            self._plc_report_side_result(side_ok)
            self._append_log(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 图片侧面检测完成 | 文件: {file_name} | "
                f"最终结果: {'OK' if side_ok else 'NG'} | AI耗时: {r['ai_time']:.3f}s"
            )
        finally:
            self._set_station_busy("side", False)
            self._set_task_thread("side", None, image_mode=True)

    def _on_image_side_detect_error(self, msg: str):
        try:
            self._append_log(f"图片侧面检测失败: {msg}")
        finally:
            self._set_station_busy("side", False)
            self._set_task_thread("side", None, image_mode=True)

    def _update_rate_bar(self, ok_count: int, total_count: int, ok_bar: QFrame, ng_bar: QFrame):
        if total_count <= 0:
            ok_bar.setFixedWidth(self._rate_bar_width)
            ng_bar.setFixedWidth(0)
        else:
            ok_w = int(self._rate_bar_width * ok_count / total_count)
            ok_bar.setFixedWidth(ok_w)
            ng_bar.setFixedWidth(max(self._rate_bar_width - ok_w, 0))

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

    def clear_front_stats(self):
        self.total_count = 0
        self.good_count = 0
        self.label_total.setText("0")
        self.label_good.setText("0")
        self.label_rate.setText("0.00%")
        self._update_rate_bar(self.good_count, self.total_count, self.front_ok_bar, self.front_ng_bar)
        self._append_log("正面工位统计已清零")

    def clear_side_stats(self):
        self.side_total_count = 0
        self.side_good_count = 0
        self.side_label_total.setText("0")
        self.side_label_good.setText("0")
        self.side_label_rate.setText("0.00%")
        self._update_rate_bar(self.side_good_count, self.side_total_count, self.side_ok_bar, self.side_ng_bar)
        self._append_log("侧面工位统计已清零")

    # ── 关闭 ──
    def closeEvent(self, event):
        self._stop_plc_poller()  # [FIX-5] 先停轮询线程
        self.status_timer.stop()
        for thread_attr in ("_detect_thread", "_side_detect_thread", "_image_detect_thread", "_image_side_detect_thread", "_warmup_thread"):
            t = getattr(self, thread_attr, None)
            if t and t.isRunning():
                t.requestInterruption()
                t.wait(1500)
        for cam in (self.camera, self.camera2):
            if cam:
                try:
                    cam.close()
                except Exception:
                    pass
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
