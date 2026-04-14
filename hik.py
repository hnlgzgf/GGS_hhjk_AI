# -*- coding: utf-8 -*-
"""
海康威视 MVS SDK + PyQt5 单文件 Demo
功能：
- 枚举设备
- 打开/关闭相机
- 连续采集 / 软触发模式切换
- 实时显示彩色图像 (RGB8/BGR8/Mono8 自动转换)
- 显示曝光、增益、帧率
- 保存当前帧为 bmp
- 软触发按钮（仅在触发模式下有效）
"""

import sys
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='hik_debug.log',
    filemode='w'
)

# 添加 MvImport 到搜索路径（放在最前面！）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MvImport'))
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QGroupBox, QRadioButton,
    QMessageBox, QLineEdit, QFormLayout, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# 海康 SDK 导入
try:
    from MvCameraControl_class import *
    from CameraParams_header import *
    from MvErrorDefine_const import *
except ImportError:
    print("缺少海康SDK文件，请将 MvImport 目录下的文件复制到当前目录")
    sys.exit(1)


class CameraThread(QThread):
    """采集线程"""
    new_image = pyqtSignal(np.ndarray)     # BGR图像
    status_msg = pyqtSignal(str)
    error_msg  = pyqtSignal(str)

    def __init__(self, cam: MvCamera):
        super().__init__()
        self.cam = cam
        self.running = False
        self.grabbing = False
        self.trigger_mode = False
        self.trigger_event = False
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def run(self):
        self.running = True
        self.status_msg.emit("采集线程启动")
        logging.info("采集线程启动")

        while self.running:
            if not self.grabbing:
                time.sleep(0.02)
                continue

            if self.trigger_mode and not self.trigger_event:
                time.sleep(0.01)
                continue

            if self.trigger_mode:
                self.trigger_event = False

            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            # 尝试不同的缓冲区大小
            buffer_size = 50 * 1024 * 1024  # 50MB
            logging.debug(f"创建缓冲区，大小: {buffer_size / 1024 / 1024:.1f} MB")
            data_buf = (c_ubyte * buffer_size)()

            logging.debug("开始获取图像")
            ret = self.cam.MV_CC_GetOneFrameTimeout(
                data_buf, len(data_buf),
                stFrameInfo, 1000
            )
            logging.debug(f"获取图像返回值: 0x{ret:X}")

            if ret == MV_OK:
                w = stFrameInfo.nWidth
                h = stFrameInfo.nHeight
                pixel_type = stFrameInfo.enPixelType
                logging.info(f"获取图像成功：宽度={w}, 高度={h}, 像素格式=0x{pixel_type:X}, 帧号={stFrameInfo.nFrameNum}")

                # 彩色优先处理
                if pixel_type == PixelType_Gvsp_RGB8_Packed:
                    img = np.frombuffer(data_buf, dtype=np.uint8, count=w*h*3)
                    img = img.reshape((h, w, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    logging.debug("处理RGB8格式图像")
                elif pixel_type == PixelType_Gvsp_BGR8_Packed:
                    img = np.frombuffer(data_buf, dtype=np.uint8, count=w*h*3)
                    img = img.reshape((h, w, 3))
                    logging.debug("处理BGR8格式图像")
                elif pixel_type == PixelType_Gvsp_Mono8:
                    img = np.frombuffer(data_buf, dtype=np.uint8, count=w*h)
                    img = img.reshape((h, w))
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    logging.debug("处理Mono8格式图像")
                else:
                    error_msg = f"不支持的像素格式: 0x{pixel_type:X}"
                    logging.warning(error_msg)
                    self.error_msg.emit(error_msg)
                    continue

                self.new_image.emit(img)
                # 计算帧率
                current_time = time.time()
                self.frame_count += 1
                elapsed = current_time - self.last_frame_time
                if elapsed > 1.0:  # 每秒更新一次
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.last_frame_time = current_time
                # 显示状态信息，包含帧率
                status_text = f"帧 {stFrameInfo.nFrameNum} | {w}x{h} | {self.fps:.1f} fps"
                self.status_msg.emit(status_text)
                logging.debug(status_text)

            else:
                # 跳过常见非错误：超时(0x80000002) / 无数据(0x80000003)
                if ret in (0x80000002, 0x80000003):
                    logging.debug(f"获取图像超时或无数据: 0x{ret:X}")
                    continue
                # 在触发模式下，跳过更多错误（如未触发）
                if self.trigger_mode:
                    logging.debug(f"触发模式下获取图像失败: 0x{ret:X}")
                    continue
                # 其他错误才报告
                error_msg = f"取图失败 0x{ret:X}"
                logging.error(error_msg)
                self.error_msg.emit(error_msg)

            time.sleep(0.001)  # 防止空转过高CPU

        self.status_msg.emit("采集线程退出")
        logging.info("采集线程退出")

    def start_grab(self):
        self.grabbing = True

    def stop_grab(self):
        self.grabbing = False

    def trigger_once(self):
        if self.trigger_mode:
            self.trigger_event = True

    def execute_software_trigger(self):
        if not self.grabbing or not self.trigger_mode:
            return False
        self.trigger_event = True
        ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
        return ret == MV_OK

    def stop(self):
        self.running = False
        self.wait()


class HikCameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("海康相机采集 - 软触发彩图版")
        self.resize(1280, 720)

        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.thread = None
        self.is_open = False
        self.is_grabbing = False

        self.init_ui()
        logging.debug("开始初始化SDK")
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            logging.error(f"SDK初始化失败 0x{ret:X}")
            QMessageBox.critical(self, "致命错误", f"SDK初始化失败 0x{ret:X}")
            sys.exit(-1)
        logging.info("SDK初始化成功")

    def closeEvent(self, event):
        self.close_device()
        MvCamera.MV_CC_Finalize()
        super().closeEvent(event)

    def resizeEvent(self, event):
        # 窗口大小改变时，重新调整图片大小
        self._scale_pixmap()
        super().resizeEvent(event)

    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main_h = QHBoxLayout(cw)

        left = QWidget()
        vl = QVBoxLayout(left)
        self.lbl_image = QLabel()
        self.lbl_image.setMinimumSize(640, 480)
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background:black;")
        vl.addWidget(self.lbl_image)

        self.lbl_status = QLabel("状态：未连接")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        vl.addWidget(self.lbl_status)

        main_h.addWidget(left, 7)

        right = QWidget()
        vr = QVBoxLayout(right)

        gb_dev = QGroupBox("设备")
        hdev = QHBoxLayout(gb_dev)
        self.cb_dev = QComboBox()
        btn_enum = QPushButton("枚举")
        btn_enum.clicked.connect(self.enum_devices)
        hdev.addWidget(self.cb_dev)
        hdev.addWidget(btn_enum)
        vr.addWidget(gb_dev)

        gb_ctrl = QGroupBox("采集控制")
        hctrl = QHBoxLayout(gb_ctrl)
        self.btn_open    = QPushButton("打开")
        self.btn_close   = QPushButton("关闭")
        self.btn_start   = QPushButton("开始采集")
        self.btn_stop    = QPushButton("停止采集")
        self.btn_trigger = QPushButton("软触发一次")

        self.btn_open.clicked.connect(self.open_device)
        self.btn_close.clicked.connect(self.close_device)
        self.btn_start.clicked.connect(self.start_grabbing)
        self.btn_stop.clicked.connect(self.stop_grabbing)
        self.btn_trigger.clicked.connect(self.software_trigger)

        hctrl.addWidget(self.btn_open)
        hctrl.addWidget(self.btn_close)
        hctrl.addWidget(self.btn_start)
        hctrl.addWidget(self.btn_stop)
        hctrl.addWidget(self.btn_trigger)
        vr.addWidget(gb_ctrl)

        gb_trigger = QGroupBox("触发模式")
        htrig = QHBoxLayout(gb_trigger)
        self.rb_continue = QRadioButton("连续模式")
        self.rb_trigger  = QRadioButton("软触发模式")
        self.rb_continue.setChecked(True)
        self.rb_continue.toggled.connect(self.on_trigger_mode_changed)
        self.rb_trigger.toggled.connect(self.on_trigger_mode_changed)
        htrig.addWidget(self.rb_continue)
        htrig.addWidget(self.rb_trigger)
        vr.addWidget(gb_trigger)

        gb_param = QGroupBox("相机参数")
        form = QFormLayout(gb_param)
        self.le_exp  = QLineEdit("---")
        self.le_gain = QLineEdit("---")
        self.le_fps  = QLineEdit("---")
        form.addRow("曝光时间 (μs):", self.le_exp)
        form.addRow("增益:",          self.le_gain)
        form.addRow("帧率 (fps):",    self.le_fps)
        
        h_param = QHBoxLayout()
        self.btn_get_param = QPushButton("获取参数")
        self.btn_set_param = QPushButton("设置参数")
        self.btn_get_param.clicked.connect(self.get_camera_params)
        self.btn_set_param.clicked.connect(self.set_camera_params)
        h_param.addWidget(self.btn_get_param)
        h_param.addWidget(self.btn_set_param)
        form.addRow("", h_param)
        
        vr.addWidget(gb_param)

        self.btn_save = QPushButton("保存当前帧为BMP")
        self.btn_save.clicked.connect(self.save_current_frame)
        vr.addWidget(self.btn_save)

        main_h.addWidget(right, 3)

        self.update_ui_state()

    def update_ui_state(self):
        self.btn_open.setEnabled(not self.is_open)
        self.btn_close.setEnabled(self.is_open)
        self.btn_start.setEnabled(self.is_open and not self.is_grabbing)
        self.btn_stop.setEnabled(self.is_open and self.is_grabbing)
        self.btn_trigger.setEnabled(self.is_open and self.is_grabbing and self.rb_trigger.isChecked())
        self.btn_save.setEnabled(self.is_open and self.is_grabbing)
        self.btn_get_param.setEnabled(self.is_open)
        self.btn_set_param.setEnabled(self.is_open)

    def enum_devices(self):
        logging.debug("开始枚举设备")
        self.cb_dev.clear()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, self.device_list)
        if ret != MV_OK:
            logging.error(f"枚举设备失败 0x{ret:X}")
            QMessageBox.warning(self, "提示", "未找到设备")
            return
        if self.device_list.nDeviceNum == 0:
            logging.info("未找到设备")
            QMessageBox.warning(self, "提示", "未找到设备")
            return
        
        logging.info(f"找到 {self.device_list.nDeviceNum} 个设备")
        for i in range(self.device_list.nDeviceNum):
            dev = cast(self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            
            if dev.nTLayerType == MV_GIGE_DEVICE:
                user_name = bytes(dev.SpecialInfo.stGigEInfo.chUserDefinedName).decode('gbk', errors='ignore').rstrip('\x00')
                model_name = bytes(dev.SpecialInfo.stGigEInfo.chModelName).decode('gbk', errors='ignore').rstrip('\x00')
                ip = f"{(dev.SpecialInfo.stGigEInfo.nCurrentIp>>24)&0xff}.{(dev.SpecialInfo.stGigEInfo.nCurrentIp>>16)&0xff}.{(dev.SpecialInfo.stGigEInfo.nCurrentIp>>8)&0xff}.{dev.SpecialInfo.stGigEInfo.nCurrentIp&0xff}"
                text = f"[{i}] GigE {user_name} {model_name} ({ip})"
                logging.info(f"设备 {i}: {text}")
            else:
                user_name = bytes(dev.SpecialInfo.stUsb3VInfo.chUserDefinedName).decode('gbk', errors='ignore').rstrip('\x00')
                model_name = bytes(dev.SpecialInfo.stUsb3VInfo.chModelName).decode('gbk', errors='ignore').rstrip('\x00')
                sn = ''.join(chr(c) for c in dev.SpecialInfo.stUsb3VInfo.chSerialNumber if c != 0)
                text = f"[{i}] USB {user_name} {model_name} ({sn})"
                logging.info(f"设备 {i}: {text}")
            
            self.cb_dev.addItem(text)

    def open_device(self):
        idx = self.cb_dev.currentIndex()
        if idx < 0:
            logging.warning("未选择设备")
            return

        dev = cast(self.device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
        logging.debug(f"开始打开设备 {idx}")

        ret = self.cam.MV_CC_CreateHandle(dev)
        if ret != MV_OK:
            logging.error(f"创建句柄失败 0x{ret:X}")
            QMessageBox.critical(self, "错误", f"创建句柄失败 0x{ret:X}")
            return
        logging.info("创建句柄成功")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            logging.error(f"打开设备失败 0x{ret:X}")
            self.cam.MV_CC_DestroyHandle()
            QMessageBox.critical(self, "错误", f"打开设备失败 0x{ret:X}")
            return
        logging.info("打开设备成功")

        # 默认连续模式
        logging.debug("设置触发模式为关闭")
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != MV_OK:
            logging.warning(f"设置触发模式失败 0x{ret:X}")
        else:
            logging.info("设置触发模式成功")
        
        # 尝试设置像素格式，但不强制要求
        logging.debug("尝试设置像素格式为RGB8")
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
        if ret != MV_OK:
            logging.warning(f"设置RGB8格式失败 0x{ret:X}")
            # 尝试使用BGR8格式
            logging.debug("尝试设置像素格式为BGR8")
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BGR8_Packed)
            if ret != MV_OK:
                logging.warning(f"设置BGR8格式失败 0x{ret:X}")
                logging.info("使用相机默认像素格式")
            else:
                logging.info("设置像素格式为BGR8成功")
        else:
            logging.info("设置像素格式为RGB8成功")
        
        # 获取当前触发模式和像素格式
        stEnum = MVCC_ENUMVALUE()
        ret = self.cam.MV_CC_GetEnumValue("TriggerMode", stEnum)
        if ret == MV_OK:
            logging.info(f"当前触发模式: {stEnum.nCurValue}")
        ret = self.cam.MV_CC_GetEnumValue("PixelFormat", stEnum)
        if ret == MV_OK:
            logging.info(f"当前像素格式: 0x{stEnum.nCurValue:X}")

        # 启动采集线程
        self.thread = CameraThread(self.cam)
        self.thread.new_image.connect(self.show_image)
        self.thread.status_msg.connect(lambda s: self.lbl_status.setText(f"状态：{s}"))
        self.thread.error_msg.connect(lambda e: QMessageBox.warning(self, "采集错误", e))
        self.thread.start()
        logging.info("采集线程启动成功")

        self.is_open = True
        self.update_ui_state()
        self.get_params_once()
        logging.info("设备初始化完成")

    def close_device(self):
        if not self.is_open:
            return

        self.stop_grabbing()

        if self.thread:
            self.thread.stop()

        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        self.is_open = False
        self.is_grabbing = False
        self.update_ui_state()
        self.lbl_image.clear()
        self.lbl_status.setText("状态：已关闭")

    def start_grabbing(self):
        if not self.is_open:
            logging.warning("相机未打开，无法开始采集")
            QMessageBox.warning(self, "警告", "相机未打开，请先打开相机")
            return
        
        logging.debug("开始采集")
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            logging.error(f"开始采集失败 0x{ret:X}")
            QMessageBox.critical(self, "错误", f"开始采集失败 0x{ret:X}")
            return
        logging.info("开始采集成功")
        
        is_trigger = self.rb_trigger.isChecked()
        
        if self.thread:
            self.thread.trigger_mode = is_trigger
            self.thread.trigger_event = False
            self.thread.start_grab()
            logging.debug("采集线程开始抓取")
        
        self.is_grabbing = True
        self.update_ui_state()
        
        if is_trigger:
            self.lbl_status.setText("状态：等待软触发...")
        else:
            self.lbl_status.setText("状态：连续采集")

    def stop_grabbing(self):
        logging.debug("停止采集")
        if self.thread:
            self.thread.stop_grab()
            logging.debug("采集线程停止抓取")
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != MV_OK:
            logging.warning(f"停止采集失败 0x{ret:X}")
        else:
            logging.info("停止采集成功")
        
        is_trigger = self.rb_trigger.isChecked()
        if is_trigger:
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
            if ret != MV_OK:
                logging.warning(f"恢复触发模式失败 0x{ret:X}")
            ret = self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
            if ret != MV_OK:
                logging.warning(f"设置软触发源失败 0x{ret:X}")
        
        self.is_grabbing = False
        self.update_ui_state()
        self.lbl_status.setText("状态：停止采集")

    def on_trigger_mode_changed(self, checked):
        if not self.is_open:
            return

        is_trigger = self.rb_trigger.isChecked()

        was_grabbing = self.is_grabbing
        if was_grabbing:
            self.stop_grabbing()

        mode = MV_TRIGGER_MODE_ON if is_trigger else MV_TRIGGER_MODE_OFF
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", mode)
        if ret != MV_OK:
            QMessageBox.warning(self, "警告", f"设置触发模式失败 0x{ret:X}\n已恢复连续模式")
            self.rb_continue.setChecked(True)
            is_trigger = False

        if is_trigger:
            self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)

        if was_grabbing:
            self.start_grabbing()

        if self.thread:
            self.thread.trigger_mode = is_trigger

        self.update_ui_state()

    def software_trigger(self):
        if not self.thread or not self.thread.execute_software_trigger():
            QMessageBox.warning(self, "提示", "软触发失败（请确认已在触发模式 + 正在采集）")

    def get_params_once(self):
        self.get_camera_params()

    def get_camera_params(self):
        if not self.is_open:
            QMessageBox.warning(self, "警告", "请先打开相机")
            return
        
        stFloat = MVCC_FLOATVALUE()
        
        # 获取曝光时间
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloat)
        if ret == MV_OK:
            self.le_exp.setText(f"{stFloat.fCurValue:.1f}")
            logging.info(f"获取曝光时间成功: {stFloat.fCurValue:.1f}")
        else:
            logging.warning(f"获取曝光时间失败: 0x{ret:X}")
        
        # 获取增益
        ret = self.cam.MV_CC_GetFloatValue("Gain", stFloat)
        if ret == MV_OK:
            self.le_gain.setText(f"{stFloat.fCurValue:.1f}")
            logging.info(f"获取增益成功: {stFloat.fCurValue:.1f}")
        else:
            logging.warning(f"获取增益失败: 0x{ret:X}")
        
        # 获取帧率
        ret = self.cam.MV_CC_GetFloatValue("ResultingFrameRate", stFloat)
        if ret == MV_OK:
            self.le_fps.setText(f"{stFloat.fCurValue:.1f}")
        else:
            self.le_fps.setText("---")

    def set_camera_params(self):
        if not self.is_open:
            QMessageBox.warning(self, "警告", "请先打开相机")
            return
        
        try:
            exp_value = float(self.le_exp.text())
            if exp_value <= 0:
                QMessageBox.warning(self, "警告", "曝光时间必须大于0")
                return
            
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exp_value)
            if ret == MV_OK:
                logging.info(f"设置曝光时间成功: {exp_value}")
            else:
                logging.warning(f"设置曝光时间失败: 0x{ret:X}")
                QMessageBox.warning(self, "警告", f"设置曝光时间失败: 0x{ret:X}")
                return
        except ValueError:
            QMessageBox.warning(self, "警告", "曝光时间必须是数字")
            return
        
        try:
            gain_value = float(self.le_gain.text())
            if gain_value < 0:
                QMessageBox.warning(self, "警告", "增益必须大于等于0")
                return
            
            ret = self.cam.MV_CC_SetFloatValue("Gain", gain_value)
            if ret == MV_OK:
                logging.info(f"设置增益成功: {gain_value}")
            else:
                logging.warning(f"设置增益失败: 0x{ret:X}")
                QMessageBox.warning(self, "警告", f"设置增益失败: 0x{ret:X}")
                return
        except ValueError:
            QMessageBox.warning(self, "警告", "增益必须是数字")
            return
        
        QMessageBox.information(self, "成功", "参数设置成功")

    def show_image(self, img: np.ndarray):
        if img is None or img.size == 0:
            return
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w*3, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qimg)
        
        # 保存原始pixmap，用于后续缩放
        self._current_pixmap = pix
        
        # 立即显示缩放后的图片
        self._scale_pixmap()

    def _scale_pixmap(self):
        if not hasattr(self, '_current_pixmap') or self._current_pixmap is None:
            return
        
        # 确保标签能够根据窗口大小自动调整
        self.lbl_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_image.setScaledContents(False)
        
        # 缩放图片以适应标签大小，忽略宽高比（类似 Halcon 的 set_part(0,0,-1,-1)）
        target_size = self.lbl_image.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            QTimer.singleShot(50, self._scale_pixmap)
            return
        
        scaled = self._current_pixmap.scaled(target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.lbl_image.setPixmap(scaled)

    def save_current_frame(self):
        pix = self.lbl_image.pixmap()
        if pix and not pix.isNull():
            filename = f"capture_{int(time.time())}.bmp"
            pix.save(filename)
            QMessageBox.information(self, "保存成功", f"已保存：{filename}")
        else:
            QMessageBox.warning(self, "提示", "当前无图像可保存")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = HikCameraWindow()
    win.show()
    sys.exit(app.exec_())