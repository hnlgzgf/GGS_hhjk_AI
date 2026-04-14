# -*- coding: utf-8 -*-
"""
Fatek / FP series PLC 通信驱动（工业级优化版）
基于你提供的 C# 代码风格 + FATEK 协议特征
支持自动重连、超时重发、线程安全、日志
"""

import serial
import serial.tools.list_ports
import threading
import queue
import time
import logging
from datetime import datetime
from typing import Optional, List, Union
import re


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为 DEBUG 以查看原始字节
    format='%(asctime)s | %(levelname)-5s | %(threadName)-12s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FpPlc:
    def __init__(self):
        self.serial_port: Optional[serial.Serial] = None
        self.port_settings: dict = {}
        self.is_running = False
        self.is_connected = False

        # 通信队列与标志
        self.write_queue = queue.Queue(maxsize=200)     # 写指令队列
        self.send_lock = threading.Lock()
        self.send_flag = False
        self.current_cmd: str = ""
        self.current_read_addr: int = 0
        self.last_send_time = datetime.now()

        # 数据区（支持较大的寄存器范围）
        self.R = [False] * 8192                         # 扩展 R 位范围
        self.databuffer = [0] * 64                      # 增加缓冲区大小

        # 统计与错误处理
        self.send_count = 0
        self.recv_ok_count = 0
        self.crc_error_count = 0
        self.timeout_count = 0
        self.reconnect_count = 0
        self.resend_count = 0

        # 配置参数
        self.timeout_ms = 1000                          # 延长超时 ms
        self.max_resend = 4                             # 最大重发次数
        self.poll_interval_ms = 30                      # 轮询间隔 (30ms)
        self.reconnect_interval_sec = 5.0               # 重连间隔

        # 线程
        self._recv_thread: Optional[threading.Thread] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None

    def open(self, setting_str: str) -> bool:
        """
        打开串口，格式示例：
        "COM3,38400,8,1,0,10"   # 最后一位是 poll_interval_ms
        parity: 0=None, 1=Odd, 2=Even
        """
        try:
            parts = [p.strip() for p in setting_str.split(",")]
            if len(parts) < 6:
                raise ValueError("参数格式错误，至少6项")

            port = parts[0]
            baud = int(parts[1])
            databits = int(parts[2])
            stopbits = serial.STOPBITS_ONE if int(parts[3]) == 1 else serial.STOPBITS_TWO
            parity_map = {0: serial.PARITY_NONE, 1: serial.PARITY_ODD, 2: serial.PARITY_EVEN}
            parity = parity_map.get(int(parts[4]), serial.PARITY_NONE)
            self.poll_interval_ms = int(parts[5])

            self.port_settings = {
                "port": port,
                "baudrate": baud,
                "bytesize": databits,
                "parity": parity,
                "stopbits": stopbits,
                "timeout": 0.3
            }

            self._connect_serial()
            if not self.is_connected:
                return False

            self.is_running = True
            self._start_threads()
            logger.info(f"PLC 通信已启动：{port} @ {baud}")
            return True

        except Exception as e:
            logger.error(f"打开串口失败: {e}")
            return False

    def _connect_serial(self):
        try:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()

            self.serial_port = serial.Serial(**self.port_settings)
            self.is_connected = True
            self.reconnect_count = 0
            logger.info(f"串口连接成功：{self.port_settings['port']}")
        except serial.SerialException as e:
            self.is_connected = False
            logger.warning(f"串口连接失败: {e}")

    def _start_threads(self):
        self._recv_thread = threading.Thread(target=self._recv_loop, name="RecvLoop", daemon=True)
        self._recv_thread.start()

        self._poll_thread = threading.Thread(target=self._poll_loop, name="PollLoop", daemon=True)
        self._poll_thread.start()

        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, name="Reconnect", daemon=True)
        self._reconnect_thread.start()

    def close(self):
        self.is_running = False
        time.sleep(0.1)  # 等待线程退出循环

        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
            self.serial_port = None

        self.is_connected = False
        logger.info("PLC 通信已关闭")

    def _reconnect_loop(self):
        while self.is_running:
            if not self.is_connected and self.is_running:
                logger.warning("检测到断开 → 尝试重连...")
                self._connect_serial()
                self.reconnect_count += 1
            time.sleep(self.reconnect_interval_sec)

    def _recv_loop(self):
        buffer = ""
        while self.is_running:
            if not self.is_connected or not self.serial_port:
                time.sleep(0.3)
                continue

            try:
                if self.serial_port.in_waiting > 0:
                    raw_data = self.serial_port.read(self.serial_port.in_waiting)
                    # 记录接收到的原始 16 进制，便于排查乱码/极性问题
                    logger.debug(f"串口原始接收: {raw_data.hex(' ')}")
                    data = raw_data.decode('ascii', errors='ignore')
                    buffer += data

                    while '\r' in buffer:
                        pos = buffer.index('\r')
                        frame = buffer[:pos + 1]
                        buffer = buffer[pos + 1:]
                        self._process_frame(frame.strip())

                time.sleep(0.005)  # 避免 CPU 100%
            except Exception as e:
                logger.error(f"接收线程异常: {e}")
                self.is_connected = False

    def _process_frame(self, frame: str):
        # 接收到的 frame 是经过 strip() 的，不含 \r
        if len(frame) < 8:
            return

        try:
            # 支持 <EE# 或 <EE$ (C# 代码格式) 或 %01# 或 %01$ (松下标准格式)
            if not (frame.startswith("<EE") or frame.startswith("%") or frame.startswith("$")):
                return

            # 计算校验位：除了最后两位（BCC/CRC）以外的所有字符进行异或
            # C# 逻辑是 XOR 从 0 到 Length-3
            calc_bcc = self._calc_crc(frame[:-2])
            recv_bcc = frame[-2:]

            if calc_bcc != recv_bcc:
                self.crc_error_count += 1
                logger.warning(f"BCC 错误: 计算={calc_bcc} 收到={recv_bcc} 帧={frame}")
                self._handle_send_failed()
                return

            self.recv_ok_count += 1

            # 解析数据响应
            # 松下/自定义协议通常用 $ 作为成功标志， ! 作为错误标志
            if "$" in frame:
                # 兼容两种读命令的响应： RD (读寄存器) 或 RC (读触点)
                # 响应通常在 $ 符号后跟着命令字
                # e.g. <EE$RC... 或 %01$RC...
                if "RC" in frame:
                    # 获取状态位 (通常在 RC 之后的一位)
                    # 例如 <EE$RC1BCC
                    idx = frame.find("RC") + 2
                    status_char = frame[idx:idx+1]
                    if status_char in ("0", "1"):
                        status = (status_char == "1")
                        # 更新对应的 R 数组
                        if 0 <= self.current_read_addr < len(self.R):
                            self.R[self.current_read_addr] = status
                            logger.debug(f"更新 R[{self.current_read_addr}] = {status}")

            logger.debug(f"收到有效帧: {frame}")
            self.send_flag = False  # 成功后允许发下一帧

        except Exception as e:
            logger.error(f"解析帧失败: {frame} → {e}")

    def _parse_read_response(self, frame: str):
        """MEWTOCOL 响应已在 _process_frame 中直接处理"""
        pass

    def _poll_loop(self):
        while self.is_running:
            if not self.is_connected:
                time.sleep(0.5)
                continue

            try:
                self._send_next()
            except Exception as e:
                logger.error(f"发送轮询异常: {e}")

            time.sleep(self.poll_interval_ms / 1000.0)

    def _send_next(self):
        with self.send_lock:
            if self.send_flag:
                # 检查超时
                elapsed_ms = (datetime.now() - self.last_send_time).total_seconds() * 1000
                if elapsed_ms > self.timeout_ms:
                    self.timeout_count += 1
                    logger.warning(f"命令超时: {self.current_cmd}")
                    self._handle_send_failed()
                    self.send_flag = False

            if self.send_flag:
                return  # 还在等待响应

            cmd = None
            if not self.write_queue.empty():
                cmd = self.write_queue.get()
            else:
                # 默认空闲周期读取指令（如不需空闲轮询，可直接 return）
                # 注意：实际生产中建议只在主程序显式调用 read_bits
                return

            if not cmd:
                return

            # --- 自动计算并替换 CRC (**) ---
            if "**" in cmd:
                # 记录起始地址（如 R1205 -> 1205），用于解析回读的数据
                m = re.search(r"[Rr](\d+)", cmd)
                if m:
                    self.current_read_addr = int(m.group(1))

                # 提取 ** 前的内容进行校验计算
                parts = cmd.split("**")
                if len(parts) >= 2:
                    content_to_crc = parts[0]
                    suffix = parts[1]
                    real_crc = self._calc_crc(content_to_crc)
                    cmd = f"{content_to_crc}{real_crc}{suffix}"

            self.current_cmd = cmd
            self.last_send_time = datetime.now()
            self.send_flag = True
            self.send_count += 1

            try:
                self.serial_port.write(cmd.encode('ascii'))
                logger.debug(f"发送: {cmd.rstrip()}")
            except Exception as e:
                logger.error(f"发送失败: {cmd} → {e}")
                self.is_connected = False

    def _handle_send_failed(self):
        # 重发逻辑（只对写指令重发）
        if self.current_cmd and self.current_cmd[4] == "W":
            self.resend_count += 1
            if self.resend_count <= self.max_resend:
                logger.info(f"重发 ({self.resend_count}/{self.max_resend}): {self.current_cmd.rstrip()}")
                self.write_queue.put(self.current_cmd)  # 重新入队
            else:
                logger.warning(f"达到最大重试次数，放弃: {self.current_cmd.rstrip()}")
        self.send_flag = False

    @staticmethod
    def _calc_crc(s: str) -> str:
        """
        Panasonic MEWTOCOL BCC (Block Check Code) 计算
        算法：对从 '%' 或 '<' 开始的所有字符进行异或
        """
        bcc = 0
        for c in s:
            bcc ^= ord(c)
        return f"{bcc:02X}"

    # ── 业务友好接口 ─────────────────────────────────────────────

    def write_bit(self, addr: str, value: bool):
        """
        写单个触点 (WCP2 / WCS)
        兼容 C# 代码中的 Fatek-like 封装
        """
        flag = "1" if value else "0"
        # 使用 C# 中的格式: <EE#WCP2 R1205 E 1 R1205 F 1 ** \r
        # 注意: 这里的 E1 和 F1 可能是协议特定的参数
        # 简化为: <EE#WCP2{addr}E1{addr}F{flag}**
        cmd = f"<EE#WCP2{addr}E1{addr}F{flag}**\r"
        self.write_queue.put(cmd)

    def write_word(self, addr: str, value: int):
        """写单个寄存器"""
        pass

    def read_bits(self, start: str, count: int):
        """
        读触点状态 (RCC)
        兼容 C# 代码中的 Fatek-like 封装
        """
        # 站号 EE, 命令 RCC, 地址 start, 数量 count (补足 4 位 hex)
        # e.g. <EE#RCC R1205 0001 ** \r
        cmd = f"<EE#RCC{start}{count:04X}**\r"
        self.write_queue.put(cmd)

    def get_r(self, idx: int) -> bool:
        if 0 <= idx < len(self.R):
            return self.R[idx]
        return False

    def get_status(self) -> dict:
        return {
            "connected": self.is_connected,
            "send_count": self.send_count,
            "recv_ok": self.recv_ok_count,
            "crc_err": self.crc_error_count,
            "timeout": self.timeout_count,
            "reconnects": self.reconnect_count,
            "queue_size": self.write_queue.qsize()
        }


# ── 使用示例 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    plc = FpPlc()

    if plc.open("COM3,38400,8,1,0,8"):
        # 示例：周期性翻转 R120
        try:
            state = False
            while True:
                plc.write_bit("R120", state)
                state = not state
                print(f"R120 → {state}   |  {plc.get_status()}")
                time.sleep(1.2)

                # 读取示例（实际值在周期读取中更新）
                print(f"R120 当前状态: {plc.get_r(120)}")
        except KeyboardInterrupt:
            print("\n用户停止")
        finally:
            plc.close()
    else:
        print("无法连接 PLC")