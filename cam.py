from harvesters.core import Harvester
import cv2
import numpy as np
import os

# 设置 CTI 文件路径（根据你的实际安装路径修改！）
CTI_FILE_PATH = r"ic4-gentl-u3v_x64.cti"

def main():
    # 创建 Harvester 实例
    h = Harvester()
    
    # 加载 GenTL Producer
    if not os.path.exists(CTI_FILE_PATH):
        print(f"❌ 找不到 CTI 文件: {CTI_FILE_PATH}")
        print("请确认 JAI USB3 驱动已安装，并检查路径是否正确。")
        return
    
    h.add_file(CTI_FILE_PATH)
    h.update()

    if len(h.device_info_list) == 0:
        print("❌ 没有检测到 JAI USB3 相机，请检查连接和驱动。")
        return

    print(f"✅ 检测到 {len(h.device_info_list)} 台相机")
    for i, info in enumerate(h.device_info_list):
        print(f"  [{i}] {info.model} (ID: {info.serial_number})")

    # 连接第一台相机
    ia = h.create_image_acquirer(0)
    
    # 可选：设置参数（例如曝光时间、像素格式等）
    try:
        ia.remote_device.node_map.Width.value = 2448
        ia.remote_device.node_map.Height.value = 2048
        ia.remote_device.node_map.ExposureTime.value = 20000  #
        ia.remote_device.node_map.PixelFormat.value = "mono8"  # 或 "BayerRG8"，“mono8” 等
        ia.remote_device.node_map.AcquisitionFrameRateEnable.value = True
        ia.remote_device.node_map.AcquisitionFrameRate.value = 10.0
    except Exception as e:
        print(f"⚠️ 设置参数时出错（可能不支持）: {e}")

    # 开始采集
    ia.start_acquisition()

    print("📸 正在采集一帧图像...")
    with ia.fetch_buffer() as buffer:
        component = buffer.payload.components[0]
        # 转为 NumPy 数组
        data = component.data.reshape(component.height, component.width)
        
        # 如果是 Bayer 格式，需要去马赛克
        if component.data_format.endswith('BayerRG8'):
            img = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2BGR)
        elif component.data_format == 'Mono8':
            img = data.copy()
        else:
            img = data.copy()  # 其他格式先原样输出

        # 保存图像
        cv2.imwrite("jai_capture.png", img)
        print("✅ 图像已保存为 jai_capture.png")

    # 停止采集并释放资源
    ia.stop_acquisition()
    ia.destroy()
    h.reset()

if __name__ == "__main__":
    main()