from harvesters.core import Harvester

def enum_u3v_cameras(cti_path):
    h = Harvester()
    # 加载CTI文件
    h.add_file(cti_path, check_existence=True, check_validity=True)
    h.update()
    # 枚举所有可用相机
    cameras = h.device_info_list
    if cameras:
        print(f"找到{len(cameras)}台相机：")
        for i, cam in enumerate(cameras):
            print(f"[{i}] 型号：{cam.model} | 序列号：{cam.serial_number}")
    else:
        print("未找到USB3 Vision相机")
    # 释放资源
    h.reset()

# 替换为你的cti文件实际路径！
cti_file = r".\ic4-gentl-u3v_x64.cti"
enum_u3v_cameras(cti_file)