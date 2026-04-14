from ultralytics import YOLO

# 导出 AI 安装检测模型
model = YOLO("best_seg.pt")  # 你的原始 ONNX 模型路径
model.export(
    format="openvino",    # 导出为 OpenVINO
    imgsz=640,            # 输入尺寸（根据你的模型训练时保持一致）
    half=False,           # 不使用 FP16（CPU 上通常用 FP32）
    device="cpu"          # 强制 CPU
)
# 导出后会生成文件夹：best-seg_openvino_model/ （里面有 openvino_model.xml 和 .bin）

# 导出白点区域分割模型
seg_model = YOLO("best_DW.pt")
seg_model.export(
    format="openvino",
    imgsz=640,
    half=False,
    device="cpu"
)
# 生成文件夹：best_seg_DW_openvino_model/