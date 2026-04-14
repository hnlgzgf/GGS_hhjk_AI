import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from openvino.runtime import Core
import os
import time
from PIL import Image, ImageTk
import random

# 为分割掩膜生成随机颜色（背景透明）
random.seed(42)
num_colors = 21  # 常见分割模型有20类 + 背景
seg_colors = [(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for _ in range(num_colors)]
seg_colors[0] = (0, 0, 0)  # 背景黑色（不绘制）

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenVINO 推理窗体应用程序（支持检测 + 分割）")
        self.geometry("1600x1000")
        self.resizable(True, True)

        # 顶部按钮区
        top_frame = tk.Frame(self, pady=15)
        top_frame.pack(fill=tk.X)

        self.btn_open_img = tk.Button(top_frame, text="打开图片（支持中文路径）", width=25, height=2, command=self.open_image)
        self.btn_open_img.pack(side=tk.LEFT, padx=60)

        self.btn_open_model = tk.Button(top_frame, text="打开模型（.xml）", width=25, height=2, command=self.open_model)
        self.btn_open_model.pack(side=tk.LEFT, padx=60)

        self.btn_infer = tk.Button(top_frame, text="一键推理", width=25, height=2, command=self.infer)
        self.btn_infer.pack(side=tk.LEFT, padx=60)

        self.time_label = tk.Label(top_frame, text="推理耗时: - ", font=("Arial", 14, "bold"), fg="blue")
        self.time_label.pack(side=tk.RIGHT, padx=60)

        # 主显示区
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=30)
        tk.Label(left_frame, text="原图", font=("Arial", 16, "bold")).pack()
        self.left_label = tk.Label(left_frame, bg="lightgray")
        self.left_label.pack()

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=30)
        tk.Label(right_frame, text="推理结果（检测框 / 分割掩膜叠加）", font=("Arial", 16, "bold")).pack()
        self.right_label = tk.Label(right_frame, bg="lightgray")
        self.right_label.pack()

        self.original_image = None
        self.result_image = None
        self.photo_orig = None
        self.photo_res = None

        self.display_original()
        self.display_result()

    def cv2_to_photo(self, cv_img):
        if cv_img is None or cv_img.size == 0:
            return None
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # 更大显示区域，适应大图
        pil_img.thumbnail((750, 750), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def display_original(self):
        if self.original_image is not None:
            self.photo_orig = self.cv2_to_photo(self.original_image)
            self.left_label.config(image=self.photo_orig)
        else:
            self.left_label.config(image="", text="原图将显示在这里\n（请先打开图片）", font=("Arial", 18), fg="gray", justify="center")

    def display_result(self):
        if self.result_image is not None:
            self.photo_res = self.cv2_to_photo(self.result_image)
            self.right_label.config(image=self.photo_res)
        else:
            self.right_label.config(image="", text="推理结果将显示在这里\n（支持检测框或掩膜叠加）", font=("Arial", 18), fg="gray", justify="center")

    def open_image(self):
        path = filedialog.askopenfilename(title="选择图片", filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not path:
            return
        img_array = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("错误", "无法加载图片")
            return
        self.original_image = img
        self.result_image = None
        self.display_original()
        self.display_result()
        self.time_label.config(text="推理耗时: - ")

    def open_model(self):
        path = filedialog.askopenfilename(title="选择OpenVINO IR模型（.xml）", filetypes=[("XML文件", "*.xml")])
        if not path:
            return
        bin_path = os.path.splitext(path)[0] + ".bin"
        if not os.path.isfile(bin_path):
            messagebox.showerror("错误", f"未找到对应的.bin文件:\n{bin_path}")
            return
        try:
            core = Core()
            model = core.read_model(path)
            compiled_model = core.compile_model(model, "CPU")
            input_layer = compiled_model.input(0)
            self.compiled_model = compiled_model
            self.input_layer = input_layer
            self.output_layer = compiled_model.output(0)
            self.input_height = input_layer.shape[2]
            self.input_width = input_layer.shape[3]
            messagebox.showinfo("成功", f"模型加载成功\n输入尺寸: {self.input_width}x{self.input_height}")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败:\n{str(e)}")

    def infer(self):
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        if not hasattr(self, "compiled_model"):
            messagebox.showwarning("提示", "请先打开模型")
            return

        orig_h, orig_w = self.original_image.shape[:2]

        # Letterbox 预处理（保持比例）
        ratio = min(self.input_width / orig_w, self.input_height / orig_h)
        new_w, new_h = int(orig_w * ratio + 0.5), int(orig_h * ratio + 0.5)
        resized = cv2.resize(self.original_image, (new_w, new_h))
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        top = (self.input_height - new_h) // 2
        left = (self.input_width - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        input_data = np.transpose(canvas, (2, 0, 1))[np.newaxis].astype(np.float32)

        start_time = time.perf_counter()
        results = self.compiled_model(input_data)[self.output_layer]
        infer_time = time.perf_counter() - start_time
        self.time_label.config(text=f"推理耗时: {infer_time * 1000:.2f} ms")

        result_img = self.original_image.copy()

        # 自动判断模型类型并可视化
        output = results[0] if results.ndim == 4 else results
        shape = output.shape

        # 1. 尝试物体检测格式 [N,7] 或 [1,N,7]
        if len(shape) == 2 and shape[1] == 7:
            detections = output if output.shape[0] > 7 else output[0]
            for det in detections:
                conf = float(det[2])
                if conf < 0.5: continue
                class_id = int(det[1])
                xmin = max(0, int((det[3] * self.input_width - left) / ratio))
                ymin = max(0, int((det[4] * self.input_height - top) / ratio))
                xmax = min(orig_w, int((det[5] * self.input_width - left) / ratio))
                ymax = min(orig_h, int((det[6] * self.input_height - top) / ratio))
                color = seg_colors[(class_id + 1) % len(seg_colors)]
                cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), color, 3)
                cv2.putText(result_img, f"class_{class_id} {conf:.2f}", (xmin, ymin-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 2. 尝试语义分割格式 [C, H, W] 或 [1, C, H, W] 或 [H, W]
        elif len(shape) == 3 or (len(shape) == 4 and shape[0] == 1):
            if len(shape) == 4:
                mask = output[0]
            else:
                mask = output

            # 如果是 [1, H, W] 二值掩膜
            if mask.shape[0] == 1:
                mask = mask[0]
                mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                overlay = result_img.copy()
                overlay[mask_resized > 0.5] = [0, 0, 255]  # 红色掩膜
                result_img = cv2.addWeighted(result_img, 0.6, overlay, 0.4, 0)

            # 如果是 [C, H, W] 多类
            else:
                class_map = np.argmax(mask, axis=0)
                mask_resized = cv2.resize(class_map.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                overlay = np.zeros_like(result_img)
                for cid in range(1, min(num_colors, mask.shape[0])):
                    overlay[mask_resized == cid] = seg_colors[cid]
                result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)

        else:
            messagebox.showwarning("提示", f"未知输出格式 {results.shape}，已显示原图")
        
        if np.array_equal(result_img, self.original_image):
            cv2.putText(result_img, "未检测/分割到目标", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        self.result_image = result_img
        self.display_result()


if __name__ == "__main__":
    app = App()
    app.mainloop()