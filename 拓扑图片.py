import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ====================== 原有核心函数（保持不变） ======================
def crop_defect_with_mask(defect_img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    expand = 10
    x = max(x - expand, 0)
    y = max(y - expand, 0)
    w = min(w + 2 * expand, defect_img.shape[1] - x)
    h = min(h + 2 * expand, defect_img.shape[0] - y)
    cropped_defect = defect_img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    return cropped_defect, cropped_mask, (x, y, w, h)

def random_transform(defect_piece, mask_piece):
    h, w = defect_piece.shape[:2]
    scale = random.uniform(0.5, 2.0)
    new_w, new_h = int(w * scale), int(h * scale)
    defect_piece = cv2.resize(defect_piece, (new_w, new_h))
    mask_piece = cv2.resize(mask_piece, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    angle = random.uniform(-180, 180)
    M = cv2.getRotationMatrix2D((new_w//2, new_h//2), angle, 1.0)
    defect_piece = cv2.warpAffine(defect_piece, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask_piece = cv2.warpAffine(mask_piece, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return defect_piece, mask_piece

def paste_defect(background, defect_piece, mask_piece, position):
    x, y = position
    h, w = defect_piece.shape[:2]
    alpha = mask_piece.astype(float) / 255.0
    alpha_inv = 1.0 - alpha
    roi = background[y:y+h, x:x+w]
    for c in range(3):
        roi[:, :, c] = (alpha * defect_piece[:, :, c] + alpha_inv * roi[:, :, c])
    background[y:y+h, x:x+w] = roi
    return background

# ====================== 生成函数（带OpenCV实时预览） ======================
def generate_synthetic_samples(
    normal_dir, defect_dir, mask_dir, 
    output_img_dir, output_mask_dir, 
    num_samples, display_delay
):
    if not all(os.path.isdir(p) for p in [normal_dir, defect_dir, mask_dir, output_img_dir]):
        messagebox.showerror("错误", "请检查所有输入文件夹是否正确存在！")
        return
    
    os.makedirs(output_img_dir, exist_ok=True)
    if output_mask_dir:
        os.makedirs(output_mask_dir, exist_ok=True)
    
    normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    defect_files = [f for f in os.listdir(defect_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not normal_files or not defect_files:
        messagebox.showerror("错误", "正常图像或缺陷图像文件夹中没有图片！")
        return
    
    # OpenCV 预览窗口
    cv2.namedWindow('合成图像预览', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('合成图像预览', 900, 600)
    if output_mask_dir:
        cv2.namedWindow('合成Mask预览', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('合成Mask预览', 900, 600)
    
    status_label.config(text="生成中...（按 q 或 Esc 中断）")
    root.update()
    
    for i in range(num_samples):
        bg_file = random.choice(normal_files)
        defect_file = random.choice(defect_files)
        
        bg_path = os.path.join(normal_dir, bg_file)
        defect_path = os.path.join(defect_dir, defect_file)
        mask_path = os.path.join(mask_dir, defect_file)
        
        background = cv2.imread(bg_path)
        defect_img = cv2.imread(defect_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if background is None or defect_img is None or mask is None:
            continue
        
        defect_piece, mask_piece, _ = crop_defect_with_mask(defect_img, mask)
        if defect_piece is None:
            continue
        
        defect_piece, mask_piece = random_transform(defect_piece, mask_piece)
        
        h_bg, w_bg = background.shape[:2]
        h_d, w_d = defect_piece.shape[:2]
        if h_d >= h_bg or w_d >= w_bg:
            continue
        pos_y = random.randint(0, h_bg - h_d - 1)
        pos_x = random.randint(0, w_bg - w_d - 1)
        
        synthetic_img = paste_defect(background.copy(), defect_piece, mask_piece, (pos_x, pos_y))
        
        output_name = f"synthetic_{i:06d}.jpg"
        cv2.imwrite(os.path.join(output_img_dir, output_name), synthetic_img)
        
        if output_mask_dir:
            new_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
            new_mask[pos_y:pos_y+h_d, pos_x:pos_x+w_d] = mask_piece
            cv2.imwrite(os.path.join(output_mask_dir, output_name), new_mask)
        
        # 实时显示
        display_img = synthetic_img.copy()
        cv2.putText(display_img, f"{i+1}/{num_samples}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imshow('合成图像预览', display_img)
        if output_mask_dir:
            mask_vis = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR)
            mask_vis[new_mask > 0] = (0, 0, 255)  # 缺陷标红
            cv2.imshow('合成Mask预览', mask_vis)
        
        key = cv2.waitKey(display_delay) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cv2.destroyAllWindows()
    status_label.config(text="生成完成！")
    messagebox.showinfo("完成", f"成功生成 {i+1} 张合成图像！\n保存在：{output_img_dir}")

# ====================== GUI 界面 ======================
def browse_folder(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)

def start_generation():
    normal_dir = entry_normal.get().strip()
    defect_dir = entry_defect.get().strip()
    mask_dir = entry_mask.get().strip()
    output_img_dir = entry_output_img.get().strip()
    output_mask_dir = entry_output_mask.get().strip() or None
    try:
        num_samples = int(entry_num.get().strip() or "1000")
        display_delay = int(entry_delay.get().strip() or "800")
    except ValueError:
        messagebox.showerror("错误", "生成数量和显示延迟必须是整数！")
        return
    
    if not all([normal_dir, defect_dir, mask_dir, output_img_dir]):
        messagebox.showerror("错误", "请填写所有必填路径！")
        return
    
    # 用线程启动生成，避免GUI卡死
    import threading
    threading.Thread(target=generate_synthetic_samples, args=(
        normal_dir, defect_dir, mask_dir,
        output_img_dir, output_mask_dir,
        num_samples, display_delay
    ), daemon=True).start()

# 创建主窗口
root = tk.Tk()
root.title("缺陷图像合成工具（一键生成 + 实时预览）")
root.geometry("720x520")
root.resizable(False, False)

tk.Label(root, text="缺陷图像合成工具", font=("微软雅黑", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=15)

# 路径输入
tk.Label(root, text="正常背景文件夹：", font=("微软雅黑", 10)).grid(row=1, column=0, sticky="e", padx=10, pady=8)
entry_normal = tk.Entry(root, width=50)
entry_normal.grid(row=1, column=1, padx=5, pady=8)
tk.Button(root, text="浏览", command=lambda: browse_folder(entry_normal)).grid(row=1, column=2, pady=8)

tk.Label(root, text="缺陷图像文件夹：", font=("微软雅黑", 10)).grid(row=2, column=0, sticky="e", padx=10, pady=8)
entry_defect = tk.Entry(root, width=50)
entry_defect.grid(row=2, column=1, padx=5, pady=8)
tk.Button(root, text="浏览", command=lambda: browse_folder(entry_defect)).grid(row=2, column=2, pady=8)

tk.Label(root, text="缺陷Mask文件夹：", font=("微软雅黑", 10)).grid(row=3, column=0, sticky="e", padx=10, pady=8)
entry_mask = tk.Entry(root, width=50)
entry_mask.grid(row=3, column=1, padx=5, pady=8)
tk.Button(root, text="浏览", command=lambda: browse_folder(entry_mask)).grid(row=3, column=2, pady=8)

tk.Label(root, text="输出合成图像文件夹：", font=("微软雅黑", 10)).grid(row=4, column=0, sticky="e", padx=10, pady=8)
entry_output_img = tk.Entry(root, width=50)
entry_output_img.grid(row=4, column=1, padx=5, pady=8)
tk.Button(root, text="浏览", command=lambda: browse_folder(entry_output_img)).grid(row=4, column=2, pady=8)

tk.Label(root, text="输出Mask文件夹（可选）：", font=("微软雅黑", 10)).grid(row=5, column=0, sticky="e", padx=10, pady=8)
entry_output_mask = tk.Entry(root, width=50)
entry_output_mask.grid(row=5, column=1, padx=5, pady=8)
tk.Button(root, text="浏览", command=lambda: browse_folder(entry_output_mask)).grid(row=5, column=2, pady=8)

tk.Label(root, text="生成数量：", font=("微软雅黑", 10)).grid(row=6, column=0, sticky="e", padx=10, pady=8)
entry_num = tk.Entry(root, width=20)
entry_num.insert(0, "1000")
entry_num.grid(row=6, column=1, sticky="w", padx=5, pady=8)

tk.Label(root, text="显示延迟(ms，0=手动)：", font=("微软雅黑", 10)).grid(row=7, column=0, sticky="e", padx=10, pady=8)
entry_delay = tk.Entry(root, width=20)
entry_delay.insert(0, "800")
entry_delay.grid(row=7, column=1, sticky="w", padx=5, pady=8)

# 一键生成按钮
tk.Button(root, text="开始生成", font=("微软雅黑", 14, "bold"), bg="#4CAF50", fg="white", height=2,
          command=start_generation).grid(row=8, column=0, columnspan=3, pady=20)

# 状态栏
status_label = tk.Label(root, text="就绪", font=("微软雅黑", 10), fg="blue")
status_label.grid(row=9, column=0, columnspan=3, pady=10)

root.mainloop()