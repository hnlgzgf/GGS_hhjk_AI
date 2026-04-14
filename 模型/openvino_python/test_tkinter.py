import tkinter as tk
from tkinter import messagebox

print("测试Tkinter...")

root = tk.Tk()
root.title("测试")
root.geometry("300x200")

label = tk.Label(root, text="Tkinter测试成功！", font=("Arial", 14))
label.pack(pady=50)

button = tk.Button(root, text="点击测试", command=lambda: messagebox.showinfo("测试", "消息框测试成功！"))
button.pack()

print("进入主循环...")
root.mainloop()
print("主循环退出")
