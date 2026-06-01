import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np


class EventPixelCounterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("事件流红绿像素统计工具")

        self.image = None              # PIL 原图
        self.image_np = None           # numpy 原图 RGB
        self.tk_image = None

        self.scale = 1.0               # 显示缩放比例
        self.display_width = 0
        self.display_height = 0

        self.start_x = None
        self.start_y = None
        self.rect_id = None

        self.create_ui()

    def create_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=8, pady=6)

        open_btn = tk.Button(top_frame, text="打开图片", command=self.open_image)
        open_btn.pack(side=tk.LEFT, padx=4)

        reset_btn = tk.Button(top_frame, text="清除选择", command=self.clear_selection)
        reset_btn.pack(side=tk.LEFT, padx=4)

        self.info_label = tk.Label(
            top_frame,
            text="请打开图片，然后拖拽鼠标选择矩形区域",
            anchor="w"
        )
        self.info_label.pack(side=tk.LEFT, padx=10)

        threshold_frame = tk.Frame(self.root)
        threshold_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(threshold_frame, text="红色阈值:").pack(side=tk.LEFT)
        self.red_threshold = tk.IntVar(value=120)
        tk.Entry(threshold_frame, textvariable=self.red_threshold, width=6).pack(side=tk.LEFT, padx=4)

        tk.Label(threshold_frame, text="绿色阈值:").pack(side=tk.LEFT)
        self.green_threshold = tk.IntVar(value=120)
        tk.Entry(threshold_frame, textvariable=self.green_threshold, width=6).pack(side=tk.LEFT, padx=4)

        tk.Label(
            threshold_frame,
            text="说明：像素通道值大于阈值，并且明显强于其他通道时，会被计为对应颜色"
        ).pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(self.root, height=7)
        self.result_text.pack(fill=tk.X, padx=8, pady=6)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="选择事件流可视化图片",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        self.image = Image.open(file_path).convert("RGB")
        self.image_np = np.array(self.image)

        self.show_image()
        self.clear_selection()

        self.info_label.config(text=f"已打开: {file_path}")

    def show_image(self):
        if self.image is None:
            return

        max_width = 1200
        max_height = 800

        w, h = self.image.size

        self.scale = min(max_width / w, max_height / h, 1.0)
        self.display_width = int(w * self.scale)
        self.display_height = int(h * self.scale)

        display_image = self.image.resize(
            (self.display_width, self.display_height),
            Image.Resampling.LANCZOS
        )

        self.tk_image = ImageTk.PhotoImage(display_image)

        self.canvas.delete("all")
        self.canvas.config(width=self.display_width, height=self.display_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def clear_selection(self):
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

        self.start_x = None
        self.start_y = None
        self.result_text.delete("1.0", tk.END)

    def on_mouse_down(self, event):
        if self.image is None:
            return

        self.start_x = event.x
        self.start_y = event.y

        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)

        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="yellow",
            width=2
        )

    def on_mouse_drag(self, event):
        if self.image is None or self.rect_id is None:
            return

        self.canvas.coords(
            self.rect_id,
            self.start_x,
            self.start_y,
            event.x,
            event.y
        )

    def on_mouse_up(self, event):
        if self.image is None:
            return

        end_x = event.x
        end_y = event.y

        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)

        # 限制在显示图片范围内
        x1 = max(0, min(x1, self.display_width - 1))
        y1 = max(0, min(y1, self.display_height - 1))
        x2 = max(0, min(x2, self.display_width - 1))
        y2 = max(0, min(y2, self.display_height - 1))

        if x2 <= x1 or y2 <= y1:
            messagebox.showwarning("无效区域", "请选择一个有效的矩形区域")
            return

        # 显示坐标转换成原图坐标
        orig_x1 = int(x1 / self.scale)
        orig_y1 = int(y1 / self.scale)
        orig_x2 = int(x2 / self.scale)
        orig_y2 = int(y2 / self.scale)

        self.count_pixels(orig_x1, orig_y1, orig_x2, orig_y2)

    def count_pixels(self, x1, y1, x2, y2):
        roi = self.image_np[y1:y2, x1:x2]

        if roi.size == 0:
            return

        r = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        b = roi[:, :, 2].astype(np.int16)

        red_th = self.red_threshold.get()
        green_th = self.green_threshold.get()

        # 红色像素判断：
        # R 通道足够大，并且明显强于 G、B
        red_mask = (
            (r >= red_th) &
            (r > g * 1.3) &
            (r > b * 1.3)
        )

        # 绿色像素判断：
        # G 通道足够大，并且明显强于 R、B
        green_mask = (
            (g >= green_th) &
            (g > r * 1.3) &
            (g > b * 1.3)
        )

        red_count = int(np.sum(red_mask))
        green_count = int(np.sum(green_mask))

        total_pixels = roi.shape[0] * roi.shape[1]

        result = f"""
选中区域原图坐标:
x1 = {x1}, y1 = {y1}
x2 = {x2}, y2 = {y2}

区域尺寸:
width = {x2 - x1}
height = {y2 - y1}
total pixels = {total_pixels}

统计结果:
红色像素数 = {red_count}
绿色像素数 = {green_count}

比例:
红色占比 = {red_count / total_pixels:.6f}
绿色占比 = {green_count / total_pixels:.6f}
红绿总数 = {red_count + green_count}
红绿差值 = {red_count - green_count}
"""

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, result)


if __name__ == "__main__":
    root = tk.Tk()
    app = EventPixelCounterUI(root)
    root.mainloop()