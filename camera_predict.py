"""摄像头实时预览 + 快门识别「是否戴眼镜」（单文件 GUI）

依赖：torch、torchvision、Pillow、opencv-python(-headless)、tkinter（Python 自带）
模型：项目根目录 best_model.pth（resnet 结构，2 类）
类别：自动从 train 目录读取（with_glasses / without_glasses）
取景：把脸放进画面中央的绿色方框，拍照时裁框内方形区域送模型
预处理：与 model_train.py 保持一致 —— Resize((224,224)) + ToTensor，不做 normalize
"""

import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 必须在 import torch 之前

import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk
import torch
from torchvision import transforms

from model import resnet, residual

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
DATASET_DIR = os.path.join(BASE_DIR, "train")

PREVIEW_W, PREVIEW_H = 800, 600          # 预览尺寸
MODEL_SIZE = 224
GUIDE_SIDE = int(PREVIEW_H * 0.55)       # 取景框边长（正方形），框内区域送模型

CN_NAMES = {"with_glasses": "戴眼镜", "without_glasses": "不戴眼镜"}


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("眼镜识别 - 实时摄像头")
        self.root.resizable(False, False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
                                              transforms.ToTensor()])
        self.classes = self._load_classes()
        self.model = self._load_model()

        self.cap = self._open_camera()
        self.current_frame = None        # 当前预览帧（BGR, PREVIEW_W x PREVIEW_H）
        self.frame_count = 0
        self.fps = 0.0
        self.fps_time = time.time()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<space>", lambda e: self.capture())
        self.root.bind("<Escape>", lambda e: self.on_close())

        self.root.after(15, self.update_frame)
        self.root.after(300, self.keep_focus)

    # ---------- 初始化 ----------
    def _load_classes(self):
        if not os.path.isdir(DATASET_DIR):
            messagebox.showerror("缺少数据", "找不到数据目录：\n{}\n\n请把数据按类别放在 train 下。"
                                 .format(DATASET_DIR))
            sys.exit(1)
        classes = sorted(d for d in os.listdir(DATASET_DIR)
                         if os.path.isdir(os.path.join(DATASET_DIR, d)))
        if not classes:
            messagebox.showerror("没有类别", "{} 里没有找到类别子文件夹。".format(DATASET_DIR))
            sys.exit(1)
        return classes

    def _load_model(self):
        if not os.path.isfile(MODEL_PATH):
            messagebox.showerror("缺少模型", "找不到模型文件：\n{}\n\n请先运行 model_train.py 训练模型。"
                                 .format(MODEL_PATH))
            sys.exit(1)
        state = torch.load(MODEL_PATH, map_location="cpu")
        n_out = state["b6.2.weight"].shape[0]

        model = resnet(residual)
        if n_out != 26:
            # model.py 默认是 26 类头，这里按 checkpoint 的实际输出维度适配
            model.b6[2] = torch.nn.Linear(512, n_out)
        model.load_state_dict(state)

        if n_out != len(self.classes):
            messagebox.showerror("模型与数据不匹配",
                                 "模型输出 {} 类，但 train 里有 {} 类。\n"
                                 "请重新训练后再使用。".format(n_out, len(self.classes)))
            sys.exit(1)

        model.to(self.device).eval()
        # 预热一次，避免第一次点击时卡顿
        with torch.no_grad():
            model(torch.zeros(1, 3, MODEL_SIZE, MODEL_SIZE, device=self.device))
        return model

    def _open_camera(self):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
        messagebox.showerror("摄像头打开失败", "没有检测到可用的摄像头，或摄像头被其他程序占用。")
        sys.exit(1)

    def _build_ui(self):
        self.video_label = tk.Label(self.root, bg="#111111", width=PREVIEW_W, height=PREVIEW_H)
        self.video_label.pack(padx=10, pady=(10, 6))

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.shutter_btn = tk.Button(bottom, text="📷  拍照识别  (空格)",
                                     font=("Microsoft YaHei", 15, "bold"),
                                     bg="#2d7ff9", fg="white",
                                     activebackground="#1b66d6", activeforeground="white",
                                     relief="flat", padx=18, pady=10,
                                     command=self.capture)
        self.shutter_btn.pack(side="left")

        info = tk.Frame(bottom)
        info.pack(side="left", padx=18, fill="x", expand=True)

        self.result_label = tk.Label(info, text="—", font=("Microsoft YaHei", 30, "bold"),
                                     fg="#1a8f3c")
        self.result_label.pack(anchor="w")

        self.detail_label = tk.Label(info, text="把脸放进绿框，然后点按钮或按空格",
                                     font=("Microsoft YaHei", 11), fg="#555555", justify="left")
        self.detail_label.pack(anchor="w")

        self.status_label = tk.Label(self.root, text="", font=("Microsoft YaHei", 9),
                                     fg="#777777", anchor="w")
        self.status_label.pack(fill="x", padx=12, pady=(0, 8))

    def keep_focus(self):
        self.root.focus_force()

    # ---------- 摄像头循环 ----------
    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.status_label.config(text="读取摄像头画面失败，请检查设备…")
            self.root.after(100, self.update_frame)
            return

        # 摄像头一般是 16:9，先居中裁成 4:3 再缩放，避免画面被拉扁
        h, w = frame.shape[:2]
        target_w = int(h * 4 / 3)
        if w > target_w:
            x0 = (w - target_w) // 2
            frame = frame[:, x0:x0 + target_w]
        elif w < target_w:
            target_h = int(w * 3 / 4)
            y0 = (h - target_h) // 2
            frame = frame[y0:y0 + target_h, :]

        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        self.current_frame = frame

        # 绿色取景框：拍照时只裁框内区域（数据集是脸部特写，构图要接近）
        display = frame.copy()
        gx0, gy0, gx1, gy1 = self.guide_box()
        cv2.rectangle(display, (gx0, gy0), (gx1, gy1), (80, 220, 80), 2)
        cv2.putText(display, "put your face here", (gx0, max(20, gy0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 1, cv2.LINE_AA)
        cv2.putText(display, "only the area inside the box is used", (12, PREVIEW_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self.photo)

        self.frame_count += 1
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.fps_time)
            self.frame_count = 0
            self.fps_time = now
        self._update_status()

        self.root.after(15, self.update_frame)

    def guide_box(self):
        gx0 = (PREVIEW_W - GUIDE_SIDE) // 2
        gy0 = (PREVIEW_H - GUIDE_SIDE) // 2
        return gx0, gy0, gx0 + GUIDE_SIDE, gy0 + GUIDE_SIDE

    def _update_status(self):
        dev = "CUDA" if self.device.type == "cuda" else "CPU"
        self.status_label.config(
            text="设备: {}   |   预览 {:.0f} FPS   |   类别: {}   |   空格拍照   |   ESC 退出"
                 .format(dev, self.fps, " / ".join(self.classes)))

    # ---------- 拍照识别 ----------
    def capture(self):
        if self.current_frame is None:
            return

        gx0, gy0, gx1, gy1 = self.guide_box()
        crop = self.current_frame[gy0:gy1, gx0:gx1].copy()

        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        cost = (time.time() - t0) * 1000

        top_p, top_i = torch.topk(probs, k=min(2, probs.numel()))
        names = [self.classes[i] for i in top_i.tolist()]
        values = top_p.tolist()

        self.result_label.config(text=self._display_name(names[0]))
        self.detail_label.config(
            text="{}   |   置信度 {:.1%}   |   耗时 {:.0f} ms\n"
                 "另一类：{} {:.1%}".format(
                     names[0], values[0], cost,
                     names[1] if len(names) > 1 else "-",
                     values[1] if len(values) > 1 else 0.0))
        self.status_label.config(text="识别结果已更新，可继续按空格拍照。")

    def _display_name(self, name):
        cn = CN_NAMES.get(name.lower())
        return "{}  {}".format(cn, name) if cn else name

    # ---------- 退出 ----------
    def on_close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    CameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
