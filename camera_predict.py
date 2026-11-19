"""摄像头实时预览 + 快门识别 ASL 手语字母（单文件 GUI）

依赖：torch、torchvision、Pillow、opencv-python(-headless)、tkinter（Python 自带）
模型：项目根目录 best_model.pth（resnet 结构，26 类）
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
from torchvision.datasets import ImageFolder

from model import resnet, residual

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
TRAIN_DIR = os.path.join(BASE_DIR, "data1", "train")

PREVIEW_W, PREVIEW_H = 800, 600          # 预览尺寸
GUIDE_RATIO = 0.72                       # 画面中央取景框边长占比
MODEL_SIZE = 224


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL 手语字母识别 - 实时摄像头")
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
        self.last_photo = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<space>", lambda e: self.capture())
        self.root.bind("<Escape>", lambda e: self.on_close())

        self.root.after(15, self.update_frame)
        self.root.after(300, self.keep_focus)

    # ---------- 初始化 ----------
    def _load_classes(self):
        if os.path.isdir(TRAIN_DIR):
            try:
                return ImageFolder(TRAIN_DIR).classes
            except Exception:
                pass
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def _load_model(self):
        if not os.path.isfile(MODEL_PATH):
            messagebox.showerror("缺少模型", "找不到模型文件：\n{}\n\n请先运行 model_train.py 训练模型。"
                                 .format(MODEL_PATH))
            sys.exit(1)
        model = resnet(residual)
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
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

        self.result_label = tk.Label(info, text="—", font=("Consolas", 34, "bold"),
                                     fg="#1a8f3c")
        self.result_label.pack(anchor="w")

        self.detail_label = tk.Label(info, text="点击按钮或按空格开始识别",
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

        frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        self.current_frame = frame

        display = frame.copy()
        x0, y0, x1, y1 = self.guide_box()
        cv2.rectangle(display, (x0, y0), (x1, y1), (80, 220, 80), 2)
        cv2.putText(display, "put your hand here", (x0 + 8, y0 - 10),
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
        side = int(min(PREVIEW_W, PREVIEW_H) * GUIDE_RATIO)
        x0 = (PREVIEW_W - side) // 2
        y0 = (PREVIEW_H - side) // 2
        return x0, y0, x0 + side, y0 + side

    def _update_status(self):
        dev = "CUDA" if self.device.type == "cuda" else "CPU"
        self.status_label.config(text="设备: {}   |   预览 {:.0f} FPS   |   空格 / 按钮拍照   |   ESC 退出"
                                 .format(dev, self.fps))

    # ---------- 拍照识别 ----------
    def capture(self):
        if self.current_frame is None:
            return
        x0, y0, x1, y1 = self.guide_box()
        crop = self.current_frame[y0:y1, x0:x1].copy()
        self.last_photo = crop

        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        cost = (time.time() - t0) * 1000

        top_p, top_i = torch.topk(probs, k=min(3, probs.numel()))
        names = [self.classes[i] for i in top_i.tolist()]
        values = top_p.tolist()

        self.result_label.config(text="{}".format(names[0]))
        self.detail_label.config(
            text="置信度 {:.1%}   |   耗时 {:.0f} ms\n其它可能：{}".format(
                values[0], cost,
                "   ".join("{} {:.1%}".format(n, v) for n, v in zip(names[1:], values[1:]))))
        self.status_label.config(text="识别结果已更新，可继续按空格拍照。")

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
