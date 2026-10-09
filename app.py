"""Desktop GUI for the final six-class fruit YOLO detector."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("FRUIT_MODEL_PATH", BASE / "models" / "best.pt"))
OUTPUT_DIR = BASE / "results" / "gui"
VIDEO_DIR = BASE / "results" / "video_results"


class YOLOGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("六类水果 YOLO 检测 - 图片 / 摄像头 / 视频")
        self.root.geometry("1280x860")
        self.root.minsize(900, 650)
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.ready = False
        self.closed = False
        self.stream_id = 0
        self.stream_after = None
        self.capture = None
        self.writer = None
        self.running = False
        self.mode = None
        self.current_image = None
        self.original_image = None
        self.photo = None
        self.total_boxes = 0
        self.max_people = 0
        self.class_counts = {}
        self.frame_index = 0

        top = tk.Frame(root, padx=14, pady=10, bg="#f3f6fa")
        top.pack(fill="x")
        self.button(top, "选择图片", self.choose_image).pack(side="left", padx=3)
        self.button(top, "检测图片", self.detect_image).pack(side="left", padx=3)
        self.button(top, "选择视频并播放", self.choose_video).pack(side="left", padx=3)
        self.camera_button = self.button(top, "打开摄像头", self.toggle_camera)
        self.camera_button.pack(side="left", padx=3)
        self.stop_button = self.button(top, "停止视频/摄像头", self.stop_stream)
        self.stop_button.pack(side="left", padx=3)
        self.button(top, "保存当前图片", self.save_current).pack(side="left", padx=3)

        settings = tk.Frame(root, padx=14, pady=7)
        settings.pack(fill="x")
        classes = "apple、orange、banana、kiwi、lemon、strawberry"
        tk.Label(settings, text=f"类别：{classes}", font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        tk.Label(settings, text="置信度阈值", padx=22).pack(side="left")
        self.confidence = tk.DoubleVar(value=0.35)
        tk.Scale(settings, from_=0.10, to=0.90, resolution=0.05, orient="horizontal",
                 variable=self.confidence, length=190).pack(side="left")
        self.stats = tk.StringVar(value="当前水果：0  |  累计检测框：0  |  单帧最多：0")
        tk.Label(settings, textvariable=self.stats, fg="#0b6b36", font=("Microsoft YaHei", 11, "bold")).pack(side="right")

        self.status = tk.StringVar(value="正在加载并预热模型，请稍候…")
        tk.Label(root, textvariable=self.status, anchor="w", padx=16, fg="#174ea6").pack(fill="x")
        self.canvas = tk.Label(root, text="六类水果 YOLO 检测\n\n选择图片 / 视频，或打开摄像头开始",
                               bg="#17212f", fg="white", font=("Microsoft YaHei", 20), justify="center")
        self.canvas.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        # Run the first CUDA inference before the user clicks a detection button.
        self.submit(self.load_model, self.model_ready)

    def load_model(self):
        self.model = YOLO(MODEL_PATH)
        self.model.predict(np.zeros((640, 640, 3), dtype="uint8"), imgsz=640, verbose=False)

    def model_ready(self, _):
        self.ready = True
        self.status.set("模型已就绪，请选择图片、视频或摄像头")

    def submit(self, work, callback, *args):
        if self.pending is not None:
            self.status.set("正在处理，请稍候…")
            return False
        self.pending = self.executor.submit(work, *args)
        self.root.after(30, lambda: self.poll(callback))
        return True

    def poll(self, callback):
        if self.closed:
            return
        if not self.pending.done():
            self.root.after(30, lambda: self.poll(callback))
            return
        future, self.pending = self.pending, None
        try:
            callback(future.result())
        except Exception as exc:
            self.stop_stream()
            self.status.set(f"处理失败：{exc}")
            messagebox.showerror("处理失败", str(exc))

    def available(self):
        if not self.ready or self.pending is not None:
            self.status.set("模型正在加载或处理，请稍候…")
            return False
        return True

    @staticmethod
    def button(parent, text, command):
        return tk.Button(parent, text=text, command=command, width=14, height=1,
                         font=("Microsoft YaHei", 10), cursor="hand2")

    def reset_stats(self):
        self.total_boxes = self.max_people = self.frame_index = 0
        self.class_counts = {}
        self.refresh_stats(0)

    def refresh_stats(self, count):
        detail = "、".join(f"{name}:{num}" for name, num in self.class_counts.items()) or "无"
        self.stats.set(f"当前水果：{count}（{detail}） | 累计检测框：{self.total_boxes} | 单帧最多：{self.max_people}")

    def choose_image(self):
        if not self.available():
            return
        path = filedialog.askopenfilename(title="选择待检测图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.stop_stream()
        self.current_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.current_image is None:
            messagebox.showerror("打开失败", "无法读取这张图片。")
            return
        self.reset_stats()
        self.original_image = self.current_image.copy()
        self.status.set(f"已选择图片：{Path(path).name}；点击“检测图片”开始")
        self.show_frame(self.current_image)

    def choose_video(self):
        if not self.available():
            return
        path = filedialog.askopenfilename(title="选择待检测视频", filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.stop_stream()
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            self.stop_stream()
            messagebox.showerror("打开失败", "无法读取这个视频。")
            return
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25
        width, height = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = VIDEO_DIR / f"detected_{Path(path).stem}.mp4"
        self.writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            self.stop_stream()
            messagebox.showerror("保存失败", "无法创建输出视频，请检查目录和视频编码器。")
            return
        self.mode, self.running = "video", True
        self.reset_stats()
        self.status.set(f"正在检测视频：{Path(path).name}；带框视频将保存到 {output.name}")
        self.update_stream()

    def detect(self, frame, confidence):
        result = self.model.predict(frame, conf=confidence, imgsz=640, verbose=False)[0]
        count = 0 if result.boxes is None else len(result.boxes)
        class_counts = {}
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().tolist():
                name = result.names[class_id]
                class_counts[name] = class_counts.get(name, 0) + 1
        return result.plot(conf=True, labels=True, boxes=True), count, class_counts

    def detect_image(self):
        if self.original_image is None:
            messagebox.showinfo("提示", "请先选择一张图片。")
            return
        if not self.available():
            return
        self.stop_stream()
        self.status.set("正在检测图片…")
        self.submit(self.detect, self.image_ready, self.original_image.copy(), self.confidence.get())

    def image_ready(self, detection):
        annotated, count, self.class_counts = detection
        self.current_image = annotated
        self.total_boxes, self.max_people = count, count
        self.refresh_stats(count)
        self.show_frame(annotated)
        self.status.set(f"图片检测完成：识别到 {count} 个水果，阈值 = {self.confidence.get():.2f}")

    def toggle_camera(self):
        if self.mode == "camera" and self.running:
            self.stop_stream()
            return
        if not self.available():
            return
        self.stop_stream()
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("摄像头不可用", "无法打开摄像头。请检查摄像头是否被其他程序占用。")
            return
        self.mode, self.running = "camera", True
        self.camera_button.configure(text="关闭摄像头")
        self.reset_stats()
        self.status.set("摄像头实时检测已启动")
        self.update_stream()

    def update_stream(self):
        self.stream_after = None
        if not self.running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            if self.mode == "video":
                self.status.set("视频检测完成，带框结果已保存到 results/video_results")
            self.stop_stream()
            return
        generation = self.stream_id
        self.submit(self.detect, lambda detection: self.stream_ready(detection, generation), frame, self.confidence.get())

    def stream_ready(self, detection, generation):
        if not self.running or generation != self.stream_id:
            return
        annotated, count, self.class_counts = detection
        self.current_image = annotated
        self.frame_index += 1
        self.total_boxes += count
        self.max_people = max(self.max_people, count)
        self.refresh_stats(count)
        self.show_frame(annotated)
        if self.writer is not None:
            self.writer.write(annotated)
        source = "摄像头" if self.mode == "camera" else "视频"
        self.status.set(f"{source}实时检测中：第 {self.frame_index} 帧，当前 {count} 个水果，阈值 = {self.confidence.get():.2f}")
        self.stream_after = self.root.after(10, self.update_stream)

    def stop_stream(self):
        self.stream_id += 1
        if self.stream_after is not None:
            self.root.after_cancel(self.stream_after)
            self.stream_after = None
        was_video = self.mode == "video"
        self.running = False
        self.mode = None
        self.camera_button.configure(text="打开摄像头")
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if was_video:
            self.status.set("视频已停止，已保存当前检测结果到 results/video_results")

    def save_current(self):
        if self.current_image is None:
            messagebox.showinfo("提示", "还没有可保存的检测结果。")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = OUTPUT_DIR / f"detection_{len(list(OUTPUT_DIR.glob('*.jpg'))) + 1:03d}.jpg"
        ok, encoded = cv2.imencode(".jpg", self.current_image)
        if not ok:
            messagebox.showerror("保存失败", "无法编码当前图片。")
            return
        encoded.tofile(str(filename))
        self.status.set(f"当前帧已保存：{filename.name}")

    def show_frame(self, frame):
        height, width = frame.shape[:2]
        scale = min(max(self.canvas.winfo_width(), 100) / width, max(self.canvas.winfo_height(), 100) / height)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.configure(image=self.photo, text="")

    def close(self):
        self.closed = True
        self.stop_stream()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == "__main__":
    app = tk.Tk()
    gui = YOLOGui(app)
    app.protocol("WM_DELETE_WINDOW", gui.close)
    app.mainloop()
