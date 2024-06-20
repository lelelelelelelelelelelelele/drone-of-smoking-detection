import tkinter as tk
from tkinter import ttk
from djitellopy import Tello
import cv2
from PIL import Image, ImageTk
# import numpy as np
# import os
from datetime import datetime
import threading
# from ultralytics import YOLO
from yolo_detection import YOLODetector
import tkinter.messagebox  # 导入messagebox模块用于弹出提示框


class VideoRecorderThread(threading.Thread):
    def __init__(self, tello, should_stop, canvas_update_func):
        super().__init__()
        self.tello = tello
        self.should_stop = should_stop
        self.canvas_update_func = canvas_update_func
        self.daemon = True
        self.detector = YOLODetector("best_smoke.pt")
        self.recording = False

    def run(self):
        self.start_video_and_recording()

    def start_video_and_recording(self):
        self.tello.streamon()
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = f'tello_video_{now}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (960, 720))

        while not self.should_stop.is_set():
            frame = self.tello.get_frame_read().frame
            if frame is not None:
                annotated_frame, smoke_detected, _, _, _ = self.detector.findsmoke(frame)
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)
                self.canvas_update_func(img_tk)
                
                # 控制LED和点阵屏显示
                if smoke_detected:
                    self.tello.send_expansion_command("led 0 0 255")  # 红色LED
                    self.tello.send_expansion_command("mled l b 1 FORBIDDEN")
                    if not self.recording:
                        # 实际启动录制的逻辑需要实现
                        self.recording = True
                        print("Started recording.")
                else:
                    self.tello.send_expansion_command("led 0 255 0")  # 绿色LED
                    self.tello.send_expansion_command("mled l b 1 PATROL")
                    if self.recording:
                        # 实际停止录制的逻辑需要实现
                        self.recording = False
                        print("Stopped recording.")
                
                # 注意：此处未实现视频帧的实际写入逻辑，需根据需求决定是否启用
                out.write(annotated_frame)

        self.tello.streamoff()
        # 如果有视频录制，则释放资源
        out.release()
        print("Video processing completed.")


class TelloStatusControlGUI:
    def __init__(self, master):
        self.master = master
        master.title("Tello Drone Status, Control & Auto Video Recording")

        self.tello = Tello()
        self.tello.connect()
        print("Connected to Tello.")

        self.should_stop = threading.Event()
        self.create_widgets()
        self.master.after(1000, self.update_status)  
        self.video_thread = VideoRecorderThread(self.tello, self.should_stop, self.update_canvas_image)
        self.video_thread.start()

    def create_widgets(self):
        self.status_label = ttk.Label(self.master, text="Status: Disconnected", font=("Arial", 12))
        self.status_label.pack(pady=10)
        self.battery_label = ttk.Label(self.master, text="Battery: Unknown%", font=("Arial", 12))
        self.battery_label.pack(pady=5)
        self.takeoff_button = ttk.Button(self.master, text="Take Off", command=self.takeoff)
        self.takeoff_button.pack(ipadx=5, ipady=5, pady=10)
        self.land_button = ttk.Button(self.master, text="Land", command=self.land)
        self.land_button.pack(ipadx=5, ipady=5, pady=10)
        self.canvas = tk.Canvas(self.master, width=960, height=720)
        self.canvas.pack()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_canvas_image(self, img_tk):
        self.canvas.image = img_tk  # 保留图像的引用以防止垃圾回收
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def on_close(self):
        """处理窗口关闭事件，设置停止标志并等待线程结束"""
        self.should_stop.set()
        self.master.destroy()

    def update_status(self):
        """更新无人机状态和电量显示"""
        try:
            battery = self.tello.get_battery()
            status = "Connected" if battery > 0 else "Disconnected"
            self.update_labels(status, battery)
            self.master.after(1000, self.update_status)  # 继续定时更新
        except Exception as e:
            print(f"Error updating status: {e}")
            self.update_labels("Disconnected", "Unknown")

    def update_labels(self, status, battery):
        """更新状态和电量标签的文本"""
        self.status_label.config(text=f"Status: {status}")
        self.battery_label.config(text=f"Battery: {battery}%")

    def takeoff(self):
        self.tello.takeoff()

    def land(self):
        self.tello.land()

if __name__ == "__main__":
    root = tk.Tk()
    app = TelloStatusControlGUI(master=root)
    # 设置窗口标题
    root.title("GUI")

    # 设置窗口的初始宽度和高度
    window_width = 1280  # 你可以根据需要调整这些值
    window_height = 1080

    # 设置窗口的大小
    root.geometry(f"{window_width}x{window_height}")

    root.mainloop()
