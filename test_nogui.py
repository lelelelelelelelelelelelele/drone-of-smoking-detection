import cv2
from PIL import Image
from djitellopy import Tello
import threading
from datetime import datetime
from yolo_detection import YOLODetector


class VideoRecorderThread(threading.Thread):
    def __init__(self, tello, should_stop):
        super().__init__()
        self.tello = tello
        self.should_stop = should_stop
        self.daemon = True
        self.detector = YOLODetector("best_smoke.pt")
        self.recording = False

    def run(self):
        self.start_video_and_recording()

    def start_video_and_recording(self):
        self.tello.streamon()
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_path = f'tello_video_{now}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (1080, 720))

        while not self.should_stop.is_set():
            frame = self.tello.get_frame_read().frame
            if frame is not None:
                annotated_frame, smoke_detected, _, _ = self.detector.findsmoke(frame)
                
                # 控制LED和点阵屏显示
                if smoke_detected:
                    self.tello.send_expansion_command("led 255 0 0")  # 红色LED
                    self.tello.send_expansion_command("mled s r X")

                else:
                    self.tello.send_expansion_command("led 0 255 0")  # 绿色LED
                    self.tello.send_expansion_command("mled l b 0.5 ^_^")

                # 注意：此处未实现视频帧的实际写入逻辑，需根据需求决定是否启用
                out.write(annotated_frame)

        self.tello.streamoff()
        print("Video processing completed.")


def main():
    tello = Tello()
    tello.connect()
    print("Connected to Tello.")

    should_stop = threading.Event()
    video_thread = VideoRecorderThread(tello, should_stop)
    video_thread.start()

    try:
        while True:
            user_input = input("Press 'q' to quit: ")
            if user_input.lower() == 'q':
                should_stop.set()
                break
    except KeyboardInterrupt:
        should_stop.set()
        print("\nExiting...")

    video_thread.join()
    tello.end()


if __name__ == "__main__":
    main()
