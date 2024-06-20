import threading
import cv2  # 假设你使用OpenCV处理图像
import torch
from ultralytics import YOLO
from djitellopy import Tello
import math
import keyboard
import time

def distance(dx,dy):
    dis = math.sqrt((dx)**2 + (dy)**2)
    return int(dis)
class YOLODetector:
    def __init__(self, model_path='best_smoke.pt'):
        """
        初始化YOLOv8检测器。
        参数:
        - model_path (str)
        """
        # 加载YOLOv8模型
        self.model = YOLO(model_path)
    def findsmoke(self, img):
        """
        在图像中查找烟雾
        :param img: 输入图像（假设为OpenCV格式）
        :return: 处理后的图像以及检测到的位置信息等
        """
        # 这里只是一个示例框架，需要根据实际模型API调整
        results = self.model(img)
        height, width = img.shape[:2]
        image_center = (int(width / 2), int(height / 2))
        output_image = results[0].plot()
        x1=0
        x2=0
        y1=0
        y2=0
        # center_diff_y=0
        # center_diff_x=0
        detected = -1
        detections = results[0].boxes  # 获取检测结果，具体属性取决于模型输出
        if detections.shape[0] != 0:  # 如果检测到对象
            print("检测到任何对象")
            # detected = False
            detected = 1
        if results[0]:
            x1, y1, x2, y2 = results[0].boxes.cpu().xyxy.numpy()[0]
            # print(x1, y1, x2, y2)
        mid_x=int((x1+x2)/2)
        mid_y=int((y1+y2)/2)
        if(mid_x+mid_y>15):
            # center_diff_x = mid_x - image_center[0]
            # center_diff_y = mid_y - image_center[1]
            # dis = distance(center_diff_x, center_diff_y)
            radius = 15
            color = (0, 255, 0)
            # output_image,dis=draw_point(output_image,int(mid_x),int(mid_y),image_center[0],image_center[1])
            cv2.circle(output_image, (mid_x, mid_y), radius, color, -1)
            # cv2.circle(output_image, image_center, radius, color, -1)
        return output_image, detected, mid_x, mid_y

if __name__ == "__main__":
    model_path = 'best_smoke.pt'
    # 初始化YOLO检测器
    detector = YOLODetector(model_path)
    tello = Tello()
    tello.connect()
    tello.streamon()
    frame_read = tello.get_frame_read()
    while frame_read is None:
        frame_read = tello.get_frame_read()
    time.sleep(5)
    # 检测图像中的对象
    width, height = 960, 720
    video = cv2.VideoWriter('video_yolo.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    while True:
        image= frame_read.frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image, detected, mid_x, mid_y = detector.findsmoke(image)
        # 打印检测到的对象信息
        if detected == 1:
            cv2.putText(output_image, f"centerx:{mid_x:.2f},centery:{mid_y:.2f}",(10, 30),    cv2.FONT_HERSHEY_SIMPLEX,   1, (255, 255, 255),2, cv2.LINE_AA)
        video.write(output_image)
        cv2.imshow("Frame", output_image)
        cv2.resizeWindow("Frame",960, 720)
        cv2.waitKey(1)
        if keyboard.is_pressed('esc'):
            # detectmode=0
            break
    tello.streamoff()
    cv2.destroyAllWindows()
    