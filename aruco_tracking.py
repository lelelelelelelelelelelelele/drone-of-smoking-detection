import cv2
from djitellopy import Tello
import time
from pid_controller import PIDctl # 假设这是一个自定义的PID控制器类
from ArucoDetector import Aruco_detector # 假设这是处理Aruco码的类
# from yolo_detection import YOLODetector # 假设这是调用预训练YOLO模型的类
import keyboard
from threading import Thread
from ultralytics import YOLO
import torch
from PIL import Image
import sys
import os
import numpy as np
import math
import re
def distance(dx,dy):
    dis = math.sqrt((dx)**2 + (dy)**2)
    return int(dis)
def fly_to(height):
    current_height = tello.get_distance_tof()
    dh = height - current_height
    if dh > 20:
        tello.move_up(int(dh))
    elif dh < -20:
        tello.move_down(int(dh))
    time.sleep(1)
def get_front_dis():
    response = tello.send_read_command('EXT tof?')
    ret = -1
    if response is not None:
        tof = response.strip()
        dis = re.findall(r'\d+', tof)
        if dis:
            ret = int(dis[0])
            print("距离：", ret)
        else:
            print("未读取到距离.")
    return ret

def videoSW():
    global ddsX, ddsY, detected
    # global number
    global photo
    photo_cnt = 0
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    frame_read = tello.get_frame_read()
    while VideoS:
        img = frame_read.frame
        start_time = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 根据detectmode选择不同的检测逻辑
        #mode 
        '''Aruco码识别模式：
从摄像头捕获的图像中识别Aruco码，确定无人机相对于码的位置和方向，以便进行精确的PID控制跟随或悬停。
障碍物检测模式：

在飞行过程中，实时分析图像以检测前方是否有障碍物，这可能涉及背景减除、边缘检测、形状分析或深度学习模型，以识别潜在的碰撞风险并触发避障逻辑。
抽烟行为识别模式：

使用YOLO算法对图像进行实时处理，识别是否有抽烟行为。这需要加载预训练的YOLO模型，对图像进行前向传播，分析输出，识别出吸烟者的特征或特定的烟雾形态。
图像记录模式：

当检测到特定事件（如抽烟）时，触发相机进行拍照，记录关键图像或视频证据。
状态反馈模式：'''
        if detectmode == 0 and img.size != 0: #丢失目标
            img, detected, ddsX, ddsY = aruco_detector.detect_markers(img, number)
        elif detectmode == 1 and img.size != 0: #巡线
            img, detected, ddsX, ddsY = aruco_detector.detect_markers(img, number)
        # elif detectmode == 2 or detectmode == 3 and img.size != 0: # 巡视点（未检测到)/检测到
        #     img, detected, ddsX, ddsY = yolo_detector.findsmoke(img)
        time.sleep(0.1)
        if photo:
            # Save image based on detectmode
            filename = f"picture{photo_cnt}.png"
            cv2.imwrite(filename, img)
            photo = 0
            photo_cnt += 1
        '''video.write(cv2.resize(img, (320, 240)))#*'''
        cv2.circle(img, (standard_x, standard_y), 4, (0, 255, 255), -1)
        if detectmode == 0:
            cv2.putText(img,    f"find code...,number: {number:d}",    (10, 30),    cv2.FONT_HERSHEY_SIMPLEX,   1, (255, 255, 255),2, cv2.LINE_AA)
        elif detectmode == 1:
            cv2.putText(img,    f"track code...,number: {number:d}",    (10, 30),    cv2.FONT_HERSHEY_SIMPLEX,   1, (255, 255, 255),2, cv2.LINE_AA)
        if detectmode == 2:
            cv2.putText(img,    f"search smoker...",    (10, 30),    cv2.FONT_HERSHEY_SIMPLEX,   1, (255, 255, 255),2, cv2.LINE_AA)

        cv2.imshow("Frame", img)
        cv2.resizeWindow("Frame", width, height)
        cv2.waitKey(1)
        # if cti > 10 and detectmode != 0 and dds != 0:
        #     cti = 0
        #     print("distance:", dds, " ", ddsX, " ", ddsY)
        # else:
        #     cti += 1

    cv2.destroyAllWindows()
    video.release()

# def main():
# 引入相机内参矩阵
intrinsic_camera = np.load('./cam_intri_for.npy')
# 引入相机畸变系数
distortion = np.load('./coff_dis_for.npy')
# model_path = 'best_smoke.pt'
# 初始化无人机
tello = Tello()
tello.connect()

tello.streamoff()

tello.streamon()
tello.enable_mission_pads
tello.set_mission_pad_detection_direction(0)
# 初始化全局变量
dds, ddsX, ddsY = 0, 0, 0  # 初始化距离和其他变量
width, height = 960, 720
number = 0 # 在aruco追踪时标记当前目标点
''''''
VideoS = True
photo = False # on every supervision point for 1
detectmode = 1
detected = -1
standard_x,standard_y=int(width/2), int(height/7*4)
videoLine=Thread(target=videoSW)
videoLine.start()
print('TT_battery:', tello.get_battery())
# 初始化各个模块
pidx=PIDctl(-0.3,-0.1,0,0.21)
pidy=PIDctl(-0.3,0,0,0.21)
aruco_detector = Aruco_detector(aruco_dict_type= cv2.aruco.DICT_7X7_1000, intrinsic_camera= intrinsic_camera,distortion= distortion)

# 起飞
print("Taking off...")
tello.takeoff()
time.sleep(1)
fly_to(120)
# 无人机循环执行任务
while True:
    if get_front_dis() < 100:
        tello.send_rc_control(0, 0, 0, 0)
        print(tello.get_distance_tof())
        time.sleep(0.1)
        continue
    if detectmode == 1: #xunxian
        if keyboard.is_pressed('esc'):
            break
        if detected == -1: #避障或丢失
            detectmode = 0
            tello.send_rc_control(0, 0, 0, 0)
            pidx.reset_integral()
            pidy.reset_integral()
            start_time = time.time()
            time.sleep(0.5)
        elif detected != number and find_another == False:
            find_another = True#atthemoment
            detectmode = 0
            start_time = time.time()
        else:
            dx = ddsX - standard_x
            dy = ddsY - standard_y
            dds = distance(dx, dy)
            print("dds=", dds, 'x=', dx, 'y=', dy)
            if dds< 60:
                number = number + 1 # new target
                print('****')
                if number == 4:
                    break
                detectmode = 2#find smoker
                detected = -1
                tello.send_rc_control(0, 0, 0, 0)
                pidx.reset_integral()
                pidy.reset_integral()
                smoker_cnt = 0
                time.sleep(1)
                start_time = time.time()
                photo = True
            else:
                upgrade_yaw = round(pidx.update(-dx))
                if upgrade_yaw < 0:
                    upgrade_yaw = max (upgrade_yaw, -40)
                else:
                    upgrade_yaw = min(upgrade_yaw, 40)
                upgrade_forback = round(pidy.update(dy))
                print('v_f = ', upgrade_forback)
                tello.send_rc_control(0, min(30,upgrade_forback), 0, upgrade_yaw)
        time.sleep(0.2)
    else: #find
        tello.send_rc_control(0, 0, 0, 20)
        print(detectmode)
        if detected == number :
            detectmode = 1
            print("find target: ",number)
        time.sleep(0.2)
    if keyboard.is_pressed('esc'):
        detectmode=0
        break
    # if aruco_detected:
    #     print("Aruco detected, adjusting position...")
    #     # 使用PID调整无人机至目标位置
    #     pid.update(target_position)
    #     tello.send_rc_control(int(pid.output_x), int(pid.output_y), 0, 0)
    # else:
    #     print("Lost target, searching...")
    # time.sleep(0.1)
# 任务结束，降落
VideoS = False
tello.land()
tello.streamoff()

# def simple_obstacle_avoidance(frame):
#     # 这里仅作为示例，实际避障逻辑需基于深度学习或更复杂的图像处理
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, threshold = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # 假设如果找到轮廓则表示有障碍物
#     if len(contours) > 0:
#         return True, frame
#     else:
#         return False, frame


# if __name__ == "__main__":
#     main()