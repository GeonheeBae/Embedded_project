from __future__ import annotations
from pathlib import Path
from typing import Sequence

# 0.355/0.335 해야할듯
# top_speed = 0.36
# slow_speed = 0.34
# top_speed = 0.355
# slow_speed = 0.335
plus = -0.001
top_speed = 0.345 + plus
slow_speed = 0.33 + plus


import argparse
import cv2
import datetime
import glob
import logging
import numpy as np
import os
import time
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.utils import bgr8_to_jpeg
import PIL.Image
from postprocess import *

error_prev = 0; error_accum = 0; error_diff = 0
height = 360; widt = 640

logging.getLogger().setLevel(logging.INFO)

def draw_boxes(image, pred, classes, colors):
    results = [] 
    for r in pred:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = round(float(box.conf[0]), 2)
            label = int(box.cls[0])
            color = colors[label].tolist()
            cls_name = classes[label]
            
            height = abs(y2 - y1)
            results.append((cls_name, score, height))

            print(x1, y1, x2, y2) # 각 바운딩박스 좌표
            print(abs(x2-x1), abs(y2-y1)) #각 바운딩박스 폭 / 높이
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{cls_name} {score}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return results

class Camera:
    def release(self):
        for cap in self.cap:
            cap.release()
            
    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        _width: int = 640,
        _height: int = 360,
        frame_rate: int = 25, #30,
        flip_method: int = 0,
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = False,
        save: bool = False,
        log: bool = True,
    ) -> None:
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self._width = _width
        self._height = _height
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log
        self.model = None

        # Check if OpenCV is built with GStreamer support
        # print(cv2.getBuildInformation())

        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence) and len(sensor_id) > 1:
            raise NotImplementedError("Multiple cameras are not supported yet")

        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        self.cap = [cv2.VideoCapture(self.gstreamer_pipeline(sensor_id = id), \
        				cv2.CAP_GSTREAMER) for id in self.sensor_id]

        # Make record directory
        if save:
            assert save_path is not None, "Please provide a save path"
            os.makedirs(self.save_path, exist_ok=True) # if path does not exist, create it
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)

            logging.info(f"Save directory: {self.save_path}")

    def gstreamer_pipeline(self, sensor_id: int) -> str:
        """
        Return a GStreamer pipeline for capturing from the CSI camera
        """
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                sensor_id,
                self.width,
                self.height,
                self.frame_rate,
                self.flip_method,
                self._width,
                self._height,
            )
        )

    def set_model(self, model: YOLO, classes: dict) -> None:
        """
        Set a YOLO model
        """
        self.model = model
        self.classes = classes                
        self.colors = np.random.randn(len(self.classes), 3)
        self.colors = (self.colors * 255.0).astype(np.uint8)
        self.visualize_pred_fn = lambda img, pred: draw_boxes(img, pred, self.classes, self.colors)

    @property
    def frame(self) -> np.ndarray:
        """
        !!! Important: This method is not efficient for real-time rendering !!!

        [Example Usage]
        ...
        frame = cam.frame # Get the current frame from camera
        cv2.imshow('Camera', frame)
        ...

        """
        if self.cap[0].isOpened():
            return self.cap[0].read()[1]
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

cam = Camera()

# if args.yolo_model_file is not None:
from ultralytics import YOLO
# HACK: TensorRT YOLO model doesn't have classes info.
#classes = YOLO("yolov8n_traffic.engine", task='detect').names
classes = YOLO("yolov8n_traffic.engine", task='detect').names
model = YOLO("yolov8n_traffic.engine", task='detect')
cam.set_model(model, classes)
        
# 모델 로딩
from cnn.center_dataset import TEST_TRANSFORMS
def preprocess(image: PIL.Image):
    device = torch.device('cuda')    
    image = TEST_TRANSFORMS(image).to(device)
    return image[None, ...]

import torch
import torchvision
def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model

device = torch.device('cuda')

# normal
model1 = get_model()
model1.load_state_dict(torch.load('road_following_model_normal_1F.pth'))
model1 = model1.to(device)
model_normal = model1
# right
model2 = get_model()
model2.load_state_dict(torch.load('road_following_model_right_1F.pth'))
model2 = model2.to(device)
model_right = model2
# straight
model3 = get_model()
model3.load_state_dict(torch.load('road_following_model_straight_1F.pth'))
model3 = model3.to(device)
model_straight = model3

car = NvidiaRacecar()

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

running = False

# save image data as a file
def save_image(image_data, filename):
    # using "wb" when processing image
    # using with/as -> automatically closing file
    with open(filename, "wb") as file:
        file.write(image_data)

stopping = True

# 준비함수
while running == False:
    pygame.event.pump()
    if joystick.get_button(0): # race finish button
        running = True
    # print(running, stopping)
    if joystick.get_button(7): # video start button
        stopping = False

# 초기 모델 설정
mode_name = "normal_tracking"

last_mode_time = 0

# 횡단보도 정지 시간
stop_mode_time = 2

stop_start_time = 0

# 횡단보도 task 수행 후, 추가적으로 횡단보도 인지하지 않는 시간 
stop_add_time = 10

# 횡단보도 task 경험 유무, 횡단보도 인지하지 않는 시간 (10s)가 지날 시 0으로 초기화
stop_experience = 0

last_stop_time = 0

# 횡단보도 정지에서 출발할 때 boost하는 시간
power_mode_time = 0.3

# 버스에서 느리게 가는 시간
slow_mode_time = 2

# 교차로 모드 적용 시간
stay_inter_time = 10

pred_count = 0
prev_mode = ""

start_time = time.time()

steering = 0
# image_number = 0

last_throttle_time = 0
while running:
    pygame.event.pump()
    current_time = time.time()
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    _, frame = cam.cap[0].read()
    
    jpeg_data = bgr8_to_jpeg(frame)
    
    # 주행 중의 이미지 저장
    # image_number = int(image_number)
    # save_image(jpeg_data, f'dataset/images/frame_1F_{image_number:09d}.jpg')    
    # image_number +=1
    pred_count +=1
    
    # delay방지를 위해 이미지 capture 10번 시, 1번만 표지판 classification
    if cam.model is not None and pred_count == 10:
        pred_count = 0
        pred = cam.model(frame, stream=True)
        results = cam.visualize_pred_fn(frame, pred)
        for cls_name, score, height in results:
            # print(f"Class: {cls_name}, Score: {score}, height: {height}")
            if score >= 0.7 and height >= 60:
                # 버스 task == slow 모드 / 직전 모드가 slow모드가 아닐 시, 버스 task 진입
                if cls_name == "bus" and mode_name != "slow":
                    mode_name = "slow"
                    last_mode_time = current_time
            if score >= 0.45 and height >= 70:
                # crosswalk task == stop mode / 직전 모드가 stop mode 가 아니고, stop 경험이 최근에 없을 시
                if cls_name == "crosswalk" and mode_name != "stop" and stop_experience != 1:
                    stop_experience = 1
                    mode_name = "stop" 
                    last_mode_time = current_time
                    stop_start_time = current_time
                    # throttle이 돌아가며 속도가 줄어드는 것을 보정하기 위해 1바퀴 돌때마다 throttle default값 증가
                    top_speed = top_speed + 0.0075
                    slow_speed = slow_speed + 0.0075
                    
            if height >= 40 and score >= 0.55:
                # right task == right_tracking 모드 / 직전 모드가 right_tracking 모드가 아닐 시, right task 진입
                if cls_name == "right" and mode_name != "right_tracking":
                    mode_name = "right_tracking"            
                    last_mode_time = current_time
            if height >= 40 and score >= 0.8:
                # straight task == straight_tracking 모드 / 직전 모드가 straight_tracking 모드가 아닐 시, straight task 진입
                if cls_name == "straight" and mode_name != "straight_tracking":
                    mode_name = "straight_tracking"          
                    last_mode_time = current_time
            if height >= 40 and score >= 0.6:
                # left task == left_tracking 모드 / 직전 모드가 left_tracking 모드가 아닐 시, left task 진입
                if cls_name == "left" and mode_name != "left_tracking":
                    mode_name = "left_tracking"           
                    last_mode_time = current_time
                

    image_ori = PIL.Image.fromarray(frame)
    
    # 횡단보도 인지하지 않는 시간 (10s)가 지날 시 0으로 초기화
    if current_time - stop_start_time >= stop_mode_time + stop_add_time:
        stop_experience = 0
    
    # 공통적으로 if current_time - last_mode_time >= slow_mode_time: 문을 통해 각 task의 시간이 지나면 모드 해제
    # slow 모드일 시, throttle 감소 & normal_tracking 모드로 동작
    if mode_name == "slow":
        car.throttle = slow_speed - 0.015
        with torch.no_grad():
            image = preprocess(image_ori)
            output = model_normal(image).detach().cpu().numpy()
        if current_time - last_mode_time >= slow_mode_time:
            mode_name = "normal_tracking"
    # stop 모드일 시, throttle 0 & steering 변화 X
        car.throttle = 0
        with torch.no_grad():
            image = preprocess(image_ori)
            output = model_normal(image).detach().cpu().numpy()
        if current_time - last_mode_time >= stop_mode_time:
            prev_mode = "stop"
            mode_name = "normal_tracking"
            last_stop_time = current_time
    # right_tracking 모드일 시, right_tracking 모드로 동작
    elif mode_name == "right_tracking":
        with torch.no_grad():
            image = preprocess(image_ori)
            output = model_right(image).detach().cpu().numpy()
        if current_time - last_mode_time >= stay_inter_time:
            mode_name = "normal_tracking"
    # straight_tracking 모드일 시, straight_tracking 모드로 동작
    elif mode_name == "straight_tracking":
        with torch.no_grad():
            image = preprocess(image_ori)
            output = model_straight(image).detach().cpu().numpy()
        if current_time - last_mode_time >= stay_inter_time:
            mode_name = "normal_tracking"
    # left_tracking 모드일 시, normal_tracking 모드로 동작
    elif mode_name == "left_tracking":
        with torch.no_grad():
            image = preprocess(image_ori)
            output = model_normal(image).detach().cpu().numpy()
        if current_time - last_mode_time >= 3:
            mode_name = "normal_tracking"
    # 나머지 모드일 시, normal_tracking 모드로 동작
    else:
        with torch.no_grad():
            image = preprocess(image_ori)
            output = model_normal(image).detach().cpu().numpy()
            
    
    # control 
    # stop이 아닐시 각 모델로 구한 에러를 통해 PID 제어
    if mode_name != "stop":
        x, y = output[0]
        [steering, error_accum, error_prev, error_diff] = steering_input(x, error_prev, error_accum, Ts)
        car.steering = steering
    
    # left 일 시, 모서리 부딪힘 방지를 위해 gain 감소
    if mode_name == "left_tracking":
        car.steering = 0.7*steering
    
    if mode_name != "stop" and mode_name != "slow":
        # car.steering 0 -> topspeed, abs(car.steering) 1 ->  slow speed  
        # car.steering에 따른 비선형 속도 제어
        car.throttle = top_speed + (slow_speed - top_speed - 0.02)*abs(car.steering)
        if car.throttle <= slow_speed:
            car.throttle = slow_speed

    # stop모드에서 시작할 시, boost 하는 throttle 증가량
    if prev_mode == "stop" and current_time - last_stop_time <= power_mode_time:
        car.throttle = top_speed + 0.02
    
    # print(f"mode: {mode_name}", f"throttle: {car.throttle}", f"Ts: {Ts}", f"stop: {stopping}")

    if joystick.get_button(7): # video start button
        stopping = False

    if joystick.get_button(6): # video finish button
        stopping = True

    if joystick.get_button(10): # camera release
        if stopping == True:
            cam.release()
            print("release nice")
    
    # 버튼이 눌릴 시, 한번만 작동하게 하기 위해 0.3초당 한번만 동작하도록 설정
    if joystick.get_button(3) and current_time - last_throttle_time >= 0.3: # increase velocity
        last_throttle_time = current_time
        top_speed = top_speed + 0.002
        slow_speed = slow_speed + 0.002
    if joystick.get_button(1) and current_time - last_throttle_time >= 0.3: # increase velocity
        last_throttle_time = current_time
        top_speed = top_speed - 0.002
        slow_speed = slow_speed - 0.002
            
    x, y = output[0]
    # print(x, steering, car.steering)

    if joystick.get_button(11): # race finish button
        # 바로 정지하면 에러 생겨서 0으로 설정 후, 잠시 대기
        car.steering = 0
        car.throttle = 0
        time.sleep(1)
        cam.release()
        print("release nice")
        running = False
            
