import os
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
car = NvidiaRacecar()
from jetcam.csi_camera import CSICamera
import cv2
import time
import numpy as np
import warnings

from filter import calibration
from filter import sobel
from filter import img2world
from filter import yellow
from othercar_detect import car_detect

warnings.filterwarnings('ignore')

image_path = 'lane_keeping/'

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

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

camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

from ultralytics import YOLO
classes = YOLO("best_traffic_new.pt", task='detect').names
model = YOLO("best_traffic_new.pt", task='detect')
colors = np.random.randn(len(classes), 3)
colors = (colors * 255.0).astype(np.uint8)

class Line:
    def __init__(self):
        self.detected = False
        self.window_margin = 56
        self.prevx = []
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.startx = None
        self.endx = None
        self.allx = None
        self.ally = None
        self.road_inf = None
        self.curvature = None
        self.deviation = None

# Array to store data
data = []

# initialization
running = False
boost_exp = False
end = False
running_exp = False
image_number = 0
steering = 0
mode = 'normal'
left_start_time = 0
before_white_pixels = 0

# tuning
left_mode_time = 2
throttle = 0.292
boost = 0.0095
# 아주 작음 / 작음 / 최적 / 빠름 / 아주 빠름

# 0.335 0.011
# 0.335 0.01
# 0.338 0.008 약간 느림
# 0.363 0.008

# boost = 0.011 - (throttle-0.335)/7
 
# check joystick
while running == False:
    pygame.event.pump()
    if joystick.get_button(0): # race finish button
        running = True
    print(joystick.get_axis(1), joystick.get_axis(2), running)

# wait camera performance <= maybe camera needs time to receive light
start_time = time.time()
current_time = start_time
while current_time - start_time < 3:
    print(current_time - start_time)
    current_time = time.time()

last_time = 0
y_pos_before = 0
pred_count = 0

sign_detect = 'baegunhee'

while running:
    pygame.event.pump()
        
    frame = camera.read()
    
    pred_count += 1
    if model is not None and pred_count == 10:
    # if model is not None and pred_count == 1:
        pred_count = 0
        pred = model(frame, stream=True)
        results = draw_boxes(frame, pred, classes, colors)
        for cls_name, score, height in results:
            print(f"Class: {cls_name}, Score: {score}, Height: {height}")
            if cls_name == 'RED' and score > 0.5:
                sign_detect = 'red'
            elif cls_name == 'Green' and score > 0.5:
                sign_detect = 'green'
    image_filename = os.path.join(image_path, f"frame_{image_number:05d}.jpg")
    cv2.imwrite(image_filename, frame)
    
    frame_cal = calibration(frame, mode)
    image_filename = os.path.join(image_path, f"frame_cal{image_number:05d}.jpg")
    cv2.imwrite(image_filename, frame_cal)
    
    # hls_combine = sobel(frame_cal)
    hls_combine = yellow(frame_cal)
    image_filename = os.path.join(image_path, f"frame_filter{image_number:05d}.jpg")
    cv2.imwrite(image_filename, hls_combine)
    
    output, new_hls_combine = img2world(hls_combine)
    current_time = time.time()
    
    ## sliding_window
    left_line = Line()
    right_line = Line()

    cut_height = 170
    mid_x, mid_y = 320, 180

    window_height = 10
    num_windows = int((new_hls_combine.shape[0] - cut_height)/ window_height)
    
    nonzero = new_hls_combine.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    min_num_pixel = 50

    if mode == 'normal':
        detection, y_pos = car_detect(output, y_pos_before, current_time - last_time, image_number)
        
        y_pos_before = y_pos
        last_time = current_time
        
        current_leftX = 270
        current_rightX = 370
        win_left_lane = []
        win_right_lane = []

        for window in range(num_windows):
            win_y_low = new_hls_combine.shape[0] - (window + 1) * window_height
            win_y_high = new_hls_combine.shape[0] - window * window_height
            
            if window == 0:
                win_leftx_max = current_leftX + 40
                win_rightx_max = current_rightX + 30
                win_leftx_min = current_leftX - 30
                win_rightx_min = current_rightX - 30
            else:
                win_leftx_max = current_leftX + 25
                win_rightx_min = current_rightX - 15
                win_rightx_max = current_rightX + 15
                win_leftx_min = current_leftX - 15

            cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)
            
            left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
                nonzerox <= win_leftx_max)).nonzero()[0]
            right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
                nonzerox <= win_rightx_max)).nonzero()[0]
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)
            
            if len(left_window_inds) > min_num_pixel:
                current_leftX = int(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds) > min_num_pixel:
                current_rightX = int(np.mean(nonzerox[right_window_inds]))
            
            if window == 7 and end == False:
                window8_white_pixels = len(left_window_inds)

        win_left_lane = np.concatenate(win_left_lane)
        win_right_lane = np.concatenate(win_right_lane)

        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
        rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255]

        if leftx.size > 0 and lefty.size > 0:
            left_fit = np.polyfit(lefty, leftx, 3)
        else:
            left_fit = [0, 0, 0, 0]

        if rightx.size > 0 and righty.size > 0:
            right_fit = np.polyfit(righty, rightx, 3)
        else:
            right_fit = [0, 0, 0, 0]

        left_line.current_fit = left_fit
        right_line.current_fit = right_fit
        ploty = np.linspace(0, hls_combine.shape[0] - 1, hls_combine.shape[0])

        look_ahead_distance = 360 - 50
        
        offset_ahead_left = left_fit[0] * look_ahead_distance ** 3 + left_fit[1] * look_ahead_distance ** 2 + left_fit[2] * look_ahead_distance +  left_fit[3]
        offset_ahead = (left_fit[0]+right_fit[0])/2 * look_ahead_distance ** 3 + (left_fit[1]+right_fit[1])/2 * look_ahead_distance ** 2 + (left_fit[2]+right_fit[2])/2 * look_ahead_distance +  (left_fit[3]+right_fit[3])/2
        
        left_plotx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
        right_plotx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
        center_plotx = (left_plotx + right_plotx) / 2

        for i in range(len(ploty)):
            if 0 <= int(left_plotx[i]) < output.shape[1] and 0 <= int(ploty[i]) < output.shape[0]:
                output[int(ploty[i]), int(left_plotx[i])] = [255, 0, 0]
            if 0 <= int(right_plotx[i]) < output.shape[1] and 0 <= int(ploty[i]) < output.shape[0]:
                output[int(ploty[i]), int(right_plotx[i])] = [0, 0, 255]
            if 0 <= int(center_plotx[i]) < output.shape[1] and 0 <= int(ploty[i]) < output.shape[0]:
                output[int(ploty[i]), int(center_plotx[i])] = [0, 255, 0]
                
        offset_ahead_x = int(offset_ahead)
        offset_ahead_y = look_ahead_distance
        if 0 <= offset_ahead_x < output.shape[1] and 0 <= offset_ahead_y < output.shape[0]:
            cv2.circle(output, (offset_ahead_x, offset_ahead_y), 10, (0, 0, 255), -1) # Draw red circle

        # Draw the mid_x line
        cv2.line(output, (mid_x, 0), (mid_x, output.shape[0]), (255, 0, 0), 2) # Draw blue line

        error = mid_x - offset_ahead
        # print(offset_ahead-offset_ahead_left)
        
        if window8_white_pixels > 85:
            print('left_mode_checking')
            while sign_detect == 'red':
                car.throttle = 0
                frame = camera.read()
                pred = model(frame, stream=True)
                results = draw_boxes(frame, pred, classes, colors)
                for cls_name, score, height in results:
                    print(f"Class: {cls_name}, Score: {score}, Height: {height}")
                    if cls_name == 'RED':
                        sign_detect = 'red'
                    elif cls_name == 'Green':
                        sign_detect = 'green'
            print('start go, detection:', sign_detect)
            left_image_number = image_number
            while detection == 'stop':
                left_image_number = left_image_number+1
                car.throttle = 0
                frame = camera.read()
                image_filename = os.path.join(image_path, f"frame_{image_number:05d}.jpg")
                cv2.imwrite(image_filename, frame)
                frame_cal = calibration(frame, mode)
                # save image
                image_filename = os.path.join(image_path, f"frame_cal{image_number:05d}.jpg")
                cv2.imwrite(image_filename, frame_cal)
                hls_combine = sobel(frame_cal)
                image_filename = os.path.join(image_path, f"frame_filter{image_number:05d}.jpg")
                cv2.imwrite(image_filename, hls_combine)
                output, new_hls_combine = img2world(hls_combine)
                current_time = time.time()
                detection, y_pos = car_detect(output, y_pos_before, current_time - last_time, left_image_number)
                y_pos_before = y_pos
                last_time = current_time
                
                
            print("left_mode start")
            car.throttle = 0
            left_start_time = current_time
            end = True
            mode = 'left'
        
        # if window8_white_pixels> 85:
        #     print("left_mode start")
        #     left_start_time = current_time
        #     end = True
        #     mode = 'left'
    
    elif mode == 'left':
        current_leftX = 270
        win_left_lane = []
        
        for window in range(int(num_windows/2)):
            win_y_low = new_hls_combine.shape[0] - (window + 1) * window_height
            win_y_high = new_hls_combine.shape[0] - window * window_height
            win_leftx_min = current_leftX - 40
            win_leftx_max = current_leftX + 5

            cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
            
            left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
                nonzerox <= win_leftx_max)).nonzero()[0]

            win_left_lane.append(left_window_inds)
    
            if len(left_window_inds) > min_num_pixel:
                current_leftX = int(np.mean(nonzerox[left_window_inds]))
 

        win_left_lane = np.concatenate(win_left_lane)

        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]

        output[lefty, leftx] = [255, 0, 0]

        if leftx.size > 0 and lefty.size > 0:
            left_fit = np.polyfit(lefty, leftx, 3)
        else:
            left_fit = [0, 0, 0, 0]

        left_line.current_fit = left_fit
        ploty = np.linspace(0, hls_combine.shape[0] - 1, hls_combine.shape[0])

        look_ahead_distance = 360 - 40
        
        offset_ahead = left_fit[0] * look_ahead_distance ** 3 + left_fit[1] * look_ahead_distance ** 2 + left_fit[2] * look_ahead_distance +  left_fit[3]
        
        left_plotx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
        center_plotx = (left_plotx + right_plotx) / 2

        for i in range(len(ploty)):
            if 0 <= int(left_plotx[i]) < output.shape[1] and 0 <= int(ploty[i]) < output.shape[0]:
                output[int(ploty[i]), int(left_plotx[i])] = [255, 0, 0]
                
        offset_ahead_x = int(offset_ahead) + 45
        offset_ahead_y = look_ahead_distance
        if 0 <= offset_ahead_x < output.shape[1] and 0 <= offset_ahead_y < output.shape[0]:
            cv2.circle(output, (offset_ahead_x, offset_ahead_y), 10, (0, 0, 255), -1) # Draw red circle

        # Draw the mid_x line
        cv2.line(output, (mid_x, 0), (mid_x, output.shape[0]), (255, 0, 0), 2) # Draw blue line
            
        error = mid_x - offset_ahead - 45
        if current_time - left_start_time > left_mode_time:
            # error = mid_x - offset_ahead - 100
            end = True
            mode = 'normal'
            print('normal')
    
    # 러프하게 P 제어
    if mode == 'normal':
        car.steering = -error*0.02
    elif mode == 'left':
        car.steering = -error*0.015
    
    if abs(before_white_pixels - window8_white_pixels) < 15 and boost_exp == False and running_exp == True:
        print("현 픽셀 개수:", window8_white_pixels, "전 픽셀 개수", before_white_pixels)
        throttle = throttle + 0.0005

    elif abs(before_white_pixels - window8_white_pixels) > 15 and boost_exp == False and running_exp == True:
        boost_exp = True 
        # boost = 0.014 + (0.335-throttle)/3
        throttle = throttle + boost
    
    before_white_pixels = window8_white_pixels
    
    if car.throttle > 0.45:
        running = False
    
    car.throttle = float("{:.4f}".format(throttle))
    
    # car.throttle = 0
    # car.steering = 0
    # cv2.imshow('Lane Detection', output)
    
    image_filename = os.path.join(image_path, f"frame_world{image_number:05d}.jpg")
    cv2.imwrite(image_filename, output)
    
    data.append((image_number, window8_white_pixels))
    
    # print(image_number, "steering : ",car.steering, "error : ",error, offset_ahead-offset_ahead_left, stopping)
    print(image_number, "steering : ",car.steering, "throttle : ", car.throttle, window8_white_pixels)
    
    image_number += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if joystick.get_button(11): # race finish button
        # 바로 정지하면 에러 생겨서 0으로 설정 후, 잠시 대기
        car.steering = 0
        car.throttle = 0
        time.sleep(1)
        camera.release()
        print("release nice")
        # 저장된 데이터를 파일로 저장
        with open("save_data.txt", "w") as f:
            for image_number, window9_white_pixels in data:
                f.write(f"{image_number}, 9번째 픽셀 개수 : {window9_white_pixels}, 8번째 픽셀 개수 : {window8_white_pixels}\n")
        running = False
    
    if joystick.get_button(7):
        stopping = False

    if joystick.get_button(6):
        stopping = True

    if joystick.get_button(10):
        if stopping:
            camera.release()
            print("release nice")
            
    if running_exp == False:
        running_exp = True

cv2.destroyAllWindows()

