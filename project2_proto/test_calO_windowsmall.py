import os
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
car = NvidiaRacecar()
from jetcam.csi_camera import CSICamera
import cv2
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

image_path = 'lane_keeping/'

with np.load('camera_calibration_params2.npz') as data:
    mtx = data['mtx']
    dist = data['dist']
    newcameramtx = data['new']

th_h, th_l, th_s = (160, 255), (50, 160), (0, 255)
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)

# 차선 테두리만 검출
# th_h, th_l, th_s = (20, 100), (120, 255), (100, 255)  # 변경됨
# th_sobelx, th_sobely, th_mag, th_dir = (20, 100), (20, 100), (30, 100), (0.7, 1.3)  # 변경됨

# 가까운 부분 검출 X
# th_h, th_l, th_s = (0, 180), (200, 255), (90, 255)  # 변경됨
# th_sobelx, th_sobely, th_mag, th_dir = (10, 200), (10, 200), (20, 200), (0.7, 1.3)  # 변경됨

# 가까운 부분 검출 X
# th_h, th_l, th_s = (0, 255), (200, 255), (90, 255)  # 변경됨
# th_sobelx, th_sobely, th_mag, th_dir = (10, 255), (10, 255), (30, 255), (0.7, 1.3)  # 변경됨

# th_h, th_l, th_s = (0, 180), (200, 255), (100, 255)  # 변경됨
# th_sobelx, th_sobely, th_mag, th_dir = (10, 200), (10, 200), (20, 200), (0.7, 1.3)  # 변경됨

def sobel_xy(img, orient, thresh):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output

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

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

running = False

camera = CSICamera(capture_width=1280, capture_height=720, downsample=2, capture_fps=30)

stopping = True
image_number = 0
steering = 0
throttle = 0.3

# Array to store data
data = []

mode = 'normal'
end = False
left_start_time = 0
before_white_pixels = 0
boost_exp = False
boost = 0.015
# 아주 작음 / 작음 / 최적 / 빠름 / 아주 빠름

# 0.318 : 0.018 ㅈㄴ최적
# 0.338 : 0.017 작음
# 0.324 : 0.015 최적 

left_mode_time = 2

while running == False:
    pygame.event.pump()
    if joystick.get_button(0): # race finish button
        running = True
    print(joystick.get_axis(1), joystick.get_axis(2), running)

start_time = time.time()
current_time = start_time
while current_time - start_time < 6:
    print(current_time - start_time)
    current_time = time.time()
    
running_exp = False
while running:
    pygame.event.pump()
    current_time = time.time()
        
    frame = camera.read()
    last_save_time = current_time

    left_line = Line()
    right_line = Line()

    height, width = frame.shape[:2]
    cut_height = 170
    
    if mode == 'normal':
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
        undistorted_img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        vertices = np.array([[(0, cut_height), (0, height), (620, height), (400, cut_height)]], dtype=np.int32)
        mask = np.zeros_like(undistorted_img)
        #mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        frame = cv2.bitwise_and(undistorted_img, mask)
        #frame = cv2.bitwise_and(frame, mask)

        
    else:
        vertices = np.array([[(0, cut_height), (0, height), (620, height), (400, cut_height)]], dtype=np.int32)
        #mask = np.zeros_like(undistorted_img)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        #frame = cv2.bitwise_and(undistorted_img, mask)
        frame = cv2.bitwise_and(frame, mask)
        
    image_filename = os.path.join(image_path, f"frame_{image_number:05d}.jpg")
    cv2.imwrite(image_filename, frame)

    cut_high = 0
    img = frame[cut_high:height, :width, 2]
    abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_x = np.zeros_like(scaled_sobel)
    sobel_x[(scaled_sobel >= th_sobelx[0]) & (scaled_sobel <= th_sobelx[1])] = 255

    abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sobel_y = np.zeros_like(scaled_sobel)
    sobel_y[(scaled_sobel >= th_sobely[0]) & (scaled_sobel <= th_sobely[1])] = 255

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    gradient_magnitude = np.zeros_like(gradmag)
    gradient_magnitude[(gradmag >= th_mag[0]) & (gradmag <= th_mag[1])] = 255

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=15)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=15)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    gradient_direction = np.zeros_like(absgraddir)
    gradient_direction[(absgraddir >= th_dir[0]) & (absgraddir <= th_dir[1])] = 255
    gradient_direction = gradient_direction.astype(np.uint8)

    grad_combine = np.zeros_like(gradient_direction).astype(np.uint8)
    grad_combine[((sobel_x > 1) & (gradient_magnitude > 1) & (gradient_direction > 1)) | ((sobel_x > 1) & (sobel_y > 1))] = 255

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    H = hls[cut_high:height, 0:width, 0]
    L = hls[cut_high:height, 0:width, 1]
    S = hls[cut_high:height, 0:width, 2]

    h_img = np.zeros_like(H)
    h_img[(H > th_h[0]) & (H <= th_h[1])] = 255

    l_img = np.zeros_like(L)
    l_img[(L > th_l[0]) & (L <= th_l[1])] = 255

    s_img = np.zeros_like(S)
    s_img[(S > th_s[0]) & (S <= th_s[1])] = 255

    hls_combine = np.zeros_like(s_img).astype(np.uint8)
    hls_combine[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255 
    
    image_filename = os.path.join(image_path, f"frame_filter{image_number:05d}.jpg")
    cv2.imwrite(image_filename, hls_combine)

    white_pixel_coords = np.argwhere(hls_combine == 255)
    new_hls_combine = np.zeros_like(hls_combine)

    mid_x = 320
    mid_y = 180
    b = mid_y - 120 # 소실점

    k_y = 380
    k_x = 1/380

    cut_y = cut_height
    
    if white_pixel_coords.size > 0:
        y = white_pixel_coords[:, 0]
        x = white_pixel_coords[:, 1]
        origin_y = y.copy()
        y = mid_y - y
        x = x - mid_x

        mask1 = origin_y > cut_y
        mask2 = ~mask1

        y1 = y[mask1]
        x1 = x[mask1]

        if y1.size > 0:
            new_y1 = k_y * (y1) / (b - y1)
            new_x1 = k_x * x1 * (k_y + new_y1)
            new_x1 = new_x1 + mid_x
            new_y1 = mid_y - new_y1 - 104

            new_y1 = new_y1.astype(np.int32)
            new_x1 = new_x1.astype(np.int32)

            valid_mask1 = (0 <= new_x1) & (new_x1 < new_hls_combine.shape[1]) & (0 <= new_y1) & (new_y1 < new_hls_combine.shape[0])
            new_hls_combine[new_y1[valid_mask1], new_x1[valid_mask1]] = 255

        x2 = x[mask2]
        y2 = y[mask2] + 1000

        valid_mask2 = (0 <= x2) & (x2 < new_hls_combine.shape[1]) & (0 <= y2) & (y2 < new_hls_combine.shape[0])
        new_hls_combine[y2[valid_mask2], x2[valid_mask2]] = 255

    output = np.dstack((new_hls_combine, new_hls_combine, new_hls_combine)) * 10

    histogram = np.sum(new_hls_combine[int(new_hls_combine.shape[0] / 2):, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    midpoint = 320
    
    # current_leftX = np.argmax(histogram[160:midpoint])
    # current_rightX = np.argmax(histogram[midpoint:]) + midpoint
 
    # window_height = 18
    # num_windows = int((new_hls_combine.shape[0] - cut_height)/ window_height)
    
    window_height = 10
    num_windows = int((new_hls_combine.shape[0] - cut_height)/ window_height)
    
    nonzero = new_hls_combine.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    min_num_pixel = 50
    
    # gunhee(output)

    if mode == 'normal':
        current_leftX = 270
        current_rightX = 370
        win_left_lane = []
        win_right_lane = []

        # 6번째는 그냥 임의로 정함
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
            
            if window == 8 and end == False:
                window_white_pixels = len(left_window_inds)
                if window_white_pixels > 200:
                    print("left_mode start")
                    left_start_time = current_time
                    end = True
                    mode = 'left'
            if window == 7 and end == False:
                window_white_pixels_add = len(left_window_inds)
                if window_white_pixels_add > 85:
                    print("left_mode start")
                    left_start_time = current_time
                    end = True
                    mode = 'left'



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
        print(offset_ahead-offset_ahead_left)
    
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

        look_ahead_distance = 360 - 50
        
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
    
    # 일단 러프하게 P 제어
    if mode == 'normal':
        car.steering = -error*0.01
    elif mode == 'left':
        car.steering = -error*0.015

    # cv2.imshow('Output', output)
    
    if abs(before_white_pixels - window_white_pixels) < 15 and boost_exp == False and running_exp == True:
        throttle = throttle + 0.001

    elif abs(before_white_pixels - window_white_pixels) > 15 and boost_exp == False and running_exp == True:
        boost_exp = True 
        # boost = 0.024 + (0.3 - throttle)/5 
        throttle = throttle + boost
    
    before_white_pixels = window_white_pixels
    
    if car.throttle > 0.45:
        running = False
    
    car.throttle = float("{:.3f}".format(throttle))
    
    # car.throttle = 0
    # car.steering = 0
    # cv2.imshow('Lane Detection', output)
    
    image_filename = os.path.join(image_path, f"frame_world{image_number:05d}.jpg")
    cv2.imwrite(image_filename, output)
    
    data.append((image_number, window_white_pixels, window_white_pixels_add))
    
    # print(image_number, "steering : ",car.steering, "error : ",error, offset_ahead-offset_ahead_left, stopping)
    print(image_number, "steering : ",car.steering, "throttle : ", car.throttle, window_white_pixels, window_white_pixels_add)
    
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
            for image_number, window_white_pixels, window_white_pixels_add in data:
                f.write(f"{image_number}, 9번째 픽셀 개수 : {window_white_pixels}, 8번째 픽셀 개수 : {window_white_pixels_add}\n")
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
