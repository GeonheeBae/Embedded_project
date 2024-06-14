import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')
import cv2

# 교차로 detect 같이 할 생각도 고려
cross_y_start = 195 # 상대방 기준 교차로 시작  
# cross_y_start = 100

# left_time = 2 # 좌회전 완주 시간
left_time = 1

# 특정 x좌표와 y좌표 범위 필터링 
x_min, x_max = 230, 240
# frame_world00133, frame_world00134 기준으로는 210, 220 적당..

y_min, y_max = 0, 200 # y_max는 교차로보다 짧게 => 이래도 차 검출 가능할 듯 

my_speed = 0

threshold = 360 - y_max # 위에서부터 얼마나 볼지

image_path = 'lane_keeping/'

def car_detect(output, y_pos_before, Ts, image_number):
    
    myspeed = my_speed/Ts
    # 새로운 빈 이미지 생성 (원본 이미지와 동일한 크기)
    filtered_image = np.zeros_like(output)

    filtered_image[y_min:y_max+1, x_min:x_max+1] = output[y_min:y_max+1, x_min:x_max+1]

    # 원본 이미지에 필터링 박스 그리기
    image_with_box = cv2.rectangle(output.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 배열 변환
    array = np.array(filtered_image)

    # 흰색 픽셀 저장
    coords_whiteish = np.column_stack(np.where(array >= threshold))

    # 속도 계산을 위한 y좌표 추출
    y_coords = coords_whiteish[:, 0]

    # 5개는 임의로 정한 값
    # def get_max(coords):
    #     unique, counts = np.unique(coords, return_counts=True)
    #     max_value = unique[counts >= 5]
    #     if len(max_value) > 0:
    #         return max_value[-1]
    #     return 'non-detected'

    def get_max(coords):
        unique, counts = np.unique(coords, return_counts=True)
        max_value = unique[counts >= 10]
        if len(max_value) > 0:
            max_count = counts[unique == max_value[-1]][0]
            return max_value[-1], max_count
        return 'non-detected', 0

    y_pos, max_count = get_max(y_coords)

    # 검출된 y_pos가 있으면 이미지에 점 찍기
    if y_pos != 'non-detected':
        image_with_box = cv2.circle(image_with_box, (int((x_min + x_max) / 2), y_pos), 5, (0, 0, 255), -1)

    # 모드 결정
    if y_pos_before == 'non-detected' and y_pos == 'non-detected':
        detection = 'go' # 아무 것도 없으면 출발
    elif y_pos == 'non-detected':
        detection = 'go' # 차가 사라 졌으니 출발
    elif y_pos_before == 'non-detected':
        detection = 'stop' # 차량의 빛의 속도일수도 있으니 정지 
    else: # y_pos_before, y_pos 모두 존재
        y_difference = y_pos - y_pos_before
        myspeed = my_speed/Ts 
        speed = y_difference / Ts 
        
        print(image_number, '속도: ', speed, "교차로까지 거리 :", cross_y_start-y_pos)

        # if cross_y_start-y_pos > (speed - myspeed)*left_time:
        if cross_y_start-y_pos > (speed - myspeed)*left_time >= 0:
        # if 1 > (speed - myspeed)*left_time:
            detection = 'go' # 차가 오기 전 좌회전 완수 가능
        else:
            detection = 'stop'
    
    print(image_number, "현 위치:",y_pos,"전 위치:",y_pos_before, "검출 개수:", max_count,"detection:", detection, '주기:', Ts)
    # 결과 이미지 저장
    image_filename = os.path.join(image_path, f"frame_detect{image_number:05d}.jpg")
    cv2.imwrite(image_filename, image_with_box)
    
    return detection, y_pos
