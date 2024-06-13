# calibration filter / sobel filter / conversion from image to world surface 

import cv2
import numpy as np

with np.load('camera_calibration_params2.npz') as data:
    mtx = data['mtx']
    dist = data['dist']
    newcameramtx = data['new']

height, width = 360, 720 # frame.shape[:2]
th_h, th_l, th_s = (150, 255), (0, 145), (0, 255) 
# th_h, th_l, th_s = (155, 255), (50, 150), (0, 255)
# th_h, th_l, th_s = (160, 255), (50, 160), (0, 255)
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
cut_height = 160
mid_x, mid_y = 320, 180

def calibration(frame, mode):
    if mode == 'normal':
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
        undistorted_img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        vertices = np.array([[(0, cut_height), (0, height), (610, height), (390, cut_height)]], dtype=np.int32)
        mask = np.zeros_like(undistorted_img)
        #mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        frame = cv2.bitwise_and(undistorted_img, mask)
        #frame = cv2.bitwise_and(frame, mask)
        return frame
        
    else:
        vertices = np.array([[(0, cut_height), (0, height), (610, height), (390, cut_height)]], dtype=np.int32)
        #mask = np.zeros_like(undistorted_img)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, (255, 255, 255))
        #frame = cv2.bitwise_and(undistorted_img, mask)
        frame = cv2.bitwise_and(frame, mask)
        return frame

def sobel(frame):
    img = frame[0:height, :width, 2]
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
    H = hls[0:height, 0:width, 0]
    L = hls[0:height, 0:width, 1]
    S = hls[0:height, 0:width, 2]

    h_img = np.zeros_like(H)
    h_img[(H > th_h[0]) & (H <= th_h[1])] = 255

    l_img = np.zeros_like(L)
    l_img[(L > th_l[0]) & (L <= th_l[1])] = 255

    s_img = np.zeros_like(S)
    s_img[(S > th_s[0]) & (S <= th_s[1])] = 255

    hls_combine = np.zeros_like(s_img).astype(np.uint8)
    hls_combine[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255 
    
    return hls_combine

def img2world(hls_combine):
    white_pixel_coords = np.argwhere(hls_combine == 255)
    new_hls_combine = np.zeros_like(hls_combine)

    b = mid_y - 120 # 소실점

    k_y = 297
    k_x = 1/297

    if white_pixel_coords.size > 0:
        y = white_pixel_coords[:, 0]
        x = white_pixel_coords[:, 1]
        origin_y = y.copy()
        y = mid_y - y
        x = x - mid_x

        mask1 = origin_y > cut_height
        mask2 = ~mask1

        y1 = y[mask1]
        x1 = x[mask1]

        if y1.size > 0:
            new_y1 = k_y * (y1) / (b - y1)
            new_x1 = k_x * x1 * (k_y + new_y1)
            new_x1 = new_x1 + mid_x
            new_y1 = mid_y - new_y1 - 42

            new_y1 = new_y1.astype(np.int32)
            new_x1 = new_x1.astype(np.int32)

            valid_mask1 = (0 <= new_x1) & (new_x1 < new_hls_combine.shape[1]) & (0 <= new_y1) & (new_y1 < new_hls_combine.shape[0])
            new_hls_combine[new_y1[valid_mask1], new_x1[valid_mask1]] = 255

        x2 = x[mask2]
        y2 = y[mask2] + 1000

        valid_mask2 = (0 <= x2) & (x2 < new_hls_combine.shape[1]) & (0 <= y2) & (y2 < new_hls_combine.shape[0])
        new_hls_combine[y2[valid_mask2], x2[valid_mask2]] = 255

    output = np.dstack((new_hls_combine, new_hls_combine, new_hls_combine)) * 10
    
    return output, new_hls_combine

def yellow(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    H = hls[0:height, 0:width, 0]
    L = hls[0:height, 0:width, 1]
    S = hls[0:height, 0:width, 2]

    h_img = np.zeros_like(H)
    h_img[(H > th_h[0]) & (H <= th_h[1])] = 255

    l_img = np.zeros_like(L)
    l_img[(L > th_l[0]) & (L <= th_l[1])] = 255

    s_img = np.zeros_like(S)
    s_img[(S > th_s[0]) & (S <= th_s[1])] = 255

    hls_combine = np.zeros_like(s_img).astype(np.uint8)
    hls_combine[((s_img > 1) & (l_img == 0)) | ((s_img == 0) & (h_img > 1) & (l_img > 1))] = 255 
    
    return hls_combine
