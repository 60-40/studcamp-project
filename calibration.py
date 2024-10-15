import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


import calibration_params
import map_config


def undistort(img, mtx, dist, nmtx, roi):
    # undistort
    dst = cv.undistort(img, mtx, dist, None, nmtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult_left.png', dst)
    return dst


def undistort_video(path, chess_path=None, board_size=(6,8), left=True, time=None, homography=False):
    cap = cv.VideoCapture(path)
    if time:
        cap.set(cv.CAP_PROP_POS_MSEC,time)
    
    if chess_path:
        ret, frame = cap.read()
        h,  w = frame.shape[:2]
        mtx,dist,nmtx,roi = calibration_params.get_matrices(chess_path, board_size, h, w)
        print(mtx)
        print(dist)
        print(nmtx)
        print(roi)
    elif left:
        mtx = calibration_params.get_left_mtx()
        dist = calibration_params.get_left_dist()
        nmtx = calibration_params.get_left_nmtx()
        roi = calibration_params.get_left_roi()
    else:
        mtx = calibration_params.get_right_mtx()
        dist = calibration_params.get_right_dist()
        nmtx = calibration_params.get_right_nmtx()
        roi = calibration_params.get_right_roi()
    
    i = 0

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            img = undistort(frame, mtx, dist, nmtx, roi)
            # find_corners(img)
            # print(ret)
            cv.imshow('Frame',img)
            k = cv.waitKey(25)
            # Press Q on keyboard to  exit
            if  k & 0xFF == ord('q'):
                break
            elif k & 0xFF == ord('p'):
                cv.imwrite(f"data/calibration/mapping/frame{i}.png",img)
                i+=1
            else:
                continue
        
        # Break the loop
        else: 
            break


def play_video(path, i=0):
    cap = cv.VideoCapture(path)
    # cap = video_capture_right
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Frame',frame)
            k = cv.waitKey(25)
            # Press Q on keyboard to  exit
            if  k & 0xFF == ord('q'):
                break
            elif k & 0xFF == ord('p'):
                cv.imwrite(f"data/calibration/right_2/frame{i}.png",frame)
                i+=1
            else:
                continue
        # Break the loop
        else: 
            break



if __name__ == "__main__":
    # img = cv.imread("data/frame0.png")
    # find_corners(img)
    # play_video("data/upcamera_chess.mp4")
    # undistort_video("data/Right_1.avi", chess_path="data/calibration/right_2/*.png", left=False)
    # undistort_video("data/Right_1.avi", chess_path="data/calibration/right/*.png",board_size=(6,8), left=False)
    undistort_video("data/Left_1.avi", left=True)

    # undistort_video("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1", left=True, crop=True)

    # undistort_video("rtsp://Admin:rtf123@192.168.2.251/251:554/1/1", left=False)

    # play_video("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1")
    # img = cv.imread("data/left_new.png")
    # img = undistort(img,
    #           calibration_params.get_left_mtx(),
    #           calibration_params.get_left_dist(),
    #           calibration_params.get_left_nmtx(),
    #           calibration_params.get_left_roi()
    #           )
    # plt.imshow(img),plt.show()