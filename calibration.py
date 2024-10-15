import numpy as np
import cv2 as cv
import calibration_params

def undistort(img, mtx, dist, nmtx, roi):
    # undistort
    dst = cv.undistort(img, mtx, dist, None, nmtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult_left.png', dst)
    return dst


def undistort_video(path, left=True):
    cap = cv.VideoCapture(path)
    
    if left:
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
            # print(ret)
            cv.imshow('Frame',img)
            k = cv.waitKey(25)
            # Press Q on keyboard to  exit
            if  k & 0xFF == ord('q'):
                break
            elif k & 0xFF == ord('p'):
                cv.imwrite(f"data/frame{i}.png",frame)
                i+=1
            else:
                continue
        
        # Break the loop
        else: 
            break


if __name__ == "__main__":
    undistort_video("data/Right_1.avi", left=False)