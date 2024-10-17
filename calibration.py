import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


import calibration_config
import map_config


def undistort(img, mtx, dist, nmtx, roi):
    # undistort
    dst = cv.undistort(img, mtx, dist, None, nmtx)
    
    # crop the image
    x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult_left.png', dst)
    return dst


def undistort_video(path, chess_path=None, board_size=(6,8), left=True, time=None, homography=False, save_video=False, show=True):
    cap = cv.VideoCapture(path)
    if time:
        cap.set(cv.CAP_PROP_POS_MSEC,time)
    
    if chess_path:
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        mtx,dist,nmtx,roi = calibration_config.get_matrices(chess_path, board_size, h, w)
        print(mtx)
        print(dist)
        print(nmtx)
        print(roi)
    elif left:
        mtx = calibration_config.get_left_mtx()
        dist = calibration_config.get_left_dist()
        nmtx = calibration_config.get_left_nmtx()
        roi = calibration_config.get_left_roi()
    else:
        mtx = calibration_config.get_right_mtx()
        dist = calibration_config.get_right_dist()
        nmtx = calibration_config.get_right_nmtx()
        roi = calibration_config.get_right_roi()
    
    if save_video:
        fps = 30
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # print(frame_width, frame_height)
        output_filename = f"out/{path[path.rfind('/')+1:path.rfind('.')]}_undistorted.avi"  # Имя выходного файла
        # print(output_filename)
        
        fourcc = cv.VideoWriter_fourcc(*'XVID')  # Кодек для записи видео
        video_writer = cv.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    i = 0
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            img = undistort(frame, mtx, dist, nmtx, roi)
            if homography == True:    
                img = findHomography(img,calibration_config.get_border_corners())
            # print(img.shape[:2])
            # find_corners(img)
            # print(ret)
            if save_video:
                video_writer.write(img)
            
            if show:
                cv.imshow('Frame',img)
                k = cv.waitKey(25)
                # Press Q on keyboard to  exit
                if  k & 0xFF == ord('q'):
                    break
                elif k & 0xFF == ord('p'):
                    cv.imwrite(f"out/frame{i}.png",img)
                    i+=1
                else:
                    continue
        
        # Break the loop
        else: 
            break
    cap.release()
    if save_video:
        video_writer.release()
    cv.destroyAllWindows()


def findHomography(frame, borders):
    # (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = map_config.border_corners
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = borders
    (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = [(0,0),(800,0),(0,620),(800,620)]
    map_width, map_height = 800,620 #x2 чтобы смотреть нормально
    # map_width, map_height = map_config.map_size

    # Опорные точки на изображении и соответствующие точки на карте
    image_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype="float32")
    map_points = np.array([[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]], dtype="float32")

    # Вычисляем матрицу гомографии для проецирования изображения на карту
    H, status = cv.findHomography(image_points, map_points)

    # Применяем матрицу гомографии, чтобы преобразовать откалиброванное изображение
    aligned_frame = cv.warpPerspective(frame, H, (map_width, map_height))

    return aligned_frame


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
    # undistort_video("data/Left_1.avi", left=True, crop=True)

    # undistort_video("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1", left=True)

    # play_video("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1")
    # undistort_video("rtsp://Admin:rtf123@192.168.2.251/251:554/1/1", left=False)
    undistort_video("data/Left_1.avi", left=True, homography=True)
    # play_video("out/Left_1_undistorted.avi")
    # play_video("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1")
    # img = cv.imread("data/left_new.png")
    # img = undistort(img,
    #           calibration_params.get_left_mtx(),
    #           calibration_params.get_left_dist(),
    #           calibration_params.get_left_nmtx(),
    #           calibration_params.get_left_roi()
    #           )
    # plt.imshow(img),plt.show()