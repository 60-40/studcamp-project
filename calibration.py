import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from ultralytics import YOLO
import math
import socket


import calibration_config
import map_config


robot_ip = '192.168.2.165'  # Robot's IP address
robot_port = 5000  # Port to send data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Using UDP


def undistort(img, mtx, dist, nmtx, roi):
    # undistort
    dst = cv.undistort(img, mtx, dist, None, nmtx)
    
    # crop the image
    x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # cv.imwrite('calibresult_left.png', dst)
    return dst


def undistort_video(path,
                    chess_path=None, 
                    board_size=(6,8), 
                    left=True, 
                    time=None, 
                    homography=False, 
                    save_video=False, 
                    show=True, 
                    yolo=False
                    ):
    cap = cv.VideoCapture(path)
    if time:
        cap.set(cv.CAP_PROP_POS_MSEC,time)
    if yolo:
        model_path = "data/best_v3_huest_up.pt"
        model = YOLO(model_path)

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
    i = 11
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            img = undistort(frame, mtx, dist, nmtx, roi)
            if homography == True:
                borders_c = calibration_config.get_border_corners() if left else calibration_config.get_right_border_corners()
                img = find_homography(img,borders_c)
            if yolo:
                results = model(img)
                # Render the detections on the frame
                img = results[0].plot()  # Draw bounding boxes, labels, etc.
        
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
                    cv.imwrite(f"out/cubes/right/frame{i}.png",img)
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


def calculate_contour_distance(contour1, contour2):
    x1, y1, w1, h1 = cv.boundingRect(contour1)
    c_x1 = x1 + w1 / 2
    c_y1 = y1 + h1 / 2

    x2, y2, w2, h2 = cv.boundingRect(contour2)
    c_x2 = x2 + w2 / 2
    c_y2 = y2 + h2 / 2

    return max(abs(c_x1 - c_x2) - (w1 + w2) / 2, abs(c_y1 - c_y2) - (h1 + h2) / 2)


def merge_contours(contour1, contour2):
    return np.concatenate((contour1, contour2), axis=0)


def agglomerative_cluster(contours, threshold_distance=40.0):
    current_contours = contours
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours) - 1):
            for y in range(x + 1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None:
                    min_distance = distance
                    min_coordinate = (x, y)
                elif distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else:
            break

    return current_contours


def test(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours = list(filter(lambda x: cv.contourArea(x) > 500, contours))
    # cv.drawContours(img,sorted(contours, key=lambda x: -cv.contourArea(x)),0,255,5)
    
    # contours = agglomerative_cluster(list(contours), 40)

    rect = max(contours, key=cv.contourArea)
    hull = cv.convexHull(rect)
    output_image = img.copy()
    epsilon = 0.02 * cv.arcLength(hull, True)
    approx = cv.approxPolyDP(hull, epsilon, True)
    cv.drawContours(img,sorted([approx], key=lambda x: -cv.contourArea(x)),0,255,5)
    # return approx 
    x, y, w, h = cv.boundingRect(hull)
    return output_image[y:y + h, x:x + w]


def get_corners(cnt):
    corners = sorted([x[0] for x in cnt], key=lambda c: c[1])
    up = corners[:2]
    down = corners[2:]
    left_up,right_up = sorted(up, key=lambda c: c[0])
    left_down,right_down = sorted(down, key=lambda c: c[0])
    return [left_up, right_up, left_down, right_down]
    # print(up, down)

def find_homography(frame, scale_map=2, test1=True, test=False):
    # (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = map_config.border_corners
    if test1:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = find_corners(frame)
    elif test:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rect = max(contours, key=cv.contourArea)
        hull = cv.convexHull(rect)
        epsilon = 0.02 * cv.arcLength(hull, True)
        approx = cv.approxPolyDP(hull, epsilon, True)
        # cv.drawContours(frame,contours,-1,255,5)
        # cv.imshow("frame", frame)
        # cv.waitKey(0)

        # print(approx[0])
        # get_corners(approx)
        # (x,y,w,h) = cv.boundingRect(approx)
        
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = get_corners(approx)
    else:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = calibration_config.get_border_corners()
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = find_corners(frame)
    (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4) = map_config.border_corners
    X1, Y1, X2, Y2, X3, Y3, X4, Y4 = [i * scale_map for i in [X1, Y1, X2, Y2, X3, Y3, X4, Y4]]
    # Опорные точки на изображении и соответствующие точки на карте
    image_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype="float32")
    map_points = np.array([[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]], dtype="float32")

    # Вычисляем матрицу гомографии для проецирования изображения на карту
    H, status = cv.findHomography(image_points, map_points)

    return H
    # Применяем матрицу гомографии, чтобы преобразовать откалиброванное изображение
    aligned_frame = cv.warpPerspective(frame, H, (map_width, map_height))

    return aligned_frame


def warp_frame(frame, H, scale_map=2):
    w, h = map_config.map_size
    map_width, map_height = [i * scale_map for i in [w, h]]
    # H = H if H else find_homography(frame, scale_map)
    aligned_frame = cv.warpPerspective(frame, H, (map_width, map_height))
    return aligned_frame


def find_biggest_hsv_object_center(frame, lower_hsv, upper_hsv):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    red_mask = cv.inRange(hsv_frame, lower_hsv, upper_hsv)

    contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    biggest = max(contours, key=cv.contourArea)
    (x,y,w,h) = cv.boundingRect(biggest)

    return (x + w//2, y + h//2)


def get_errors(frame, robor_center, prev_center):
    end_pt = (105, 500)
    e1 = get_angle(robor_center, prev_center, end_pt)
    e2 = get_distance(robor_center, end_pt)
    return e1,e2


def play_video_with_frame_processing(path, procs, i=0, out_path="out"):
    # cap = video_capture_right
    robor_center = (100, 65) #TODO центр робота в начале
    while True:
        cap = cv.VideoCapture(path)
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Frame processing
                for p in procs:
                    frame = p(frame)
                # robor = find_biggest_robor(frame)
                # if robor:
                #     robor_center, prev_center = robor[1], robor_center
                #     print("center", robor_center)
                #     e1, e2 = get_errors(frame, robor_center, prev_center)
                #     print("error",e1,e2)
                #     message = f"{0},{0},{e1},{e2},{0},{0}".encode()  # Format: "e_x,e_y"
                #     sock.sendto(message, (robot_ip, robot_port))
                # Display the resulting frame
                cv.imshow('Frame',frame)
                k = cv.waitKey(25)
                # Press Q on keyboard to  exit
                if  k & 0xFF == ord('q'):
                    break
                elif k & 0xFF == ord('p'):
                    cv.imwrite(f"{out_path}/frame{i}.png",frame)
                    i+=1
                else:
                    continue
            # Break the loop
            else: 
                break
        cv.waitKey(0)


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


def get_yolo_coords(frame):
    results = model(frame)
    names = results[0].names
    return [
        [tuple(results[0].boxes.xyxy[i]), 
         tuple(results[0].boxes.xywh[i][:2]), 
         str(names[int(results[0].boxes.cls[i])]), 
         float(results[0].boxes.conf[i])
         ] for i in range(results[0].boxes.shape[0])]


def find_biggest_robor(frame):
    res = get_yolo_coords(frame)
    biggest_robor = None
    robor_area = 0
    for (x1,y1,x2,y2), (cx, cy), name, conf in get_yolo_coords(frame):
        if name != "robor":
            continue
        area =  (x2 - x1) * (y2 - y1)
        if area > robor_area:
            robor_area = area
            biggest_robor = [(int(x1),int(y1),int(x2),int(y2)), (int(cx), int(cy)), name, conf]
    return biggest_robor


def find_vector_of_robor(frame):
    robor = find_biggest_robor(frame)
    if not robor:
        return frame
    
    x2, y2 = find_biggest_hsv_object_center(frame, *calibration_config.get_robor_hsv())
    
    x1, y1 = robor[1]

    x2, y2 = 2 * x1 - x2, 2 * y1 - y2

    cv.arrowedLine(frame,(x1,y1),(x2,y2),(0,255,0),5)
    return frame


def get_angle(robor_center, prev_center, end_pt):
    
    AB = (prev_center[0] - robor_center[0], prev_center[1] - robor_center[1])
    BC = (end_pt[0] - prev_center[0], end_pt[1] - prev_center[1])
    
    # Calculate the angle using atan2
    angle_rad = math.atan2(BC[1], BC[0]) - math.atan2(AB[1], AB[0])
        
    return angle_rad


def find_corners(frame, show_mask =False):
    frame = ~frame
    mask = np.zeros_like(frame[:, :, 0], dtype='uint8')
    radius = 50
    centers = [(354, 100), (1432, 123), (1390, 930), (345, 915)] #Ручками
    for center in centers:
        cv.circle(mask, center, radius, 255, -1)

    masked_frame = cv.bitwise_and(frame, frame, mask=mask)
    masked_frame = ~masked_frame  # Инвертируем маску
    if show_mask:
        cv.imshow("mask", masked_frame)
    # Преобразуем обрезанное изображение в черно-белое
    # gray = cv.cvtColor(masked_frame, cv.COLOR_BGR2GRAY)
    _, th = cv.threshold(masked_frame, 110,255,cv.THRESH_BINARY)
    gray = cv.cvtColor(th, cv.COLOR_BGR2GRAY)
    # Найдем крайние черные точки на всей фотографии
    black_pixels = np.argwhere(gray == 0)  # Найдем все черные пиксели (значение 0)

    if black_pixels.size > 0:
        # Самая верхняя левая точка
        top_left = black_pixels[np.argmin(np.sum(black_pixels, axis=1))]
        # Самая верхняя правая точка
        top_right = black_pixels[np.argmin(black_pixels[:, 0] - black_pixels[:, 1])]
        # Самая нижняя левая точка
        bottom_left = black_pixels[np.argmax(black_pixels[:, 0] - black_pixels[:, 1])]
        # Самая нижняя правая точка
        bottom_right = black_pixels[np.argmax(black_pixels[:, 0] + black_pixels[:, 1])]

        
        return (tuple(top_left[::-1]), tuple(top_right[::-1]), tuple(bottom_left[::-1]), tuple(bottom_right[::-1]))


def get_distance(start_pt, end_pt):
    return int(((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)**.5)


def get_base_coord(frame, color):
    lower_hsv, upper_hsv = calibration_config.get_red_base_hsv() if color == "red" else calibration_config.get_green_base_hsv()
    return find_biggest_hsv_object_center(frame, lower_hsv, upper_hsv)


if __name__ == "__main__":
    # img = cv.imread("out/frame0.png")
    # find_corners(img)
    # play_video("data/upcamera_chess.mp4")
    # undistort_video("output_video_right_cubes.avi", left=False, homography=False, save_video=True)
    # undistort_video("data/Right_1.avi", chess_path="data/calibration/right/*.png",board_size=(6,8), left=False)
    # undistort_video("data/Left_1.avi", left=True, crop=True)

    # undistort_video("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1", left=True, homography=True, yolo=True)
    
    cap = cv.VideoCapture("output_video_left_base.avi")
    img = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img = frame
            break
    cv.imshow("orig",img)
    mtx = calibration_config.get_left_mtx()
    dist = calibration_config.get_left_dist()
    nmtx = calibration_config.get_left_nmtx()
    roi = calibration_config.get_left_roi()
    img = undistort(img, mtx, dist, nmtx, roi)
    cv.imshow("undistorted", img)
    points = find_corners(img, True)
    for p in points:
        cv.circle(img, p, 3, 255, 4)
    cv.imshow("points", img)
    H = find_homography(img)
    img = warp_frame(img, H)
    cv.imshow("warped", img)
    print(points)
    cv.waitKey(0)
    exit(0)
    procs = []
    proc_undistort_frame = lambda frame: undistort(frame, mtx, dist, nmtx, roi)
    H = None
    proc_warp_frame = lambda frame: warp_frame(frame, H)
    model_path = "data/best_v3_huest_up.pt"
    model = YOLO(model_path)
    def proc_yolo(frame):
        results = model(frame)
        # Render the detections on the frame
        return results[0].plot()  # Draw bounding boxes, labels, etc.
    
    # proc_add_mask_red = lambda frame: add_mask(frame, (21,180), (0,11), (243,255))
    # proc_add_mask_green = lambda frame: add_mask(frame, (173,180), (65,82), (169,224))

    procs = [proc_undistort_frame, proc_warp_frame]
    play_video_with_frame_processing("output_video_left_base.avi", procs)
    # play_video_with_frame_processing("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1", procs)

    # img = cv.imread("data/left.png")
    # tes = warp_frame(img)
    # cv.imshow('test',tes)
    # k = cv.waitKey(0)
    
    # play_video("rtsp://Admin:rtf123@192.168.2.250/251:554/1/1")
    # undistort_video("rtsp://Admin:rtf123@192.168.2.251/251:554/1/1", left=False)
    # undistort_video("output_video_left_base.avi", left=True)
    # play_video("data/output_video_left_cubes.avi")
    
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