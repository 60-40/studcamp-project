import numpy as np
import cv2 as cv
import glob


def get_matrices(images_path, board_size, h, w):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    print(board_size)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_size[0]*board_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[1],0:board_size[0]].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(images_path)

    # print(images)

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (board_size[1],board_size[0]), None)
        # print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img, (board_size[1],board_size[0]), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
    
    cv.destroyAllWindows()

    # print(len(objpoints))
    # print(len(imgpoints))
    # print(imgpoints)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    return mtx, dist, newcameramtx, roi


# img = cv.imread("data/frame0.png")
# h,  w = img.shape[:2]
# mtx, dist, nmtx, roi = get_matrices('data/calibration/left/*.png', h, w)

# print("Left")
# print(f"mtx = {mtx}")
# print(f"dist = {dist}")
# print(f"nmtx = {nmtx}")
# print(f"roi = {roi}")


# img = cv.imread("data/realmap_fix.png")
# h,  w = img.shape[:2]
# mtx, dist, nmtx, roi = get_matrices('data/calibration/right_2/*.png', h, w)

# print("Right")
# print(f"mtx = {mtx}")
# print(f"dist = {dist}")
# print(f"nmtx = {nmtx}")
# print(f"roi = {roi}")


def get_left_shape():
    return (869, 1836)

#Left
def get_left_mtx():
    return np.array([
        [1.17917746e+03, 0.00000000e+00, 9.51905519e+02],
        [0.00000000e+00, 1.17596282e+03, 5.95586025e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])


def get_left_dist():
    return np.array([[-0.37393277,  0.08905364, -0.00820677, -0.00703832,  0.03638609]])


def get_left_nmtx():
    return np.array([
        [871.59467944, 0., 924.77023611],
        [0., 863.83946246, 578.52675802],
        [0., 0., 1.]
    ])


def get_left_roi():
    return (34, 96, 1836, 869)

# Right
def get_right_mtx():
    return np.array([
        [1.16194332e+03, 0.00000000e+00, 9.74250968e+02],
        [0.00000000e+00, 1.16432824e+03, 6.09104292e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])


def get_right_dist():
    return np.array([[-0.39288495,  0.11864625,  0.00158226, -0.01030355,  0.12343585]])


def get_right_nmtx():
    return np.array([
        [977.79085272, 0., 945.72985712],
        [0., 947.7267782, 609.2606394],
        [0., 0., 1.]
    ])


def get_right_roi():
    return (16, 45, 1884, 974)


def get_corners_approx():
    return [(354, 100), (1432, 123), (1390, 930), (345, 915)]


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
    gray = cv.cvtColor(masked_frame, cv.COLOR_BGR2GRAY)

    # Найдем крайние черные точки на всей фотографии
    black_pixels = np.argwhere(gray == 0)  # Найдем все черные пиксели (значение 0)

    if black_pixels.size > 0:
        # Самая верхняя левая точка
        top_left = black_pixels[np.argmin(np.sum(black_pixels, axis=1))]
        # Самая верхняя правая точка
        top_right = black_pixels[np.argmin(black_pixels[:, 0] - black_pixels[:, 1])]
        # Самая нижняя левая точка
        bottom_left = black_pixels[np.argmax(black_pixels[:, 0] + black_pixels[:, 1])]
        # Самая нижняя правая точка
        bottom_right = black_pixels[np.argmax(black_pixels[:, 0] - black_pixels[:, 1])]

        
        return (tuple(top_left[::-1]), tuple(top_right[::-1]), tuple(bottom_left[::-1]), tuple(bottom_right[::-1]))


def get_border_corners():
    # return [(304,120),(1386,140),(311,947),(1353,948)]
    # return [(330,105),(1418,134),(331,933),(1375,944)]
    # return [(358,2),(1450,43),(350,836),(1400,858)]
    return [(400, 100), (1494,134), (397, 932), (1447, 953)]


def get_right_border_corners():
    return [(330,115),(1531,108),(329,1000),(1520,1011)]


def get_red_robor_hsv():
    return (21, 0, 243), (180, 11, 255)


def get_green_robor_hsv():
    return (173, 65, 169), (180, 82, 224)


def get_red_base_hsv():
    return (0, 108, 193), (6, 255, 255)


def get_green_base_hsv():
    return (77, 90, 64), (84, 255, 150)

