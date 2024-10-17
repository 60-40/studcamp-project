import cv2
import numpy as np
from control_devices.motor import MotorController
import time
time_now = time.time()

goal_x = 100 #360
goal_y = 400 #240
ex=0
ey = 0
ex_i = 0
ey_i = 0

# Открываем поток с камеры (замени на свой URL)
from data import ip_camera_url
capture = cv2.VideoCapture(ip_camera_url)


# Проверяем, удалось ли открыть камеру
if not capture.isOpened():
    print("Ошибка: Не удалось подключиться к камере.")
    exit()

# Настройки для записи видео в формате MP4V
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Частота кадров
output_filename = "output_video_cube.mp4"  # Имя выходного файла
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек MP4V
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

motorController = MotorController()

while True:
    
    ret, frame = capture.read()
    
    if not ret:
        print("Ошибка: Не удалось получить кадр.")
        break

    # Преобразуем изображение в HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Улучшаем контраст изображения для тёмных участков
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab_frame)

    # Применяем CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Объединяем обратно улучшенные каналы
    lab_frame = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(lab_frame, cv2.COLOR_LAB2BGR)

    # Преобразуем улучшенное изображение в HSV
    hsv_enhanced = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)

    # Диапазон для красного цвета (корректируем пороги для тусклых оттенков)
    lower_red_1 = np.array([0, 30, 30])   # Уменьшаем пороги по насыщенности и яркости
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 30, 30])
    upper_red_2 = np.array([180, 255, 255])

    # Маска для красного цвета
    mask_red_1 = cv2.inRange(hsv_enhanced, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv_enhanced, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask_red_1, mask_red_2)

    # Применяем морфологические операции для улучшения маски
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Находим контуры на маске
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contour = contours[0]
        # Проверяем площадь контура
        area = cv2.contourArea(contour)
        if area > 2000:  # Фильтруем мелкие объекты
            # Вычисляем ограничивающую рамку для контура
            x, y, w, h = cv2.boundingRect(contour)

            # Рисуем прямоугольник вокруг обнаруженного куба
            x_now = x + w // 2
            y_now = y + h // 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (goal_x, goal_y), 3, (255, 255, 0), 3)
            cv2.circle(frame, (x_now, y_now), 3, (255, 0, 0), 3)

            cv2.putText(frame, "Red Cube Detected", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Цикл П-регулятора
            ex_last = ex
            ey_last = ey
            ex = goal_x - x_now
            ey = goal_y - y_now

            time_last = time_now
            time_now = time.time()

            dt = time_now - time_last
            if dt > 0.15: dt = 0.15

            ex_i += (ex + ex_last)/2*dt
            ey_i += (ey + ey_last)/2*dt

            if ex_i > 1000: ex_i = 1000
            if ey_i > 1000: ey_i = 1000


            kp_x = 0.02
            kp_y = 0.2
            ki_x = 0.03
            ki_y = 0.03

            U_l = int(-kp_x * ex + kp_y * ey - ki_x * ex_i + ki_y * ey_i)
            U_l = min(100, U_l)
            U_l = max(-100, U_l)


            U_r = int(kp_x * ex + kp_y * ey + ki_x * ex_i + ki_y * ey_i)
            U_r = min(100, U_r)
            U_r = max(-100, U_r)

            

            print(U_l, U_r, ex, ey, dt)

            motorController.set_left_speed(U_l)
            motorController.set_right_speed(U_r)
        else:
            motorController.set_left_speed(0)
            motorController.set_right_speed(0)

    # Показываем оригинальное изображение и маску
    cv2.imshow("Original", frame)
    cv2.imshow("Red Mask", red_mask)

    # Запись текущего кадра в видео
    video_writer.write(frame)

    # Останавливаем при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

motorController.set_left_speed(0)
motorController.set_right_speed(0)
motorController.stop()
# Освобождаем ресурсы
capture.release()
video_writer.release()
cv2.destroyAllWindows()
