import cv2
import numpy as np

# Открываем поток с камеры (замени на свой URL)
from data import ip_camera_url
capture = cv2.VideoCapture(ip_camera_url)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not capture.isOpened():
    print("Ошибка: Не удалось подключиться к камере.")
    exit()

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

    for contour in contours:
        # Проверяем площадь контура
        area = cv2.contourArea(contour)
        if area > 500:  # Фильтруем мелкие объекты
            # Вычисляем ограничивающую рамку для контура
            x, y, w, h = cv2.boundingRect(contour)

            # Проверяем пропорции на куб (приблизительно квадрат)
            if 0.9 <= w / h <= 1.1:
                # Рисуем прямоугольник вокруг обнаруженного куба
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(x, y)
                cv2.putText(frame, "Red Cube Detected", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Показываем оригинальное изображение и маску
    cv2.imshow("Original", frame)
    cv2.imshow("Red Mask", red_mask)

    # Останавливаем при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
capture.release()
cv2.destroyAllWindows()
