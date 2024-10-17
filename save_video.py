import cv2

# URL для подключения к IP-камере (замените на ваш URL)
#ip_camera_url = "rtsp://Admin:rtf123@192.168.2.251/251:554/1/1" # Правая
#ip_camera_url = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1" #  Левая
ip_camera_url = 'http://192.168.2.165:8080/?action=stream' # На роботе
# Создаем объект VideoCapture для захвата видео с IP-камеры
video_capture = cv2.VideoCapture(ip_camera_url)

# Проверка, удалось ли открыть IP-камеру
if not video_capture.isOpened():
    print("Ошибка: Не удалось подключиться к IP-камере.")
    exit()

# Получение параметров видео
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Можно изменить на нужное значение

# Создаем объект VideoWriter для записи видео
output_filename = "output_video_left_vith_red.avi"  # Имя выходного файла
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Кодек для записи видео
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Цикл для захвата и записи кадров
while True:
    ret, frame = video_capture.read()  # Чтение кадра

    if not ret:
        print("Не удалось получить кадр. Завершаем работу.")
        break

    video_writer.write(frame)  # Запись кадра в файл

    # Отображение кадра (по желанию)
    cv2.imshow("IP Camera Stream", frame)

    # Ожидание нажатия клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()

print("Запись завершена, видео сохранено как:", output_filename)
