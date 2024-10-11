import socket
import time
import struct

host = "192.168.2.165"
port = 2001


def send_command(command):
    try:
        # Создаем сокет
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"Соединение с {host}:{port}")

        # Устанавливаем соединение
        s.connect((host, port))
        print(f"Отправка команды: {command}")

        # Отправляем команду
        s.sendall(command)

        # Добавляем небольшой задержку между отправками команд
        time.sleep(1)

        return True
    except socket.error as e:
        print(f"Ошибка сокета: {e}")
        return False
    finally:
        # Закрываем соединение
        s.close()
        print("Соединение закрыто")

command = b'\xff\x01\x01\xb4\xff'  # Пример отправки команды
esult = send_command(command)
command = b'\xff\x01\x02\x87\xff'  # Пример отправки команды
esult = send_command(command)

# Первая команда
while True:
    break
    command = b'\xff\x01\x03\xb4\xff'  # Пример отправки команды
    result = send_command(command)
    # print('Команда на установку цвета отправлена: ', result)
    time.sleep(2)
    command = b'\xff\x01\x03\x00\xff'  # Пример отправки команды
    result = send_command(command)
    time.sleep(2)
def test_servo(id):
    print(f"Проверка углов поворота сервопривода {id}")
    for angle in range(0, 181, 45):
        # Создание команды с изменённым значением угла
        command = b'\xff\x01' + bytes([id + 1]) + bytes([angle+180]) + b'\xff'
        result = send_command(command)
        print(angle, command)
        time.sleep(1)

def turnLeft():
    command = b'\xff\x01\x03\xb4\xff'

test_servo(3)
