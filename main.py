import numpy as np
import cv2 as cv
from enum import Enum
from ultralytics import YOLO
import math
import socket
import subprocess
import sys
from multiprocessing import Process

import calibration, calibration_config
import map as mp, map_config
from yolo_object_detection import find_target
import time


class Color(Enum):
    Red = 0
    Green = 1
    

class Target(Enum):
    Cube = 0
    Ball = 1
    Basket = 2
    Base = 3
    OpponentBasket = 4 #ну а вдруг???


class CameraProcessing():
    H = None
    preprocs = []
    rt_procs = []
    out_path = "out/main"
    address = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1" # Top view camera
    robot_ip = '192.168.2.165'  # Robot's IP address
    robot_port = 5000  # Port to send data
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Using UDP
    
    model_path = "data/best_v3_huest_up.pt" # YOLO model
    
    start_frame = None
    
    cur_frame = None
    yolo_res = []
    robor = []
    opponent = []

    map_graph = []
    path = []
    base_node = None
    opponent_base_node = None
    buttons1_node = None
    buttons2_node = None
    current_node = None
    cubes1_node = None #closest
    cubes2_node = None
    ball_node = None
    target_node = None
    cur_center = (-1, -1)
    prev_center = (-1, -1)
    target_proc = None
    busy = False

    we_take_cube: bool = False
    
    def __init__(self, color, address=None, pos=1):
        self.address = address or self.address
        self.color = color
        self.model = YOLO(self.model_path)
        self.map_graph = mp.generate_graph()
        self.pos = pos
        self.add_preprocs()
        self.add_rt_procs()

    
    def add_preprocs(self):
        self.preprocs.append(self.capture_start_frame)
        self.preprocs.append(self.set_undistort_configuration)
        self.preprocs.append(self.undistort_on_start)
        self.preprocs.append(self.calculate_homography_matrix)
        self.preprocs.append(self.warp_on_start)

        # self.preprocs.append(self.get_base_coord_on_start) #возможно не нужно
        self.preprocs.append(self.find_inner_walls)
        self.preprocs.append(self.find_base_node)
        self.preprocs.append(self.get_yolo_coords_on_start)
        self.preprocs.append(self.find_robor_on_start) #ручками
        self.preprocs.append(self.find_current_node_on_start)
        self.preprocs.append(self.set_ball_on_graph)

        # here we go
    
    
    def add_rt_procs(self):
        self.rt_procs.append(lambda frame: calibration.undistort(frame, self.mtx, self.dist, self.nmtx, self.roi))
        self.rt_procs.append(lambda frame: calibration.warp_frame(frame, self.H))
        self.rt_procs.append(self.get_yolo_coords)
        self.rt_procs.append(self.find_robor) #трекаем по кадрам
        self.rt_procs.append(self.find_current_node) 
        self.rt_procs.append(self.find_cubes)
        self.rt_procs.append(self.draw_graph)
        self.rt_procs.append(self.draw_vector)
        self.rt_procs.append(self.total_shit) #тут срем


    def capture_start_frame(self):
        cap = cv.VideoCapture(self.address)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                self.start_frame = frame
                self.cur_frame = frame
                print("INFO: captured first frame")
                cap.release()
                return
        
        
    def undistort_on_start(self):
        self.start_frame = calibration.undistort(self.start_frame, self.mtx, self.dist, self.nmtx, self.roi)
        

    def warp_on_start(self):
        self.start_frame = calibration.warp_frame(self.start_frame, self.H)
        

    def start(self, show=True):
        print("INFO: starting...")
        cnt = 0
        while True:
            try:
                self.prepocess()
            except Exception as e:
                cnt += 1
            if cnt == 100 or self.current_node is not None:
                break
        print("INFO: preprocess done.")
        cap = cv.VideoCapture(self.address)
        cap.set(cv.CAP_PROP_FPS, 5)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 0)
        cnt = 0
        k = 0
        while True:
            cnt += 1
            try:
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        self.cur_frame = frame
                        self.process_frame()
                        if show:
                            cv.imshow('Frame',self.cur_frame)
                            k = cv.waitKey(25)
                            # Press Q on keyboard to  exit
                            if  k & 0xFF == ord('q'):
                                break
                            elif k & 0xFF == ord('p'):
                                cv.imwrite(f"{self.out_path}/frame{i}.png",frame)
                                i+=1
                        else:
                            continue
                    else: 
                        break
            except Exception as e:
                cap.release()
                cap = cv.VideoCapture(self.address)
                cap.set(cv.CAP_PROP_FPS, 5)
                cap.set(cv.CAP_PROP_BUFFERSIZE, 0)
            if cnt == 20 or k & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        

    def set_undistort_configuration(self):
        self.mtx = calibration_config.get_left_mtx()
        self.dist = calibration_config.get_left_dist()
        self.nmtx = calibration_config.get_left_nmtx()
        self.roi = calibration_config.get_left_roi()


    def calculate_homography_matrix(self):
        self.H = calibration.find_homography(self.start_frame)


    def set_capture_config(self):
        #TODO попробовать
        self.cap.set(cv.CAP_PROP_FPS, 5)
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 0)


    def prepocess(self):
        for func in self.preprocs:
            func()


    def process_frame(self):
        for func in self.rt_procs:
            self.cur_frame = func(self.cur_frame)


    def get_yolo_coords(self, frame):
        results = self.model(frame)
        names = results[0].names
        self.yolo_res = [
            [tuple(int(c) for c in results[0].boxes.xyxy[i]), 
            tuple(int(c) for c in results[0].boxes.xywh[i][:2]), 
            str(names[int(results[0].boxes.cls[i])]), 
            float(results[0].boxes.conf[i])
            ] for i in range(results[0].boxes.shape[0])]
        return frame

    
    def get_yolo_coords_on_start(self):
        self.get_yolo_coords(self.start_frame)
    

    def get_base_coord(self, frame):
        lower_hsv, upper_hsv = calibration_config.get_red_base_hsv() if self.color == Color.Red else calibration_config.get_green_base_hsv()
        return find_biggest_object_center_by_hsv_filters(frame, lower_hsv, upper_hsv)

    
    def get_base_coord_on_start(self):
        self.get_base_coord(self.start_frame)


    def update_vector(self, frame):
        x, y = self.vector
        print(x,y, x * x + y * y)
        if x * x + y * y < 100:
            
            self.get_constant_vector(frame)
            self.vector = self.constant_vector
        return frame


    def get_constant_vector(self, frame):
        robor_center = self.robor[1]
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_frame, np.array([0,0, 245]), np.array([180, 18, 255]))
        # применяем фильтр
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        near_counter = []
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # cv.imshow("mask_", mask)
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 10:
                x, y, w, h = cv.boundingRect(contour)
                center_counter = (x + w//2 , y + h//2)
                x1, y1, x2, y2 = self.robor[0]
                if x1 - 10 > center_counter[0] or center_counter[0] > x2 + 10 or y1 - 10 > center_counter[1] or center_counter[1] > y2 + 10:
                    continue
                distance = ((center_counter[0] - robor_center[0])**2 + (center_counter[1] - robor_center[1])**2)**0.5
                if not near_counter:
                    near_counter = [(center_counter[0], center_counter[1]), distance]
                else:
                    if near_counter[1] > distance:
                        near_counter = [(center_counter[0], center_counter[1]), distance]
        if len(near_counter) == 0:
            return frame
        print(near_counter)
        next_dot = (2*robor_center[0] - near_counter[0][0], 2*robor_center[1] - near_counter[0][1])
        self.constant_vector = (next_dot[0] - self.robor[1][0], next_dot[1] - self.robor[1][1])
        return frame
    

    def find_robor(self, frame):
        prev_frame_robor = self.robor
        robots = []
        for (x1,y1,x2,y2), (cx, cy), name, conf in self.yolo_res:
            if name != "robor":
                continue
            robots.append([(x1,y1,x2,y2), (cx, cy), name, conf])
        closest = []
        closest_dst = 10000000
        x, y = prev_frame_robor[1]
        for robot in robots:
            dst = get_distance((x,y), robot[1])
            if dst < closest_dst:
                closest_dst = dst
                closest = robot
        
        #if closest is too far stay on same coords
        x1, y1, x2, y2 = self.robor[0]
        if closest_dst > 2 * get_distance((x1,y1), (x2,y2)):
            return
        self.robor = closest
        self.vector = (self.robor[1][0] - prev_frame_robor[1][0], self.robor[1][1] - prev_frame_robor[1][1])
        self.update_vector(frame)
        # cv.rectangle(frame,(x1, y1), (x2, y2), 255, 5)
        return frame
        # print("INFO: searching for robots")
        # robor_hsv = calibration_config.get_red_robor_hsv() if self.color == Color.Red else calibration_config.get_green_robor_hsv()
        # print(f"INFO: robor hsv {robor_hsv}")
        # light_x, light_y = find_biggest_object_center_by_hsv_filters(frame, *robor_hsv)
        # if light_x == -1:
        #     print("INFO: cant find robot's lights")
        #     return frame
        # print(light_x, light_y)
        # robots = []
        # cnt = 0

        # for (x1,y1,x2,y2), (cx, cy), name, conf in self.yolo_res:
        #     print(name)
        #     if name != "robor":
        #         continue
        #     cnt += 1
        #     print(x1, y1, x2, y2, cx, cy)
        #     #TODO фильтр по conf
        #     #TODO мб добавить погрешность, надо потестить
        #     if x1 - 5 < light_x and light_x < x2 + 5 and y1 - 5 < light_y < y2 + 5:
        #         print(f"INFO: robor found on frame at ({cx}, {cy})")
        #         self.robor = [(x1,y1,x2,y2), (cx, cy), conf]
        #         self.prev_center = self.cur_center
        #         self.cur_center = (cx,cy)
        #         self.calculate_orientation((light_x, light_y))
        #     else:
        #         robots.append([(x1,y1,x2,y2), (cx, cy), name, conf])
        
        # if len(robots) == 0 and cnt == 0:
        #     print("INFO: no robots found on frame")
        # elif len(robots) == 0:
        #     print("INFO: cant find opponent on frame")
        # elif len(robots) == cnt:
        #     print("INFO: cant find robor on frame")
        # elif len(robots) == 1:
        #     print(f"INFO: found opponent on frame at {robots[0][1]}")
        #     self.opponent = robots[0]
        # else:
        #     print("INFO: found more than 2 robots on frame")
        #     #TODO add fix when robots splits into several parts
        # return frame #for consistency


    def find_robor_on_start(self):
        robots = []
        for (x1,y1,x2,y2), (cx, cy), name, conf in self.yolo_res:
            if name != "robor":
                continue
            robots.append([(x1,y1,x2,y2), (cx, cy), name, conf])
        closest = []
        closest_dst = 10000000
        x, y = (0, 620) if self.pos == 0 else (800, 0) if self.pos == 1 else (800, 620)
        self.constant_vector = (0, 20) if self.pos == 0 else (0, -20) if self.pos == 1 else (0,20)
        for robot in robots:
            dst = get_distance((x,y), robot[1])
            if dst < closest_dst:
                closest_dst = dst
                closest = robot
        self.robor = closest
        #в данный момент забили хуй на противника
    
    
    def calculate_orientation(self, light_center):
        robor_center = self.robor[1]
        self.orientation_vector = (robor_center[0] - light_center[0], robor_center[1] - light_center[1])

    
    def draw_vector(self, frame):
        #TODO check for edges как же похуй
        copy = frame.copy()
        cv.arrowedLine(copy, self.robor[1], (self.robor[1][0] + self.vector[0], self.robor[1][1] + self.vector[1]),(255,255,0),3)
        return copy


    def find_inner_walls(self):
        if self.check_for_wall(self.map_graph[7], self.map_graph[12]):
            print("INFO: found wall")
            self.map_graph[7].remove_neighbour(12)
            self.map_graph[22].remove_neighbour(27)
            self.map_graph[12].remove_neighbour(7)
            self.map_graph[27].remove_neighbour(22)
        
        elif self.check_for_wall(self.map_graph[15], self.map_graph[16]):
            print("INFO: found wall")
            self.map_graph[15].remove_neighbour(16)
            self.map_graph[16].remove_neighbour(15)
            self.map_graph[18].remove_neighbour(19)
            self.map_graph[19].remove_neighbour(18)


    def check_for_wall(self, node1, node2):       
        img = cv.medianBlur(self.start_frame, 5)
        _, th1 = cv.threshold(img,110,255,cv.THRESH_BINARY_INV)
        print(node1.x, node1.y, node2.x, node2.y)
        if node1.x == node2.x:
            x = node1.x
            s, e = min(node1.y, node2.y), max(node1.y, node2.y)
            for j in range(s, e):
                if all(th1[j, x] == [255,255,255]):
                    return True
        if node1.y == node2.y:
            y = node1.y
            s, e = min(node1.x, node2.x), max(node1.x, node2.x)
            for j in range(s, e):
                if all(th1[y, j] == [255,255,255]):
                    return True
        # cv.imshow("th", th1)
        return False
    

    def find_base_node(self):
        print("INFO: start searching for base node")
        lower_hsv_red, upper_hsv_red = calibration_config.get_red_base_hsv() 
        lower_hsv_green, upper_hsv_green = calibration_config.get_green_base_hsv()
        red_base =  find_biggest_object_center_by_hsv_filters(self.start_frame, lower_hsv_red, upper_hsv_red)
        green_base =  find_biggest_object_center_by_hsv_filters(self.start_frame, lower_hsv_green, upper_hsv_green)
        red_base_node, _ = mp.find_closest_node(self.map_graph,*red_base)
        green_base_node, _ = mp.find_closest_node(self.map_graph,*green_base)
        print(f"INFO: found red base at {red_base}, node = {red_base_node.id}")
        print(f"INFO: found green base at {green_base}, node = {green_base_node.id}")
        if green_base[0] >=0 and green_base_node.id == 2:
            self.buttons1_node = self.map_graph[19]
            self.buttons2_node = self.map_graph[15]
            self.base_node = self.map_graph[2] if self.color == Color.Green else self.map_graph[32]
            self.opponent_base_node = self.map_graph[32] if self.color == Color.Green else self.map_graph[2]
        
        elif green_base[0] >=0 and green_base_node.id == 32:
            self.buttons1_node = self.map_graph[19]
            self.buttons2_node = self.map_graph[15]
            self.base_node = self.map_graph[32] if self.color == Color.Green else self.map_graph[2]
            self.opponent_base_node = self.map_graph[2] if self.color == Color.Green else self.map_graph[32]
        
        elif green_base[0] >=0 and green_base_node.id == 15:
            self.buttons1_node = self.map_graph[2]
            self.buttons2_node = self.map_graph[32]
            self.base_node = self.map_graph[15] if self.color == Color.Green else self.map_graph[19]
            self.opponent_base_node = self.map_graph[19] if self.color == Color.Green else self.map_graph[15]
        
        elif green_base[0] >=0 and green_base_node.id == 19:
            self.buttons1_node = self.map_graph[2]
            self.buttons2_node = self.map_graph[32]
            self.base_node = self.map_graph[19] if self.color == Color.Green else self.map_graph[15]
            self.opponent_base_node = self.map_graph[15] if self.color == Color.Green else self.map_graph[19]
        
        if green_base[0] >=0 and red_base_node.id == 2:
            self.buttons1_node = self.map_graph[19]
            self.buttons2_node = self.map_graph[15]
            self.base_node = self.map_graph[2] if self.color == Color.Red else self.map_graph[32]
            self.opponent_base_node = self.map_graph[32] if self.color == Color.Red else self.map_graph[2]
        
        elif green_base[0] >=0 and red_base_node.id == 32:
            self.buttons1_node = self.map_graph[19]
            self.buttons2_node = self.map_graph[15]
            self.base_node = self.map_graph[32] if self.color == Color.Red else self.map_graph[2]
            self.opponent_base_node = self.map_graph[2] if self.color == Color.Red else self.map_graph[32]
        
        elif green_base[0] >=0 and red_base_node.id == 15:
            self.buttons1_node = self.map_graph[2]
            self.buttons2_node = self.map_graph[32]
            self.base_node = self.map_graph[15] if self.color == Color.Red else self.map_graph[19]
            self.opponent_base_node = self.map_graph[19] if self.color == Color.Red else self.map_graph[15]
        
        elif green_base[0] >=0 and red_base_node.id == 19:
            self.buttons1_node = self.map_graph[2]
            self.buttons2_node = self.map_graph[32]
            self.base_node = self.map_graph[19] if self.color == Color.Red else self.map_graph[15]
            self.opponent_base_node = self.map_graph[15] if self.color == Color.Red else self.map_graph[19]
        if green_base[0] == -1 and red_base[0] == -1:
            #ПИЗДЕЦ
            return
        self.buttons1_node.add_feature("buttons")
        self.buttons2_node.add_feature("buttons")
        
        if self.color == Color.Red:
            self.base_node.add_feature("red_base")
            self.opponent_base_node.add_feature("green_base")
        else:
            self.base_node.add_feature("green_base")
            self.opponent_base_node.add_feature("red_base")
 

    def find_current_node(self, frame):
        print(f"INFO: find_current_node robor {self.robor}")
        if self.robor == []:
            return frame
        print(f"INFO: find_current_node prev node {self.current_node}")
        if self.current_node is not None:
            self.current_node.remove_feature("robor")
        self.current_node, _ = mp.find_closest_node(self.map_graph,*self.robor[1])
        print(f"INFO: find_current_node new node {self.current_node}")
        self.current_node.add_feature("robor")

        return frame # for consistency

    
    def find_current_node_on_start(self):
        self.find_current_node(self.start_frame)


    def draw_graph(self, frame):
        if not self.target_node:
            if self.cubes1_node is not None:
                self.target_node = self.cubes1_node
            else:
                self.target_node = self.base_node
        self.path = mp.find_path(self.map_graph, self.current_node, self.target_node)
        return mp.draw_graph(frame, self.map_graph, self.path)


    # Le Govnocode
    def total_shit(self, frame):
        if self.busy and (self.target_proc is None or not self.target_proc.is_alive()):
            self.busy = False
            self.target_proc = None
            self.we_take_cube = not self.we_take_cube
        if self.we_take_cube == False:
            if self.cubes1_node is not None:
                self.target_node = self.cubes1_node
            else:
                self.target_node = self.base_node
        else:
            self.target_node = self.base_node
        self.path = mp.find_path(self.map_graph, self.current_node, self.target_node)
        for i, node in enumerate(self.path):
            cv.circle(frame, (node.x, node.y), 5, (0, 255, 0), 5)
            if i > 0:
                cv.line(frame, (node.x, node.y), (self.path[i - 1].x, self.path[i - 1].y), (0, 255, 0), 2)
        if len(self.path) <= 2:
            print("[INFO] FUCK YEAH! WE ARE HERE!")
            if self.busy:
                print(self.target_proc.is_alive())
            if self.target_node.id == self.base_node.id and not self.we_take_cube:
                pass
                #стоять как блять???
            elif not self.busy and (self.target_proc is None or not self.target_proc.is_alive()):
                if self.we_take_cube:
                    self.target_proc = Process(target=find_target, args=("basket", ))
                    self.target_proc.start()
                    self.busy = True
                else:
                    self.target_proc = Process(target=find_target, args=("cube", ))
                    self.target_proc.start()
                    self.busy = True
                # yolo_object_detection_process = subprocess.call([sys.executable, "yolo_object_detection.py"])
            elif self.busy and (self.target_proc is None or not self.target_proc.is_alive()):
                self.busy = False
                self.target_proc = None
                self.we_take_cube = True
        else:
            print("[INFO] HOW DO I DRIVE ROBOR?")
            #А я ебу?
            dist, angle = self.get_errors()
            print("error",angle,dist)
            message = f"{0},{0},{angle},{dist},{0},{0}".encode()  # Format: "e_x,e_y"
            self.sock.sendto(message, (self.robot_ip, self.robot_port))

        return frame


    def find_optimal_way_to(self, frame, target):
        self.path = mp.find_path(self.map_graph, self.current_node, target)
        frame = mp.draw_path(frame, self.path)
        return frame
        
    
    def find_cubes(self, frame):
        cubes = []
        not_sure = []
        self.inside_robor = False
        self.inside_opponent = False
        if len(self.robor):
            r_x1,r_y1,r_x2,r_y2 = self.robor[0]
        if len(self.opponent):
            o_x1,o_y1,o_x2,o_y2 = self.opponent[0]
        print(self.yolo_res)
        for (x1,y1,x2,y2), (cx, cy), name, conf in self.yolo_res:
            if name == "cube":
                if conf > 0.5:
                    cubes.append((cx, cy))
                    print(f"INFO: found cube at ({cx}, {cy})")
                else:
                    not_sure.append((cx, cy))
                    print(f"INFO: cube found at ({cx}, {cy}), but conf is low")
                if len(self.robor):
                    if r_x1 > cx and cx > r_x2 and r_y1 > cy > r_y2:
                        self.inside_robor = True
                if len(self.opponent):
                    if o_x1 > cx and cx > o_x2 and o_y1 > cy > o_y2:
                        self.inside_opponent = True
        if len(cubes) < 2:
            cubes += not_sure
        closest = sorted([(mp.find_closest_node(self.map_graph, *cube)) for cube in cubes], key=lambda x: x[1])
        
        if len(cubes) > 0:
            self.cubes1_node = closest[0][0]
            self.cubes1_node.add_feature("cube")
        
        if len(cubes) > 1:
            if closest[0][0].id == self.base_node.id:
                self.cubes1_node = closest[1][0]
                self.cubes1_node.add_feature("cube")
            else:
                self.cubes2_node = closest[1][0]
                self.cubes2_node.add_feature("cube")

        return frame # for consistency


    def set_ball_on_graph(self):
        self.ball_node = self.map_graph[17]
        self.ball_node.add_feature("ball")


    def remove_ball_from_graph(self):
        #TODO думаю по цвету искать его будет запарно, удалять будем если мы за ним заехали или если соперник за ним заехал
        pass

    
    def get_errors(self):
        end_pt = (self.path[1].x, self.path[1].y)
        # angle = get_angle_of_robor_movement(self.cur_center, self.prev_center, end_pt)
        B = self.robor[1]
        A = (B[0] + self.vector[0], B[1] + self.vector[1])
        C = (self.path[1].x, self.path[1].y)
        angle = get_angle(A, B, C)
        dist = get_distance(self.robor[1], end_pt)
        return dist, angle
    
        if get_distance(self.prev_center, self.cur_center):
            BA = self.orientation_vector
            BC = (end_pt[0] - self.cur_center[0], end_pt[1] - self.cur_center[1])
            angle_rad = math.atan2(BC[1], BC[0]) - math.atan2(BA[1], BA[0])
            return dist, angle_rad
        return dist, angle
    
    
def get_angle_of_robor_movement(robor_center, prev_center, end_pt):
    #TODO fix
    A = (2 * robor_center[0] - prev_center[0], 2 * robor_center[1] - prev_center[1])
    B = robor_center
    C = end_pt
    return get_angle(A, B, C)


def get_angle(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    angle_rad = math.atan2(BC[1], BC[0]) - math.atan2(BA[1], BA[0])
    print(angle_rad)
    
    return angle_rad


def get_distance(start_pt, end_pt):
    return int(((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)**.5)


def get_errors(frame, robor_center, prev_center):
    end_pt = (105, 500)
    e1 = get_angle(robor_center, prev_center, end_pt)
    e2 = get_distance(robor_center, end_pt)
    return e1,e2


def find_biggest_object_center_by_hsv_filters(frame, lower_hsv, upper_hsv, gaussian_blur=False):
    print(f"INFO: lower hsv = {lower_hsv}, upper_hsv = {upper_hsv}")
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_frame, lower_hsv, upper_hsv)
    # cv.imshow("mask", mask)
    # cv.waitKey(0)
    if gaussian_blur:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    print(f"INFO: contours cnt {len(contours)}")
    if len(contours) == 0:
        return -1, -1
    biggest = max(contours, key=cv.contourArea)    
    (x,y,w,h) = cv.boundingRect(biggest)

    return (x + w//2, y + h//2)


def main():
    # im = cv.imread("out/both.png")
    # cv.imshow("Frame",im)
    # while True:
    #     k = cv.waitKey(25)
    #     if  k & 0xFF == ord('q'):
    #         break
                    
    CameraProcessing(Color.Green, "data/green_lights.avi", pos=2).start()
    #короче считаем pos = 0 - мы снизу слева, смотрим вверх
    #короче считаем pos = 1 - мы сверху справа, смотрим вниз
    #короче считаем pos = 2 - мы снизу справа, (для видоса который у нас есть)
    

if __name__ == "__main__":
    main()