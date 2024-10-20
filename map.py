from enum import Enum
from PIL import Image
import numpy as np
import map_config

graph = []


class MapNode():
    x = 0
    y = 0
    id = 0 
    neighbours = dict()
    features = set()
    
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
    
    def add_neighbour(self, id, w):
        self.add_neighbour[id] = w

    def remove_heighbour(self, id):
        del self.neighbours[id]

    def has_base_feature(self, color):
        return f"{color}_base" in self.features

    def has_buttons_feature(self):
        return "buttons" in self.features

    def has_cube_feature(self):
        return "cube" in self.features
    
    def has_ball_feature(self):
        return "ball" in self.features

    def has_robor_feature(self):
        return "robor" in self.features

    def add_feature(self, feature):
        self.features.add(feature)

    def find_way(self, end_pt):
        pass


def generate_graph():
    pass


def find_closest_node(x,y):
    closest = None
    distance = 1000000
    for node in graph:
        d = int(((node.x - x) ** 2 + (node.y - y) ** 2) ** 0.5)
        closest, distance = node, d if d < distance else closest, distance
    return closest, distance