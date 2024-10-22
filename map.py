import cv2 as cv
import heapq

class MapNode():
    x = 0
    y = 0
    id = 0 
    neighbours: dict
    features: set
    
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.neighbours = dict()
        self.features = set()
    
    def add_neighbour(self, id, w):
        self.neighbours[id] = w

    def remove_neighbour(self, id):
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

    def remove_feature(self, feature): #удаляем 
        self.features.remove(feature)

    def find_way(self, end_pt):
        #TODO хочется учитывать положение соперника 
        pass


def generate_graph():
    graph = [
        MapNode( 57,  62,  0),
        MapNode( 57, 161,  1),
        MapNode( 57, 310,  2),
        MapNode( 57, 459,  3),
        MapNode( 57, 558,  4),
        MapNode(136,  62,  5),
        MapNode(136, 161,  6),
        MapNode(136, 310,  7),
        MapNode(136, 459,  8),
        MapNode(136, 558,  9),
        MapNode(260,  62, 10),
        MapNode(260, 161, 11),
        MapNode(260, 310, 12),
        MapNode(260, 459, 13),
        MapNode(260, 558, 14),
        MapNode(400,  72, 15),
        MapNode(400, 161, 16),
        MapNode(400, 310, 17),
        MapNode(400, 459, 18),
        MapNode(400, 548, 19),
        MapNode(539,  62, 20),
        MapNode(539, 161, 21),
        MapNode(539, 310, 22),
        MapNode(539, 459, 23),
        MapNode(539, 558, 24),
        MapNode(663,  62, 25),
        MapNode(663, 161, 26),
        MapNode(663, 310, 27),
        MapNode(663, 459, 28),
        MapNode(663, 558, 29),
        MapNode(742,  62, 30),
        MapNode(742, 161, 31),
        MapNode(742, 310, 32),
        MapNode(742, 459, 33),
        MapNode(742, 558, 34)
    ]

    def add_neighbours(a: MapNode, b: MapNode) -> None:
        distance = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)
        a.add_neighbour(b.id, distance)
        b.add_neighbour(a.id, distance) 

    add_neighbours(graph[0], graph[1])
    add_neighbours(graph[0], graph[5])
    add_neighbours(graph[0], graph[6])

    add_neighbours(graph[1], graph[2])
    add_neighbours(graph[1], graph[5])
    add_neighbours(graph[1], graph[6])
    add_neighbours(graph[1], graph[7])

    add_neighbours(graph[2], graph[3])
    add_neighbours(graph[2], graph[6])
    add_neighbours(graph[2], graph[7])
    add_neighbours(graph[2], graph[8])

    add_neighbours(graph[3], graph[4])
    add_neighbours(graph[3], graph[7])
    add_neighbours(graph[3], graph[8])
    add_neighbours(graph[3], graph[9])

    add_neighbours(graph[4], graph[8])
    add_neighbours(graph[4], graph[9])


    add_neighbours(graph[5], graph[10])
    add_neighbours(graph[5], graph[6])

    add_neighbours(graph[6], graph[7])

    add_neighbours(graph[7], graph[8])

    add_neighbours(graph[8], graph[9])

    add_neighbours(graph[10], graph[15])

    add_neighbours(graph[11], graph[12])
    add_neighbours(graph[11], graph[16])

    add_neighbours(graph[12], graph[13])

    add_neighbours(graph[13], graph[18])

    add_neighbours(graph[18], graph[23])

    add_neighbours(graph[22], graph[23])

    add_neighbours(graph[21], graph[22])

    add_neighbours(graph[16], graph[21])

    add_neighbours(graph[9], graph[14])

    add_neighbours(graph[15], graph[20])

    add_neighbours(graph[14], graph[19])

    add_neighbours(graph[19], graph[24])

    add_neighbours(graph[20], graph[25])

    add_neighbours(graph[25], graph[30])
    add_neighbours(graph[25], graph[26])
    add_neighbours(graph[25], graph[31])

    add_neighbours(graph[30], graph[31])

    add_neighbours(graph[26], graph[31])
    add_neighbours(graph[26], graph[27])
    add_neighbours(graph[26], graph[32])


    add_neighbours(graph[27], graph[31])
    add_neighbours(graph[31], graph[32])

    add_neighbours(graph[27], graph[32])
    add_neighbours(graph[27], graph[28])

    add_neighbours(graph[27], graph[33])

    add_neighbours(graph[28], graph[32])
    add_neighbours(graph[32], graph[33])

    add_neighbours(graph[28], graph[33])

    add_neighbours(graph[28], graph[34])

    add_neighbours(graph[28], graph[29])

    add_neighbours(graph[29], graph[33])
    add_neighbours(graph[29], graph[34])

    add_neighbours(graph[33], graph[34])

    add_neighbours(graph[24], graph[29])

    add_neighbours(graph[26], graph[30])

    ### ZALUPA NEGRA ###
    add_neighbours(graph[7], graph[12])
    add_neighbours(graph[15], graph[16])
    add_neighbours(graph[18], graph[19])
    add_neighbours(graph[22], graph[27])
    add_neighbours(graph[12], graph[17])
    add_neighbours(graph[16], graph[17])
    add_neighbours(graph[17], graph[18])
    add_neighbours(graph[17], graph[22])

    return graph


def opposite_side(id):
    if id == 2:
        return 32
    if id == 32:
        return 2
    if id == 15:
        return 19
    if id == 19:
        return 15


def find_closest_node(graph, x, y):
    closest = None
    distance = 1000000
    for node in graph:
        d = int(((node.x - x) ** 2 + (node.y - y) ** 2) ** 0.5)
        closest, distance = (node, d) if d < distance else (closest, distance)
    return closest, distance


def find_path(graph: list[MapNode], start: MapNode, end: MapNode) -> list[MapNode]:
    distances = {node.id: float('inf') for node in graph}
    distances[start.id] = 0

    previous_nodes = {node.id: None for node in graph}

    priority_queue = [(0, start.id)]

    while priority_queue:
        current_distance, current_node_id = heapq.heappop(priority_queue)

        if current_node_id == end.id:
            break

        current_node = graph[current_node_id]

        for neighbour_id, weight in current_node.neighbours.items():
            distance = current_distance + weight

            if distance < distances[neighbour_id]:
                distances[neighbour_id] = distance
                previous_nodes[neighbour_id] = current_node_id
                heapq.heappush(priority_queue, (distance, neighbour_id))

    path = []
    current_node_id = end.id
    while current_node_id is not None:
        path.append(graph[current_node_id])
        current_node_id = previous_nodes[current_node_id]

    path.reverse()

    return path


def draw_graph(frame, graph, path = None):
    img = frame.copy()
    for node in graph:
        color = 255
        if node.has_base_feature("red"):
            color = (0,0,255)
        if node.has_base_feature("green"):
            color = (0,255,0)
        if node.has_buttons_feature():
            color = (102,255,178)
        if node.has_cube_feature():
            color = (51,153,255)
        if node.has_ball_feature():
            color = (51,255,255)
        
        cv.circle(img, (node.x, node.y), 5, color, 5)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, str(node.id), (node.x, node.y), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        for neighbor in node.neighbours:
            cv.line(img, (node.x, node.y), (graph[neighbor].x, graph[neighbor].y), 255, 2)
    for i, node in enumerate(path):
        cv.circle(img, (node.x, node.y), 5, (0, 255, 0), 5)
        if i > 0:
            cv.line(img, (node.x, node.y), (path[i - 1].x, path[i - 1].y), (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    img = cv.imread("out/frame0.png")
    graph = generate_graph()
    for node in graph:
        cv.circle(img, (node.x, node.y), 5, 255, 5)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, str(node.id), (node.x, node.y), font, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        for neighbor in node.neighbours:
            cv.line(img, (node.x, node.y), (graph[neighbor].x, graph[neighbor].y), 255, 2)
    cv.imshow("g", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
