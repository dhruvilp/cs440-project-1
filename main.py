import pygame
from pygame.locals import *
import threading
import argparse
from random import *
import sys
import re
from time import sleep, monotonic
from heap import heapify, heap_pop, heap_push

block_width = 7  # dimensions of block
GridCols = 101  # of columns
GridRows = 101  # of rows
result = ""


def result_store(text):
    global result
    result += text + "\n"


def manhattan_dist(start, target): return abs(target.x - start.x) + abs(target.y - start.y)


def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None: seen = set()
    obj_id = id(obj)
    if obj_id in seen: return 0
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for member in obj.__dict__.values(): size += get_size(member, seen)
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))
    return size


class Node:
    START = 0b00000100
    TARGET = 0b00001000
    COLOR = 0b11100000
    VISITED = 0b00000001
    BLOCKED = 0b00000010
    CLOSED = 0b00010000

    colors = (None, (255, 228, 98), (80, 80, 120))

    tie_breaker = lambda x, y: x.g() > y.g()

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self._search = 0
        self._g = sys.maxsize
        self._h = sys.maxsize
        self.flags = 0

    def color(self, c_val=None):
        if c_val is not None:
            self.flags &= ~Node.COLOR
            self.flags |= c_val << 5
        else:
            col = (self.flags & Node.COLOR) >> 5
            if col == 7: return 146, 175, 227
            return Node.colors[col]

    def start(self):
        return bool(self.flags & Node.START)

    def temp_start(self):
        self.flags |= Node.START

    def target(self):
        return bool(self.flags & Node.TARGET)

    def temp_target(self):
        self.flags |= Node.TARGET

    def close(self):
        self.flags |= Node.CLOSED

    def closed(self):
        return bool(self.flags & Node.CLOSED)

    def h(self, h_val=None):
        if h_val is not None:
            self._h = h_val
        else:
            return self._h

    def f(self):
        return self.g() + self.h()

    def g(self, g_val=None):
        if g_val is not None:
            self._g = g_val
        else:
            return self._g

    def reset(self):
        self.flags &= ~Node.VISITED
        self._g = sys.maxsize
        self._h = sys.maxsize
        self._search = 0
        self.parent = None

    def search(self, s_val=None):
        if s_val is not None:
            self._search = s_val
        else:
            return self._search

    def block(self):
        self.flags |= Node.BLOCKED

    def blocked(self):
        return bool(self.flags & Node.BLOCKED)

    def unblock(self):
        self.flags &= ~Node.BLOCKED

    def visit(self):
        self.flags |= Node.VISITED

    def visited(self):
        return bool(self.flags & Node.VISITED)

    def walkable(self):
        return not self.visited() or (self.visited() and not self.blocked())

    def __eq__(self, other):
        if isinstance(other, Node): return self.x == other.x and self.y == other.y
        return False

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if isinstance(other, Node):
            if self.f() == other.f(): return Node.tie_breaker(self, other)
            return self.f() < other.f()
        raise ValueError("'%s'" % str(type(other)))

    def __repr__(self):
        return "Node<%d, %d>" % (self.x, self.y)

    def __hash__(self):
        return hash(str(self))


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.maze = tuple(tuple((Node(x, y) for x in range(cols))) for y in range(rows))

    def node(self, x, y):
        if x < 0 or x >= self.rows: return None
        if y < 0 or y >= self.cols: return None
        return self.maze[y][x]

    def neighbors(self, node):
        neighbors = (
            self.node(node.x, node.y - 1),
            self.node(node.x, node.y + 1),
            self.node(node.x - 1, node.y),
            self.node(node.x + 1, node.y))
        return tuple(filter(lambda n: n, neighbors))

    def down(self, node):
        return self.node(node.x, node.y + 1)

    def up(self, node):
        return self.node(node.x, node.y - 1)

    def left(self, node):
        return self.node(node.x - 1, node.y)

    def right(self, node):
        return self.node(node.x + 1, node.y)


class MazeBuilder:
    def __init__(self, rows, cols, maze_seed=None):
        if maze_seed is None: maze_seed = randrange(0, 10000)
        result_store("seed: %d" % maze_seed)
        seed(maze_seed)
        self.rows = rows
        self.cols = cols
        self.grid = Grid(rows, cols)
        self.limit = False

    @staticmethod
    def init_node(node):
        node.visit()
        if random() < 0.3:  # as asked in the assignment
            node.block()
            return False
        return True

    def build(self):
        for row in self.grid.maze:
            for node in row:
                node.block()
        backtrace = []
        start_x = randrange(0, self.cols)
        start_y = randrange(0, self.rows)
        finished = False
        sNode = self.grid.node(start_x, start_y)
        current = sNode
        current.unblock()
        clock = pygame.time.Clock()
        while not finished:
            if self.limit: clock.tick(100)
            extn_neighbors = (
                self.grid.node(current.x, current.y - 2),
                self.grid.node(current.x, current.y + 2),
                self.grid.node(current.x - 2, current.y),
                self.grid.node(current.x + 2, current.y))
            bridge = tuple(filter(lambda x: x and not x.blocked(), extn_neighbors))
            if len(bridge) > 0 and random() < 0.4:
                node = bridge[0]
                middle = self.grid.node(int((current.x + node.x) / 2), int((current.y + node.y) / 2))
                middle.unblock()
            extn_neighbors = tuple(filter(lambda x: x and x.blocked(), extn_neighbors))
            if len(extn_neighbors) == 0 and len(backtrace) == 0:
                finished = True
                continue
            elif len(extn_neighbors) == 0:
                current = backtrace.pop()
                continue
            node = extn_neighbors[randrange(0, len(extn_neighbors))]
            node.unblock()
            middle = self.grid.node(int((current.x + node.x) / 2), int((current.y + node.y) / 2))
            middle.unblock()
            backtrace.append(current)
            current = node
        return self.grid


class AIAgent:
    def __init__(self):
        self.counter = 0
        self.total = 0

    def run(self, grid, start, end):
        self.total = 0
        self.counter = 0
        agent = start
        agent.visit()
        final_path = set()
        start_time = monotonic()
        for neighbor in grid.neighbors(agent): neighbor.visit()
        while agent != end:
            self.counter += 1
            agent.g(0)
            agent.search(self.counter)
            end.g(sys.maxsize)
            end.search(self.counter)
            path = self.get_path(grid, agent, end)
            for node in final_path: node.color(1)
            if len(path) == 0: return False
            for node in path:
                if not node.walkable(): break
                final_path.add(node)
                agent = node
                agent.color(1)
                for neighbor in grid.neighbors(agent): neighbor.visit()
        end_time = monotonic()
        result_store("Nodes Visited by Agent. ==> %d" % len(final_path))
        result_store("Algorithm Expanded Nodes: ==> %d" % self.total)
        result_store("Visits/Iteration: ==> %d nodes" % (self.total / self.counter))
        result_store("Time(s): ==> %0.6f" % (end_time - start_time))

    @staticmethod
    def cost(next_b):
        if not next_b.walkable(): return sys.maxsize
        else: return 1

    def get_path(self, grid, target, open_list): raise NotImplemented()


class RepeatedForwardAStar(AIAgent):

    def get_path(self, grid, start, target):
        start.h(manhattan_dist(start, target))
        open_list = [start]
        path = []
        while target.g() > open_list[0].f():
            self.total += 1
            current = heap_pop(open_list)
            current.color(7)
            for neighbor in grid.neighbors(current):
                if neighbor.search() < self.counter:
                    neighbor.g(sys.maxsize)
                    neighbor.search(self.counter)
                if current.g() + self.cost(neighbor) < neighbor.g():
                    neighbor.g(current.g() + self.cost(neighbor))
                    neighbor.parent = current
                    if neighbor in open_list:
                        open_list.remove(neighbor)
                        heapify(open_list)
                    neighbor.h(manhattan_dist(neighbor, target))
                    heap_push(open_list, neighbor)
            if len(open_list) == 0: return ()
        current = target
        while current != start:
            path.append(current)
            current = current.parent
        path.reverse()
        return tuple(path)


class RepeatedBackwardAStar(RepeatedForwardAStar):
    def run(self, grid, start, target): super().run(grid, target, start)


class AdaptiveAStar(AIAgent):

    def get_path(self, grid, start, target):
        open_list = [start]
        closed = set()
        path = []
        start.h(manhattan_dist(start, target))
        while target.g() > open_list[0].f():
            self.total += 1
            current = heap_pop(open_list)
            closed.add(current)
            current.color(7)
            for neighbor in grid.neighbors(current):
                if neighbor.search() < self.counter:
                    neighbor.g(sys.maxsize)
                    neighbor.search(self.counter)
                if current.g() + self.cost(neighbor) < neighbor.g():
                    neighbor.g(current.g() + self.cost(neighbor))
                    neighbor.parent = current
                    if neighbor in open_list:
                        open_list.remove(neighbor)
                        heapify(open_list)
                    if neighbor.h() == sys.maxsize:
                        neighbor.h(manhattan_dist(neighbor, target))
                    heap_push(open_list, neighbor)
        for node in closed: node.h(target.g() - node.g())
        if len(open_list) == 0: return ()
        current = target
        while current != start:
            path.append(current)
            current = current.parent
        path.reverse()
        return tuple(path)


class PygameHandler:
    FRAME_RATE = 30

    def __init__(self):
        super().__init__()
        pygame.init()
        pygame.display.set_caption('AI Maze Runner')
        self.GameScreen = pygame.display.set_mode((GridCols * block_width + 25, GridRows * block_width + 25))
        self.GridSurface = pygame.Surface(self.GameScreen.get_size())
        self.myfont = pygame.font.Font("SourceSansPro-Regular.ttf", 14)
        self.clock = pygame.time.Clock()
        self.hide_unvisited = False
        self.running = False
        self.grid = None
        self.info = None

    def run(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.stop()
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE: self.stop()
            self.update()
            self.clock.tick(self.FRAME_RATE)
        pygame.quit()
        quit()

    def stop(self):
        self.running = False

    def update(self):
        self.draw_maze()
        self.GameScreen.blit(self.GridSurface, (0, 0))
        pygame.display.update()

    def draw_maze(self):
        if not self.grid: return
        self.GridSurface.fill((255, 255, 255))
        for y in range(self.grid.rows):
            for x in range(self.grid.cols):
                node = self.grid.node(x, y)
                icolor = (255, 255, 255)
                if node.blocked() and not self.hide_unvisited: icolor = (47, 48, 97)
                elif not node.walkable(): icolor = (0, 0, 0)
                elif node.start(): icolor = (255, 91, 219)
                elif node.target(): icolor = (73, 255, 142)
                elif not node.visited() and self.hide_unvisited: icolor = (40, 40, 40)
                elif node.color(): icolor = node.color()
                pygame.draw.rect(self.GridSurface, icolor, (node.x * block_width + 10, node.y * block_width + 10, block_width, block_width), 0)


class ProcessingThread(threading.Thread):

    def __init__(self, algorithm, pyhandler=None):
        super().__init__()
        self.sNode = None
        self.eNode = None
        self.algorithm = algorithm
        self.pyhandler = pyhandler

    def run(self):
        if threading.current_thread().getName() != "__main__":
            sleep(1)
        runs = 1
        for i in range(runs):
            seedVal = None
            builder = MazeBuilder(GridRows, GridCols, seedVal)
            if self.pyhandler: self.pyhandler.grid = builder.grid
            maze = builder.build()
            half_h = int(maze.rows / 2)
            start_half = randrange(0, 2)
            end_half = 1 - start_half
            if self.sNode is not None and self.eNode is not None:
                self.sNode = maze.node(self.sNode[0], self.sNode[1])
                self.sNode.unblock()
                self.sNode.visited()
                self.eNode = maze.node(self.eNode[0], self.eNode[1])
                self.eNode.unblock()
                self.eNode.visited()
            while self.sNode is None:
                node = maze.node(randrange(0, int(maze.cols / 4)), randrange(start_half * half_h, (start_half + 1) * half_h))
                if not node.blocked(): self.sNode = node

            while self.eNode is None:
                node = maze.node(randrange(int(3 * maze.cols / 4), maze.cols), randrange(end_half * half_h, (end_half + 1) * half_h))
                if not node.blocked(): self.eNode = node

            result_store(str(self.sNode) + " " + str(self.eNode))
            self.sNode.temp_start()
            self.eNode.temp_target()
            self.algorithm.run(maze, self.sNode, self.eNode)
            for row in maze.maze:
                for node in row:
                    node.h(sys.maxsize)
                    node.g(sys.maxsize)
                    node.search(0)
            new_maze = MazeBuilder(GridRows, GridCols, seedVal).build()
            sn_op = new_maze.node(self.sNode.x, self.sNode.y)
            gn_op = new_maze.node(self.eNode.x, self.eNode.y)
            sn_op.g(0)
            sn_op.search(1)
            gn_op.h(0)
            gn_op.search(1)
            self.sNode = None
            self.eNode = None
        parse_result(result.split("\n"))


def parse_result(text):
    data = ""
    for line in text:
        data += line
    algo_parser = \
        re.compile(
            r'seed:\s+(\d+).*?(\d+),\s(\d+).*?(\d+),\s(\d+).*?(\d+).*?(\d+).*?(\d+)'
            r'.*?(\d+).*?(\d+\.\d+).*?(\d+).*?(\d+)')
    rounds_info = algo_parser.finditer(data)
    for info in rounds_info:
        print(f'Start(x,y) ==> ({info.group(2)},{info.group(3)})')
        print(f'Target(x,y) ==> ({info.group(4)},{info.group(5)})')
        print(f'Nodes Visited by Agent. ==> {info.group(8)}')
        print(f'Nodes Visited by Algo. ==> {info.group(6)}')
        print(f'Visits/Iteration ==> {info.group(9)}')
        print(f'Time(s) ==> {info.group(10)}')


def main():
    parser = argparse.ArgumentParser(description="CS440 Proj1")
    parser.add_argument("algo", type=int, choices=[1, 2, 3], help="1. A* forward, 2. A* backward, 3. Adaptive A*")
    args = vars(parser.parse_args())
    algo = args['algo']
    algo_array = [RepeatedForwardAStar(), RepeatedBackwardAStar(), AdaptiveAStar()]
    algo = algo_array[algo - 1]

    threading.current_thread().setName("__main__")
    pygame_handler = PygameHandler()
    p_t = ProcessingThread(algo, pygame_handler)
    p_t.daemon = True
    p_t.start()
    pygame_handler.run()


if __name__ == "__main__":
    main()
