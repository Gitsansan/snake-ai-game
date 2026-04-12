import random
from collections import deque
import numpy as np

GRID_SIZE = 10

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = (1, 0)  # right
        self.head = [GRID_SIZE//2, GRID_SIZE//2]
        self.snake = deque([self.head.copy()])
        self.spawn_food()
        self.score = 0
        self.frame = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
            if self.food not in self.snake:
                break

    def step(self, action):
        self.frame += 1

        self.move(action)
        self.snake.appendleft(self.head.copy())

        reward = 0
        done = False

        if self.is_collision():
            return self.get_state(), -10, True

        if self.head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, done

    def is_collision(self):
        x, y = self.head
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return True
        if self.head in list(self.snake)[1:]:
            return True
        return False

    def move(self, action):
        # action: [straight, right turn, left turn]
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        idx = directions.index(self.direction)

        if np.array_equal(action, [1,0,0]):  # straight
            new_dir = directions[idx]
        elif np.array_equal(action, [0,1,0]):  # right
            new_dir = directions[(idx+1)%4]
        else:  # left
            new_dir = directions[(idx-1)%4]

        self.direction = new_dir
        self.head[0] += new_dir[0]
        self.head[1] += new_dir[1]

    def get_state(self):
        x, y = self.head

        def danger(dir):
            nx, ny = x + dir[0], y + dir[1]
            return int(nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE or [nx,ny] in self.snake)

        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        idx = directions.index(self.direction)

        state = [
            danger(directions[idx]),                 # straight
            danger(directions[(idx+1)%4]),           # right
            danger(directions[(idx-1)%4]),           # left

            int(self.direction == (1,0)),
            int(self.direction == (-1,0)),
            int(self.direction == (0,1)),
            int(self.direction == (0,-1)),

            int(self.food[0] > x),
            int(self.food[0] < x),
            int(self.food[1] > y),
            int(self.food[1] < y),
        ]

        return np.array(state, dtype=int)