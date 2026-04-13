import random
import numpy as np
from collections import deque
from config import GRID_SIZE

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = (1,0)
        self.head = [GRID_SIZE//2, GRID_SIZE//2]
        self.snake = deque([self.head.copy()])
        self.spawn_food()
        self.score = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            self.food = [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)]
            if self.food not in self.snake:
                break

    def step(self, action):
        # 🔹 BEFORE MOVE → compute old distance
        old_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

        self.move(action)
        self.snake.appendleft(self.head.copy())

        reward = -0.2  # step penalty
        done = False

        if self.is_collision():
            return self.get_state(), -15, True

        # 🔹 AFTER MOVE → compute new distance
        new_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

        # 🔹 distance-based reward
        if new_dist < old_dist:
            reward += 0.3
        else:
            reward -= 0.10

        if self.head == self.food:
            self.score += 1
            reward = 15
            self.spawn_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, done

    def is_collision(self):
        x,y = self.head
        if x<0 or x>=GRID_SIZE or y<0 or y>=GRID_SIZE:
            return True
        if self.head in list(self.snake)[1:]:
            return True
        return False

    def move(self, action):
        dirs = [(1,0),(0,1),(-1,0),(0,-1)]
        idx = dirs.index(self.direction)

        if action == 0:
            new = dirs[idx]
        elif action == 1:
            new = dirs[(idx+1)%4]
        else:
            new = dirs[(idx-1)%4]

        self.direction = new
        self.head[0] += new[0]
        self.head[1] += new[1]

    def get_state(self):
        x, y = self.head

        directions = [(1,0),(0,1),(-1,0),(0,-1)]
        idx = directions.index(self.direction)

        def look(dir):
            dx, dy = dir
            distance = 0
            food_seen = 0
            body_seen = 0

            cx, cy = x, y

            while True:
                cx += dx
                cy += dy
                distance += 1

                # wall
                if cx < 0 or cx >= GRID_SIZE or cy < 0 or cy >= GRID_SIZE:
                    return [
                        1.0 / distance,   # wall distance (inverse)
                        body_seen,
                        food_seen
                    ]

                if [cx, cy] in self.snake:
                    body_seen = 1

                if [cx, cy] == self.food:
                    food_seen = 1

        # relative directions
        front = directions[idx]
        right = directions[(idx+1)%4]
        left  = directions[(idx-1)%4]
        back  = directions[(idx+2)%4]

        state = []

        for d in [front, right, left, back]:
            state.extend(look(d))

        # current direction (still useful)
        state.extend([
            int(self.direction == (1,0)),
            int(self.direction == (-1,0)),
            int(self.direction == (0,1)),
            int(self.direction == (0,-1)),
        ])

        return np.array(state, dtype=float)