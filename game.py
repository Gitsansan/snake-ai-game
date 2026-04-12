# Complete Snake game with DQN AI agent using PyTorch and Pygame

import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ===== CONFIG =====
GRID_SIZE = 10
CELL = 40
WIDTH = GRID_SIZE * CELL
HEIGHT = GRID_SIZE * CELL

LR = 0.001
GAMMA = 0.9
BATCH_SIZE = 1000
MEMORY_SIZE = 100_000

# ===== MODEL =====
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# ===== GAME =====
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = (1, 0)
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
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        idx = directions.index(self.direction)

        if action == 0:
            new_dir = directions[idx]
        elif action == 1:
            new_dir = directions[(idx+1)%4]
        else:
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
            danger(directions[idx]),
            danger(directions[(idx+1)%4]),
            danger(directions[(idx-1)%4]),

            int(self.direction == (1,0)),
            int(self.direction == (-1,0)),
            int(self.direction == (0,1)),
            int(self.direction == (0,-1)),

            int(self.food[0] > x),
            int(self.food[0] < x),
            int(self.food[1] > y),
            int(self.food[1] < y),
        ]

        return np.array(state, dtype=float)

    def render(self):
        self.screen.fill((0,0,0))

        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0,255,0), (x*CELL, y*CELL, CELL, CELL))

        pygame.draw.rect(self.screen, (255,0,0), (self.food[0]*CELL, self.food[1]*CELL, CELL, CELL))

        pygame.display.flip()
        self.clock.tick(10)

# ===== AGENT =====
class Agent:
    def __init__(self):
        self.model = DQN()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,2)
        state = torch.tensor(state, dtype=torch.float)
        return torch.argmax(self.model(state)).item()

    def remember(self, s,a,r,s2,d):
        self.memory.append((s,a,r,s2,d))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        pred = self.model(states)
        target = pred.clone().detach()

        for i in range(len(batch)):
            Q_new = rewards[i]
            if not dones[i]:
                Q_new += GAMMA * torch.max(self.model(next_states[i]))
            target[i][actions[i]] = Q_new

        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(0.01, self.epsilon * 0.995)

# ===== TRAIN LOOP =====
game = SnakeGame()
agent = Agent()

for episode in range(200):
    state = game.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = agent.get_action(state)
        next_state, reward, done = game.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        game.render()

        if done:
            print(f"Episode {episode}, Score: {game.score}")
            break