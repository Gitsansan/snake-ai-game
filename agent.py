import torch
import random
import numpy as np
from collections import deque
from model import DQN
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.model = DQN().to(device)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = torch.nn.MSELoss()

        self.epsilon = 1.0
        self.step_count = 0  # for training frequency control

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        state = torch.tensor(state, dtype=torch.float).to(device)
        with torch.no_grad():
            q = self.model(state)

        return torch.argmax(q).item()

    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))

    def train(self):
        # 🔹 train every few steps (speed boost)
        self.step_count += 1
        if self.step_count % 5 != 0:
            return

        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*batch)

        # 🔹 fast tensor conversion
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.bool).to(device)
        actions = torch.tensor(actions).to(device)

        # 🔹 current Q values
        pred = self.model(states)

        # 🔹 next Q values (no grad)
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]

        # 🔹 target Q values (vectorized)
        target_q = rewards + GAMMA * next_q * (~dones)

        target = pred.clone().detach()
        target[range(BATCH_SIZE), actions] = target_q

        # 🔹 loss
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 🔹 epsilon decay
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)