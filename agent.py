import torch
import random
import numpy as np
from collections import deque
from model import DQN

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9

class Agent:
    def __init__(self):
        self.model = DQN()
        self.memory = deque(maxlen=MAX_MEMORY)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = torch.nn.MSELoss()
        self.epsilon = 1.0

    def get_action(self, state):
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state = torch.tensor(state, dtype=torch.float)
            q = self.model(state)
            move = torch.argmax(q).item()

        action = [0,0,0]
        action[move] = 1
        return action

    def remember(self, s, a, r, s2, done):
        self.memory.append((s,a,r,s2,done))

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
                Q_new = rewards[i] + GAMMA * torch.max(self.model(next_states[i]))

            target[i][torch.argmax(torch.tensor(actions[i]))] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(0.01, self.epsilon * 0.995)