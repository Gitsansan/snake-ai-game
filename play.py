import pygame
import torch
from env import SnakeGame
from model import DQN
from config import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

game = SnakeGame()
model = DQN()

model.load_state_dict(torch.load("models/model_400.pth"))
model.eval()

state = game.reset()

# ===== CONTROL MODE =====
AI_MODE = True   # press SPACE to toggle

def get_human_action(keys, current_dir):
    dirs = [(1,0),(0,1),(-1,0),(0,-1)]
    idx = dirs.index(current_dir)

    if keys[pygame.K_UP]:
        new = (0,-1)
    elif keys[pygame.K_DOWN]:
        new = (0,1)
    elif keys[pygame.K_LEFT]:
        new = (-1,0)
    elif keys[pygame.K_RIGHT]:
        new = (1,0)
    else:
        return 0  # no change

    # convert absolute direction → relative action
    if new == dirs[idx]:
        return 0  # straight
    elif new == dirs[(idx+1)%4]:
        return 1  # right
    elif new == dirs[(idx-1)%4]:
        return 2  # left
    else:
        return 0  # invalid reverse, ignore

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                AI_MODE = not AI_MODE  # toggle mode

    if AI_MODE:
        state_tensor = torch.tensor(state, dtype=torch.float)
        action = torch.argmax(model(state_tensor)).item()
    else:
        keys = pygame.key.get_pressed()
        action = get_human_action(keys, game.direction)

    state, _, done = game.step(action)

    # ===== RENDER =====
    screen.fill((20,20,20))

    # snake
    for x,y in game.snake:
        pygame.draw.rect(screen, (0,200,0), (x*CELL, y*CELL, CELL, CELL))

    # food
    pygame.draw.rect(screen, (200,0,0), (game.food[0]*CELL, game.food[1]*CELL, CELL, CELL))

    # UI
    mode_text = "AI" if AI_MODE else "HUMAN"
    score_text = font.render(f"Score: {game.score} | Mode: {mode_text}", True, (255,255,255))
    screen.blit(score_text, (10,10))

    pygame.display.flip()
    clock.tick(90)

    if done:
        state = game.reset()