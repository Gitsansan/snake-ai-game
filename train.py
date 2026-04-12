from env import SnakeGame
from agent import Agent

game = SnakeGame()
agent = Agent()

for episode in range(1000):
    state = game.reset()
    total_reward = 0

    while True:
        action = agent.get_action(state)
        next_state, reward, done = game.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {episode}, Score: {game.score}")
            break