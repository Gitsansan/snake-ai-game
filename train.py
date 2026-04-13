from env import SnakeGame
from agent import Agent
from utils import save_model

game = SnakeGame()
agent = Agent()

EPISODES = 5000

# create log file + header
log_file = "training_log.csv"
with open(log_file, "w") as f:
    f.write("episode and score\n")

for ep in range(EPISODES):
    state = game.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = game.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.train()

        state = next_state

        if done:
            score = game.score
            print(f"Ep {ep}, Score {score}")

            # save to file
            with open(log_file, "a") as f:
                f.write(f"Ep {ep}, Score {score}\n")

            break

    if ep % 100 == 0:
        save_model(agent.model, ep)