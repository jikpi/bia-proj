import numpy as np

from exb1_pole_balancing.pole_balancing import *
import gymnasium as gym


def animate_pole(agent, env_name="CartPole-v1", episodes=5, device=torch.device("cpu")):
    env = gym.make(env_name, render_mode="human")

    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        print(f"Showing episode {episode + 1}")
        while not terminated and not truncated:
            env.render()
            with torch.no_grad():
                action = agent.online_net(state).max(1)[1].view(1, 1)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            step = step + 1

            if not terminated:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            state = next_state

        print(f"Showed episode {episode + 1}, reward: {total_reward}, steps: {step}")

        if total_reward > 200:
            break
    env.close()


def pole_solve():
    # todo: >400 by melo byt dosahnutelne, i ==500. (mozna bude nutne spustit vicekrat)
    # vetsinou jakmile v trenovani reward prekrocila na nejakou dobu 200, tak rychle vyskocila na 500
    # (proto early stopping takove jake je)

    device = torch.device("cpu")
    # misto hledani syra, kde se q hodnoty ukladaly do tabulky, je nutne pro tento problem pouzit NN
    # (tento problem je oproti hledani syra kontinualni - nutne pouzit NN pro aproximaci Q funkce)

    episode_count = 1000  # max pocet epizod pro trenink, epizoda = 1 pokus agenta o vyreseni problemu
    target_network_update_count = 10  # jak casto aktualizovat NN pro target (stabilizace uceni)

    # 500 je maximalni skore pro tento problem
    env = gym.make("CartPole-v1")

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(n_observations, n_actions)
    episode_rewards = []

    print(f"Starting training...")

    # episode_count epizod
    for i_episode in range(episode_count):
        # reset prostredi do pocatecniho stavu
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        terminated = False
        truncated = False
        current_episode_reward = 0

        while not terminated and not truncated:
            # vyber akce
            action = agent.select_action(state)
            # vykonani akce, ziskani stavu (step snapshot) a odmeny
            observation, reward, terminated, truncated, _ = env.step(action.item())
            # celkova episode reward
            current_episode_reward += reward

            reward_tensor = torch.tensor([reward], device=device).unsqueeze(0)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # ulozeni step snapshot do pameti
            agent.memory.push(state, action, next_state, reward_tensor, terminated)

            state = next_state

            # uceni po kazde akci
            agent.learn()

            if terminated or truncated:
                episode_rewards.append(current_episode_reward)
                if (i_episode + 1) % 20 == 0 or i_episode == episode_count - 1:
                    print(
                        f"Episode: {i_episode + 1}/{episode_count}, reward: {current_episode_reward:.2f}, epsilon: {agent.calculate_epsilon():.3f}")
                break

        # aktualizace target site - kopie vah z online site
        if (i_episode + 1) % target_network_update_count == 0:
            agent.update_target_net()

        # early stop
        if len(episode_rewards) >= 50:
            avg_reward = np.mean(episode_rewards[-50:])
            if avg_reward >= 200.0:
                print(f"Early stopping: Reward: {avg_reward:.2f}")
                break

    env.close()

    print('!!! Finished. Press enter to show animation.')
    input()

    animate_pole(agent)
