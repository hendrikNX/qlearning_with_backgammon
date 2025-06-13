try:
    from src.gym import BackgammonEnv
except ImportError:
    from gym import BackgammonEnv

try:
    from src.agent import DQNAgent
except ImportError:
    from agent import DQNAgent

import tensorflow as tf
import numpy as np
import tqdm

if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU for training.")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPU found, using CPU for training.")

    # Environment and Agent Hyperparameters
    num_episodes = 50000 # Number of games to play for training
    max_steps_per_episode = 500 # Max moves per game to prevent infinite loops
    
    # Initialize the Backgammon Environment
    # Using render_mode=None for faster training without rendering.
    # Set to 'human' for occasional visualization.
    env = BackgammonEnv(render_mode=None) 
    
    # Get dimensions from the environment
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n # Max possible actions (2000 in this case)

    # Initialize the DQN Agent
    agent = DQNAgent(observation_dim, action_dim,
                     gamma=0.99,
                     lr=0.0005,
                     epsilon_start=1.0,
                     epsilon_end=0.5,
                     epsilon_decay=0.9999, # Slower decay for more exploration
                     replay_buffer_size=50000,
                     batch_size=128,
                     target_update_freq=500)

    episode_rewards = []
    
    print(f"Starting DQN training for {num_episodes} episodes...")

    for episode in tqdm.tqdm(range(num_episodes)):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Agent chooses action based on current state and available legal moves
            # The legal_moves_count is crucial for action masking in choose_action
            legal_moves_count = info["legal_moves_count"]
            action = agent.choose_action(state, legal_moves_count)

            # Environment takes a step with the chosen action
            next_state, reward, done, info = env.step(action)
            
            # Store the transition in the agent's replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Agent learns from experience if enough samples are in the buffer
            agent.learn()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # If the game ended due to an illegal move, break early
            if reward == -100.0 and done:
                break

        episode_rewards.append(total_reward)

        # Logging training progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            # print(env.game_board) # Uncomment for occasional board visualization during training

    print("\nTraining Finished!")
    # Optionally, save the trained model
    # agent.q_network.save("backgammon_dqn_model.h5") # Use .h5 for Keras models
    # print("Model saved as backgammon_dqn_model.h5")

    env.close()
