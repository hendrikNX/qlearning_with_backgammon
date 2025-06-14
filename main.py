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
import datetime
import time

if __name__ == "__main__":
    # Ensure TensorFlow is using GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True) # Prevent OOM errors for large models
            print(f"TensorFlow is using GPU: {gpus[0]}")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU found. TensorFlow will use CPU.")


    # Environment and Agent Hyperparameters
    num_episodes = 50000 # Number of games to play for training
    max_steps_per_episode = 500 # Max moves per game to prevent infinite loops
    
    # Initialize the Backgammon Environment
    env = BackgammonEnv(render_mode=None) # Set to None for faster training

    # Get dimensions from the environment
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n 

    # Initialize two separate DQN Agents
    agent_X = DQNAgent(observation_dim, action_dim,
                     gamma=0.99, lr=0.0005, epsilon_start=1.0, 
                     epsilon_end=0.01, # Set back to normal epsilon_end after initial debugging
                     epsilon_decay=0.9999, replay_buffer_size=50000, batch_size=512, # Try reducing batch_size for memory
                     target_update_freq=500)
    
    agent_O = DQNAgent(observation_dim, action_dim,
                     gamma=0.99, lr=0.0005, epsilon_start=1.0, 
                     epsilon_end=0.01, # Set back to normal epsilon_end
                     epsilon_decay=0.9999, replay_buffer_size=50000, batch_size=512, # Try reducing batch_size for memory
                     target_update_freq=500)

    episode_rewards_X = [] 
    episode_rewards_O = [] 
    
    print(f"Starting DQN training for {num_episodes} episodes with two agents...")

    # --- TensorBoard & Profiler Setup ---
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/dqn_backgammon/" + current_time
    # Summary writer for general TensorBoard metrics (rewards, epsilon)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Profiler configuration
    # Profile a specific range of steps after the replay buffer is filled
    # E.g., start profiling after 1000 steps, for 10 steps.
    # Adjust `start_step` and `num_profile_steps` as needed.
    # Profiling too many steps can create very large files.
    profiler_start_step = 1000 # Start profiling after this many steps
    profiler_num_steps = 10 # Number of steps to profile
    profiler_active = False # Flag to control profiler lifecycle

    global_step = 0 # To track total steps across episodes

    for episode in tqdm.tqdm(range(num_episodes)):
        state, info = env.reset()
        done = False
        total_reward_X = 0
        total_reward_O = 0
        steps = 0

        while not done and steps < max_steps_per_episode:
            global_step += 1 # Increment global step counter

            # --- Profiler Start/Stop Logic ---
            if global_step == profiler_start_step and not profiler_active:
                print(f"\n--- Starting TensorFlow Profiler at global step {global_step} ---")
                tf.profiler.experimental.start(
                    log_dir, 
                    options=tf.profiler.experimental.ProfilerOptions(
                        host_tracer_level=2, python_tracer_level=1, device_tracer_level=1
                    )
                )
                profiler_active = True
            elif profiler_active and global_step == profiler_start_step + profiler_num_steps:
                print(f"--- Stopping TensorFlow Profiler at global step {global_step} ---")
                tf.profiler.experimental.stop()
                profiler_active = False # Profiler will only run once
                # Add a small sleep to ensure profiler data is flushed to disk
                time.sleep(5)


            current_player = info["current_player"]
            legal_moves_count = info["legal_moves_count"]
            
            current_player_legal_moves_list = env._cached_legal_moves 
            
            if current_player == env.game_board.PLAYER_X:
                active_agent = agent_X
                current_observation = state 
            else: # current_player == env.game_board.PLAYER_O
                active_agent = agent_O
                current_observation = env._get_observation(current_player)

            action = active_agent.choose_action(current_observation, legal_moves_count)

            next_state_env, reward, done, info = env.step(action, current_player_legal_moves_list)
            
            active_agent.store_transition(current_observation, action, reward, next_state_env, done) 

            # Agent learns from experience if enough samples are in the buffer
            # This is the part that does the heavy lifting on the GPU
            active_agent.learn()
            
            state = next_state_env 
            
            if current_player == env.game_board.PLAYER_X:
                total_reward_X += reward
                total_reward_O += -reward 
            else:
                total_reward_O += reward
                total_reward_X += -reward 
            
            steps += 1
            
            if reward == -100.0 and done:
                print(f"Episode {episode+1} ended early due to illegal move (reward -100).")
                break

        episode_rewards_X.append(total_reward_X)
        episode_rewards_O.append(total_reward_O)

        # --- TensorBoard Logging (Metrics) ---
        with summary_writer.as_default():
            tf.summary.scalar(f'reward/agent_X', total_reward_X, step=episode)
            tf.summary.scalar(f'reward/agent_O', total_reward_O, step=episode)
            tf.summary.scalar(f'epsilon/agent_X', agent_X.epsilon, step=episode)
            tf.summary.scalar(f'epsilon/agent_O', agent_O.epsilon, step=episode)
            # You might want to log loss as well, if your learn method returns it

        if (episode + 1) % 100 == 0:
            avg_reward_X = np.mean(episode_rewards_X[-100:])
            avg_reward_O = np.mean(episode_rewards_O[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward X (last 100): {avg_reward_X:.2f}, Avg Reward O (last 100): {avg_reward_O:.2f}")
            print(f"Epsilon X: {agent_X.epsilon:.4f}, Epsilon O: {agent_O.epsilon:.4f}")
            agent_X.q_network.save("backgammon_dqn_agent_X.h5")
            agent_O.q_network.save("backgammon_dqn_agent_O.h5")
            print("Models saved.")

    print("\nTraining Finished!")

    # --- Instructions to view TensorBoard ---
    print("\nTo view TensorBoard and analyze profiling data, run the following command in your terminal:")
    print(f"tensorboard --logdir {log_dir.rsplit('/', 1)[0]}")
    print(f"(The log directory is: {log_dir})")
    print("Then open your web browser and navigate to the address provided by TensorBoard (usually http://localhost:6006)")

    env.close()