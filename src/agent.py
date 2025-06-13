import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from collections import deque # For experience replay buffer
try:
    from src.qnetwork import QNetwork
except ImportError:
    from qnetwork import QNetwork

class DQNAgent:
    """
    Deep Q-Learning Agent for Backgammon using TensorFlow/Keras.
    Manages the Q-network, target network, experience replay, and learning process.
    """
    def __init__(self, observation_dim, action_dim, gamma=0.99, lr=0.001,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 replay_buffer_size=10000, batch_size=64, target_update_freq=100):
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma # Discount factor for future rewards
        self.epsilon = epsilon_start # Exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.q_network = QNetwork(observation_dim, action_dim)
        self.target_q_network = QNetwork(observation_dim, action_dim)
        # Build networks by passing a dummy input to initialize weights
        self.q_network.build(input_shape=(None, observation_dim))
        self.target_q_network.build(input_shape=(None, observation_dim))
        self.target_q_network.set_weights(self.q_network.get_weights()) # Copy weights

        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.loss_fn = losses.MeanSquaredError() # Mean Squared Error Loss for Q-value prediction

        # Experience Replay Buffer: stores (state, action, reward, next_state, done) tuples
        self.replay_buffer = deque(maxlen=replay_buffer_size)

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def choose_action(self, observation, legal_moves_count):
        """
        Chooses an action using an epsilon-greedy strategy.
        'legal_moves_count' is essential for action masking.
        """
        if random.random() < self.epsilon:
            # Explore: Choose a random legal action
            if legal_moves_count > 0:
                action = random.randrange(legal_moves_count)
            else:
                # If no legal moves, return a default invalid action that the environment will penalize
                action = 0 # Or any other invalid index
        else:
            # Exploit: Choose the action with the highest predicted Q-value
            observation_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
            observation_tensor = tf.expand_dims(observation_tensor, 0) # Add batch dimension

            q_values = self.q_network(observation_tensor)
            
            # Apply action masking: set Q-values of illegal actions to a very low number
            # Create a mask: True for legal actions, False for illegal
            legal_mask = tf.cast(tf.range(self.action_dim) < legal_moves_count, tf.float32)
            # Replace 0s with -inf and 1s with 0
            mask_tensor = (legal_mask - 1) * 1e9 # Large negative number for illegal moves
            
            masked_q_values = q_values + mask_tensor
            
            action = tf.argmax(masked_q_values, axis=1).numpy()[0] # Get the index of the highest Q-value

        return action

    def learn(self):
        """
        Performs a single learning step using a batch of experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return # Not enough samples to learn

        # Sample a batch of experiences from the replay buffer
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert to TensorFlow tensors
        states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int64) # Actions should be int
        rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.array(dones), dtype=tf.bool)

        with tf.GradientTape() as tape:
            # Compute Q-values for current states using the main Q-network
            # We need to gather the Q-value for the specific action taken
            q_values = self.q_network(states)
            batch_indices = tf.cast(tf.range(self.batch_size), dtype=tf.int64)
            current_q_values = tf.gather_nd(q_values, tf.stack([batch_indices, actions], axis=1))
            current_q_values = tf.expand_dims(current_q_values, axis=1) # Ensure shape for loss calculation

            # Compute max Q-values for next states using the target Q-network
            next_q_values = self.target_q_network(next_states)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1, keepdims=True)
            
            # Calculate target Q-values: R + gamma * max_Q(S', A')
            # If done (game over), target Q is just the reward.
            # tf.cast(dones, tf.float32) converts True to 1.0, False to 0.0
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - tf.cast(dones, tf.float32)))
            target_q_values = tf.expand_dims(target_q_values, axis=1) # Ensure shape for loss calculation

            # Compute loss
            loss = self.loss_fn(current_q_values, target_q_values)

        # Get gradients and apply them to the Q-network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # Decay epsilon (exploration rate)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()

    def update_target_network(self):
        """Copies weights from the Q-network to the target Q-network."""
        self.target_q_network.set_weights(self.q_network.get_weights())