from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

class QNetwork(keras.Model):
    """
    Neural Network (Q-Network) to approximate Q-values using TensorFlow/Keras.
    It takes the observation (state) as input and outputs a Q-value for each possible action.
    """
    def __init__(self, observation_dim, action_dim):
        super(QNetwork, self).__init__()
        # Using a Sequential model for clarity, matching the original structure but with corrections.
        self.model = Sequential()
        self.model.add(Input(shape=(observation_dim,)))  # Input layer with correct shape
        self.model.add(Dense(128, activation='relu'))    # Hidden layer with ReLU activation
        self.model.add(Dense(128, activation='relu'))    # Another hidden layer with ReLU activation
        self.model.add(Dense(action_dim, activation='linear'))  # Output layer: linear activation for Q-values

    def call(self, x):
        """
        Forward pass through the network.
        x: input tensor (observation)
        """
        return self.model(x)