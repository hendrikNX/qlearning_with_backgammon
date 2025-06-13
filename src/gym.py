import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
try:
    from src.backgammon_board import BackgammonBoard
except ImportError:
    from backgammon_board import BackgammonBoard

class BackgammonEnv(gym.Env):
    """
    A Gymnasium environment for the game of Backgammon.
    The observation space is a flattened numerical representation of the board state.
    The action space is discrete, representing an index into the list of currently legal moves.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4} 

    def __init__(self, render_mode=None):
        super().__init__()
        self.game_board = BackgammonBoard()
        self.current_player = self.game_board.PLAYER_X # X starts
        self.dice_roll = (0, 0) # Placeholder for current dice roll

        # Define Observation Space
        # This observation space captures the checkers on each point (1-24),
        # checkers on both bars (0, 25), checkers borne off (off_x, off_o),
        # the current player, and the two dice values.
        
        # 24 points: values from -NUM_CHECKERS_PER_PLAYER to +NUM_CHECKERS_PER_PLAYER
        # 2 bar points (index 0, 25 in self.points): values from 0 to NUM_CHECKERS_PER_PLAYER
        # 2 off points (off_x, off_o): values from 0 to NUM_CHECKERS_PER_PLAYER
        # 1 current player: 0 for X, 1 for O (can be generalized to one-hot if more players)
        # 2 dice rolls: values from 1 to 6

        # The total number of features in the observation vector.
        observation_dim = self.game_board.NUM_POINTS + 2 + 2 + 1 + 2 # Points + Bars + Offs + Player + Dice
        
        # Define the lower and upper bounds for each feature in the observation space.
        low_obs = np.array([-self.game_board.NUM_CHECKERS_PER_PLAYER] * self.game_board.NUM_POINTS + \
                           [0] * 2 + [0] * 2 + [0] + [1] * 2, dtype=np.float32)
        high_obs = np.array([self.game_board.NUM_CHECKERS_PER_PLAYER] * self.game_board.NUM_POINTS + \
                            [self.game_board.NUM_CHECKERS_PER_PLAYER] * 2 + [self.game_board.NUM_CHECKERS_PER_PLAYER] * 2 + \
                            [1] + [6] * 2, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, shape=(observation_dim,), dtype=np.float32)

        # Define Action Space: This is dynamic based on legal moves.
        # We set a large discrete space, assuming the agent will only choose from valid indices.
        # The agent's learning algorithm will need to handle action masking for invalid actions.
        # Max possible legal move sequences for Backgammon can be several hundreds, potentially hitting 1000+.
        # A value like 2000 is a safe upper bound.
        self.action_space = spaces.Discrete(2000) 

        self.render_mode = render_mode
        self._cached_legal_moves = [] # To store legal moves for the current turn and map actions to move sequences

    def _get_observation(self):
        """
        Generates a flat numerical observation of the current game state for the agent.
        Combines point values, bar counts, off counts, current player, and dice rolls.
        """
        obs = []
        # Points (1-24)
        obs.extend(self.game_board.points[1:self.game_board.NUM_POINTS + 1].tolist())
        # Bar X, Bar O (indices 0 and 25 of game_board.points array)
        obs.append(self.game_board.points[self.game_board.BAR_X_INDEX])
        obs.append(self.game_board.points[self.game_board.BAR_O_INDEX])
        # Off X, Off O (separate attributes)
        obs.append(self.game_board.off_x)
        obs.append(self.game_board.off_o)
        
        # Current player (0 for X, 1 for O)
        obs.append(1 if self.current_player == self.game_board.PLAYER_O else 0)
        
        # Dice rolls
        obs.extend(list(self.dice_roll))
        
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """Returns auxiliary information for debugging or logging."""
        return {
            "current_player": self.current_player,
            "dice_roll": self.dice_roll,
            "legal_moves_count": len(self._cached_legal_moves)
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial game state."""
        super().reset(seed=seed) # Important for reproducible randomness in Gym
        self.game_board.reset()
        self.current_player = self.game_board.PLAYER_X # Player X always starts the game

        # Roll initial dice for the first player
        self.dice_roll = (random.randint(1, 6), random.randint(1, 6)) 

        observation = self._get_observation()
        info = self._get_info()

        # Cache legal moves for the starting player based on initial dice roll
        self._cached_legal_moves = self.game_board.get_legal_moves(self.current_player, self.dice_roll)
        
        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        """
        Executes a chosen action in the environment.
        The action is an integer index corresponding to a legal move sequence.
        """
        reward = 0
        done = False
        
        # Validate the chosen action index against the list of available legal moves
        if not (0 <= action < len(self._cached_legal_moves)):
            # This indicates the agent chose an invalid move index.
            # In an RL setup, this is a severe error and should be heavily penalized.
            # print(f"Warning: Invalid action {action} chosen. Player {self.current_player} had {len(self._cached_legal_moves)} legal moves.")
            reward = -100.0 # Heavy penalty for illegal move
            done = True # Terminate episode
            
            # Return current state (no change), penalty, and done=True
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, done, info

        # Retrieve the actual move sequence from the cached list
        move_sequence_to_perform, _ = self._cached_legal_moves[action]

        # Apply the chosen move sequence to the game board
        self.game_board.make_move(self.current_player, move_sequence_to_perform)

        # Check if the game has ended after the move
        if self.game_board.is_game_over():
            done = True
            winner = self.game_board.get_winner()
            if winner == self.current_player:
                reward = 1.0 # Current player wins the game
            else:
                reward = -1.0 # Current player loses (opponent won)
        else:
            # Game is not over, switch to the next player's turn
            self.current_player = self.game_board.PLAYER_O if self.current_player == self.game_board.PLAYER_X else self.game_board.PLAYER_X
            self.dice_roll = (random.randint(1, 6), random.randint(1, 6)) # Roll new dice for the next player

        observation = self._get_observation()
        info = self._get_info()
        
        # If the game is not done, generate legal moves for the *new* current player
        if not done:
            self._cached_legal_moves = self.game_board.get_legal_moves(self.current_player, self.dice_roll)
            
            # Handle cases where the current player has no legal moves (turn passes)
            if not self._cached_legal_moves and not done:
                print(f"Player {'X' if self.current_player == self.game_board.PLAYER_X else 'O'} has no legal moves. Passing turn.")
                # Switch back to the original player (the one who just made a move)
                self.current_player = self.game_board.PLAYER_O if self.current_player == self.game_board.PLAYER_X else self.game_board.PLAYER_X
                self.dice_roll = (random.randint(1, 6), random.randint(1, 6)) # Roll new dice for them
                
                observation = self._get_observation()
                info = self._get_info()
                self._cached_legal_moves = self.game_board.get_legal_moves(self.current_player, self.dice_roll) # Recalculate legal moves for the player whose turn it now is again

        if self.render_mode == 'human':
            self.render()

        return observation, reward, done, info

    def render(self):
        """Renders the current state of the board to the console."""
        if self.render_mode == 'human':
            print("\n" + str(self.game_board))
            player_char = 'X' if self.current_player == self.game_board.PLAYER_X else 'O'
            print(f"Current Player: {player_char}, Dice: {self.dice_roll}")
            if self.game_board.is_game_over():
                print(f"Game Over! Winner: {'X' if self.game_board.get_winner() == self.game_board.PLAYER_X else 'O'}")
            else:
                print(f"Legal moves for {player_char}: {len(self._cached_legal_moves)}")

    def close(self):
        """Clean up any resources (not needed for this simple environment)."""
        pass