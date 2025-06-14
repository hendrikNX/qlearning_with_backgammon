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

    This version provides symmetric observations for the active player by flipping the board
    when Player O is active.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4} 

    def __init__(self, render_mode=None):
        super().__init__()
        self.game_board = BackgammonBoard()
        self.current_player = self.game_board.PLAYER_X # X starts
        self.dice_roll = (0, 0) # Placeholder for current dice roll

        # Define Observation Space
        # This observation space captures:
        # - 24 points: values from -NUM_CHECKERS_PER_PLAYER to +NUM_CHECKERS_PER_PLAYER (relative to active player)
        # - 2 bar points: values from 0 to NUM_CHECKERS_PER_PLAYER (relative to active player: my bar, opponent's bar)
        # - 2 off points: values from 0 to NUM_CHECKERS_PER_PLAYER (relative to active player: my off, opponent's off)
        # - 2 dice rolls: values from 1 to 6
        
        # The total number of features in the observation vector.
        # This is 30, as 'current_player' feature is no longer needed since observation is player-centric.
        observation_dim = self.game_board.NUM_POINTS + 2 + 2 + 2 # Points(24) + Bars(2) + Offs(2) + Dice(2) = 30
        
        # Define the lower and upper bounds for each feature in the observation space.
        # Max checkers on a point is 15, on bar is 15, off is 15.
        low_obs = np.array([-self.game_board.NUM_CHECKERS_PER_PLAYER] * self.game_board.NUM_POINTS + \
                           [0] * 2 + [0] * 2 + [1] * 2, dtype=np.float32)
        high_obs = np.array([self.game_board.NUM_CHECKERS_PER_PLAYER] * self.game_board.NUM_POINTS + \
                            [self.game_board.NUM_CHECKERS_PER_PLAYER] * 2 + [self.game_board.NUM_CHECKERS_PER_PLAYER] * 2 + \
                            [6] * 2, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, shape=(observation_dim,), dtype=np.float32)

        # Define Action Space (remains the same maximum possible)
        self.action_space = spaces.Discrete(2000) 

        self.render_mode = render_mode
        self._cached_legal_moves = [] # To store legal moves for the *next* player's turn or for info

    def _get_observation(self, player_perspective):
        """
        Generates a flat numerical observation of the current game state,
        from the perspective of 'player_perspective'.
        Player X's perspective: X's checkers positive, O's negative. Home board 1-6.
        Player O's perspective: O's checkers positive, X's negative. Home board 19-24 (conceptually 1-6 from O's view).
        """
        obs = []
        
        if player_perspective == self.game_board.PLAYER_X:
            # Player X's perspective (standard internal representation)
            obs.extend(self.game_board.points[1:self.game_board.NUM_POINTS + 1].tolist())
            obs.append(self.game_board.points[self.game_board.BAR_X_INDEX]) # My bar
            obs.append(self.game_board.points[self.game_board.BAR_O_INDEX]) # Opponent's bar
            obs.append(self.game_board.off_x) # My off
            obs.append(self.game_board.off_o) # Opponent's off
        else: # player_perspective == self.game_board.PLAYER_O
            # Player O's perspective (flipped board and checker counts)
            # Iterate through points from X's 1 (O's 24) to X's 24 (O's 1)
            # Negate value: O's checkers become positive, X's become negative
            flipped_points = np.zeros(self.game_board.NUM_POINTS, dtype=int)
            for i in range(1, self.game_board.NUM_POINTS + 1):
                # O's point 'j' is X's point (25-j)
                # So to get the value for O's point 'j' (from 1 to 24), we look at X's point (25-j) and negate it.
                flipped_points[i-1] = -self.game_board.points[self.game_board.NUM_POINTS + 1 - i]
            obs.extend(flipped_points.tolist())

            obs.append(self.game_board.points[self.game_board.BAR_O_INDEX]) # My bar (Player O's actual bar)
            obs.append(self.game_board.points[self.game_board.BAR_X_INDEX]) # Opponent's bar (Player X's actual bar)
            obs.append(self.game_board.off_o) # My off (Player O's actual off count)
            obs.append(self.game_board.off_x) # Opponent's off (Player X's actual off count)

        # Dice rolls (always included in raw form)
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

        # Get observation for the starting player (Player X)
        observation = self._get_observation(self.current_player)

        # Cache legal moves for the starting player based on initial dice roll
        self._cached_legal_moves = self.game_board.get_legal_moves(self.current_player, self.dice_roll)
        
        info = self._get_info() # Generate info *after* caching legal moves

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action, current_player_legal_moves_list): # ADDED: current_player_legal_moves_list
        """
        Executes a chosen action in the environment.
        The action is an integer index corresponding to a legal move sequence.
        'current_player_legal_moves_list' is the list of legal moves for the player whose turn it just was.
        """
        reward = 0
        done = False
        
        # Validate the chosen action index against the list provided
        if not (0 <= action < len(current_player_legal_moves_list)):
            print(f"Warning: Invalid action {action} chosen by Player {'X' if self.current_player == self.game_board.PLAYER_X else 'O'}. Had {len(current_player_legal_moves_list)} legal moves.")
            reward = -100.0 # Heavy penalty for illegal move
            done = True # Terminate episode
            
            # Return current state (no change), penalty, and done=True
            observation = self._get_observation(self.current_player)
            info = self._get_info() # This _cached_legal_moves is from the *previous* turn still, correct for next info
            return observation, reward, done, info

        # Retrieve the actual move sequence from the PASSED list (NOT self._cached_legal_moves)
        move_sequence_to_perform, _ = current_player_legal_moves_list[action]

        # Apply the chosen move sequence to the game board
        self.game_board.make_move(self.current_player, move_sequence_to_perform)

        # Check if the game has ended after the move
        if self.game_board.is_game_over():
            done = True
            winner = self.game_board.get_winner()
            # Reward is relative to the player who *just moved* (current_player).
            if winner == self.current_player:
                reward = 1.0 # Current player wins the game
            else:
                reward = -1.0 # Current player loses (opponent won)
        else:
            # Game is not over, switch to the next player's turn
            self.current_player = self.game_board.PLAYER_O if self.current_player == self.game_board.PLAYER_X else self.game_board.PLAYER_X
            self.dice_roll = (random.randint(1, 6), random.randint(1, 6)) # Roll new dice for the next player

        # Get observation for the *new* current player
        observation = self._get_observation(self.current_player)
        
        # If the game is not done, generate legal moves for the *new* current player
        # This update to _cached_legal_moves is for the *next* iteration's info and agent's choice
        if not done:
            self._cached_legal_moves = self.game_board.get_legal_moves(self.current_player, self.dice_roll)
            
            # Handle cases where the current player (who just became active) has no legal moves (turn passes)
            if not self._cached_legal_moves: # No 'done' check here as it would be handled above if game was over
                print(f"Player {'X' if self.current_player == self.game_board.PLAYER_X else 'O'} has no legal moves. Passing turn.")
                # The turn passes back to the other player (who just played)
                # The 'reward' in this case is typically 0, no immediate change for the player whose turn was skipped.
                self.current_player = self.game_board.PLAYER_O if self.current_player == self.game_board.PLAYER_X else self.game_board.PLAYER_X
                self.dice_roll = (random.randint(1, 6), random.randint(1, 6)) # Roll new dice for them
                
                # Get observation for the player who now gets to move again
                observation = self._get_observation(self.current_player)
                self._cached_legal_moves = self.game_board.get_legal_moves(self.current_player, self.dice_roll) # Recalculate legal moves for the player whose turn it now is again
        
        info = self._get_info() # Generate info *after* caching legal moves for the *new* current player

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