import numpy as np
from collections import defaultdict # For memoization in move generation

class BackgammonBoard:
    """
    Represents the Backgammon board state.

    The board is represented as follows:
    - self.points: A NumPy array of 26 integers.
        - self.points[0] (BAR_X_INDEX): Number of Player X's checkers on the bar (positive integer).
        - self.points[1] to self.points[24]: Represent the 24 points on the board.
            - Positive value: Player X has that many checkers.
            - Negative value: Player O has abs(value) checkers.
            - Zero: The point is empty.
        - self.points[25] (BAR_O_INDEX): Number of Player O's checkers on the bar (positive integer).
    - self.off_x: Number of Player X's checkers borne off.
    - self.off_o: Number of Player O's checkers borne off.

    Player X moves from point 24 down to point 1.
    Player O moves from point 1 up to point 24.
    """

    PLAYER_X = 1
    PLAYER_O = -1

    NUM_POINTS = 24
    BAR_X_INDEX = 0  # Index for X's checkers on their bar (count)
    BAR_O_INDEX = NUM_POINTS + 1 # Index for O's checkers on their bar (count)
    
    # Sentinel values for move representation
    # These are 'locations' for from/to in a move tuple
    PLAYER_X_OFF_TARGET = BAR_X_INDEX - 1 # Moving TO X's off-board area
    PLAYER_O_OFF_TARGET = BAR_O_INDEX + 1 # Moving TO O's off-board area

    NUM_CHECKERS_PER_PLAYER = 3

    def __init__(self):
        """Initializes an empty board and then resets it to the starting position."""
        # Use np.zeros to initialize the points array with integers
        self.points = np.zeros(self.NUM_POINTS + 2, dtype=int)
        self.off_x = 0
        self.off_o = 0
        self.reset()

    def reset(self):
        """Resets the board to the standard starting position."""
        # Use np.zeros to re-initialize the points array
        self.points = np.zeros(self.NUM_POINTS + 2, dtype=int)
        self.off_x = 0
        self.off_o = 0

        # Player X's initial checkers (positive numbers indicate X's checkers)
        # Player X moves from higher numbered points to lower numbered points.
        # X's home board is points 1-6.
        # self.points[24] = 2  # Point 24 (X's starting point)
        # self.points[13] = 5  # Point 13 (X's starting point)
        # self.points[8]  = 3  # Point 8 (X's starting point)
        # self.points[6]  = 5  # Point 6 (X's starting point)
        self.points[24]  = 1  # Point 6 (X's starting point)
        self.points[8]  = 2  # Point 6 (X's starting point)

        # Player O's initial checkers (negative numbers indicate O's checkers)
        # Player O moves from lower numbered points to higher numbered points.
        # O's home board is points 19-24.
        # self.points[1]  = -2 # Point 1 (O's starting point, from X's perspective)
        # self.points[12] = -5 # Point 12 (O's starting point)
        # self.points[17] = -3 # Point 17 (O's starting point)
        # self.points[19] = -5 # Point 19 (O's starting point)
        self.points[1] = -1 # Point 19 (O's starting point)
        self.points[17] = -2 # Point 19 (O's starting point)

        # Verify total checkers
        sum_x_checkers = self.points[self.BAR_X_INDEX] + \
                         np.sum(self.points[1:self.NUM_POINTS+1][self.points[1:self.NUM_POINTS+1] > 0]) + \
                         self.off_x
        sum_o_checkers = self.points[self.BAR_O_INDEX] + \
                         np.sum(np.abs(self.points[1:self.NUM_POINTS+1][self.points[1:self.NUM_POINTS+1] < 0])) + \
                         self.off_o
        assert sum_x_checkers == self.NUM_CHECKERS_PER_PLAYER, f"X has {sum_x_checkers} checkers"
        assert sum_o_checkers == self.NUM_CHECKERS_PER_PLAYER, f"O has {sum_o_checkers} checkers"


    def get_board_state(self):
        """
        Returns the raw state of the board.
        This can be used by an environment or agent to create its feature representation.
        """
        return {
            "points": self.points.tolist(),
            "off_x": self.off_x,
            "off_o": self.off_o
        }

    def is_game_over(self):
        """Checks if the game has ended (one player has borne off all checkers)."""
        return self.off_x == self.NUM_CHECKERS_PER_PLAYER or \
               self.off_o == self.NUM_CHECKERS_PER_PLAYER

    def get_winner(self):
        """
        Determines the winner of the game.
        Returns PLAYER_X, PLAYER_O, or None if the game is not over.
        """
        if self.off_x == self.NUM_CHECKERS_PER_PLAYER:
            return self.PLAYER_X
        if self.off_o == self.NUM_CHECKERS_PER_PLAYER:
            return self.PLAYER_O
        return None

    def _get_current_board_tuple_state(self):
        """Returns a hashable tuple representing the current board state."""
        return (
            tuple(self.points), # Make points array hashable
            self.points[self.BAR_X_INDEX], # Explicitly include bar counts for clarity in state
            self.points[self.BAR_O_INDEX],
            self.off_x,
            self.off_o
        )

    def _are_all_checkers_home(self, player, points_arr, player_bar_count):
        """
        Checks if all of a player's checkers are in their home board or off.
        Assumes points_arr includes bar counts at BAR_X_INDEX and BAR_O_INDEX.
        """
        if player_bar_count > 0:
            return False

        if player == self.PLAYER_X:
            # X's home board is 1-6. Checkers outside this range (7-24).
            for i in range(7, self.NUM_POINTS + 1):
                if points_arr[i] > 0:  # X checker outside home
                    return False
        else:  # Player O
            # O's home board is 19-24 (points 1-18 from O's perspective).
            # Checkers outside this range (1-18 for O).
            for i in range(1, 19):
                if points_arr[i] < 0:  # O checker outside home
                    return False
        return True

    def _calculate_single_potential_moves(self, player, die_roll, current_points_arr,
                                          p_x_bar, p_o_bar, p_x_off, p_o_off):
        """
        Calculates all possible single checker moves for a given player and die roll
        from a given board state.
        Returns a list of tuples: (from_loc, to_loc, next_points_arr, next_bar_x, next_bar_o, next_off_x, next_off_o)
        """
        possible_single_moves = []
        temp_points = current_points_arr.copy() # Work on a copy

        if player == self.PLAYER_X:
            player_bar_loc = self.BAR_X_INDEX
            player_actual_bar_index = self.BAR_X_INDEX
            player_off_target = self.PLAYER_X_OFF_TARGET
            opponent_actual_bar_index = self.BAR_O_INDEX
            player_sign = 1
            home_board_start, home_board_end = 1, 6
        else: # PLAYER_O
            player_bar_loc = self.BAR_O_INDEX
            player_actual_bar_index = self.BAR_O_INDEX
            player_off_target = self.PLAYER_O_OFF_TARGET
            opponent_actual_bar_index = self.BAR_X_INDEX
            player_sign = -1
            home_board_start, home_board_end = 19, 24

        # 1. Moves from the Bar
        if temp_points[player_actual_bar_index] > 0:
            target_point = die_roll if player == self.PLAYER_X else (self.NUM_POINTS + 1 - die_roll)
            
            if 1 <= target_point <= self.NUM_POINTS:
                # Check if target point is not blocked (<= 1 opponent checker)
                if player_sign * temp_points[target_point] >= -1:
                    next_pts = temp_points.copy()
                    next_bar_x, next_bar_o = p_x_bar, p_o_bar
                    next_off_x, next_off_o = p_x_off, p_o_off

                    next_pts[player_actual_bar_index] -= 1
                    if player == self.PLAYER_X: next_bar_x -=1
                    else: next_bar_o -=1
                    
                    if temp_points[target_point] == -player_sign: # Hit opponent's blot
                        next_pts[target_point] = player_sign # Occupy with one checker
                        next_pts[opponent_actual_bar_index] += 1
                        if player == self.PLAYER_X: next_bar_o +=1
                        else: next_bar_x +=1
                    else:
                        next_pts[target_point] += player_sign
                    
                    possible_single_moves.append((
                        player_bar_loc, target_point, next_pts,
                        next_bar_x, next_bar_o, next_off_x, next_off_o
                    ))
            return possible_single_moves # Must move from bar if possible

        # 2. Bearing Off (only if all checkers are home)
        all_home = self._are_all_checkers_home(player, temp_points, temp_points[player_actual_bar_index])
        if all_home:
            if player == self.PLAYER_X:
                # Exact bear off
                if home_board_start <= die_roll <= home_board_end and temp_points[die_roll] > 0:
                    next_pts = temp_points.copy()
                    next_bar_x, next_bar_o = p_x_bar, p_o_bar
                    next_off_x, next_off_o = p_x_off, p_o_off

                    next_pts[die_roll] -= 1
                    next_off_x += 1
                    possible_single_moves.append((
                        die_roll, player_off_target, next_pts,
                        next_bar_x, next_bar_o, next_off_x, next_off_o
                    ))
                # Overshoot bear off (from highest pip)
                elif die_roll > home_board_end: # e.g. roll 6, highest checker on 5 or less
                    highest_pip_with_checker = -1
                    for p in range(home_board_end, home_board_start - 1, -1):
                        if temp_points[p] > 0:
                            highest_pip_with_checker = p
                            break
                    if highest_pip_with_checker != -1 and die_roll > (highest_pip_with_checker - home_board_start): # Check if die is large enough
                        next_pts = temp_points.copy()
                        next_bar_x, next_bar_o = p_x_bar, p_o_bar
                        next_off_x, next_off_o = p_x_off, p_o_off
                        
                        next_pts[highest_pip_with_checker] -= 1
                        next_off_x += 1
                        possible_single_moves.append((
                            highest_pip_with_checker, player_off_target, next_pts,
                            next_bar_x, next_bar_o, next_off_x, next_off_o
                        ))
            else: # PLAYER_O
                # Exact bear off for O (e.g., point 22, die 3 -> 22+3=25, target is NUM_POINTS+1 - die_roll)
                target_bear_off_point = self.NUM_POINTS + 1 - die_roll
                if home_board_start <= target_bear_off_point <= home_board_end and temp_points[target_bear_off_point] < 0:
                    next_pts = temp_points.copy()
                    next_bar_x, next_bar_o = p_x_bar, p_o_bar
                    next_off_x, next_off_o = p_x_off, p_o_off

                    next_pts[target_bear_off_point] += 1 # Becomes less negative or 0
                    next_off_o += 1
                    possible_single_moves.append((
                        target_bear_off_point, player_off_target, next_pts,
                        next_bar_x, next_bar_o, next_off_x, next_off_o
                    ))
                # Overshoot bear off for O
                elif die_roll > (home_board_end - home_board_start + 1): # Die roll is larger than home board span
                    lowest_pip_with_checker = -1 # O's highest pip is lowest index
                    for p in range(home_board_start, home_board_end + 1):
                        if temp_points[p] < 0:
                            lowest_pip_with_checker = p
                            break
                    if lowest_pip_with_checker != -1 and die_roll > (home_board_end - lowest_pip_with_checker):
                        next_pts = temp_points.copy()
                        next_bar_x, next_bar_o = p_x_bar, p_o_bar
                        next_off_x, next_off_o = p_x_off, p_o_off

                        next_pts[lowest_pip_with_checker] += 1
                        next_off_o += 1
                        possible_single_moves.append((
                            lowest_pip_with_checker, player_off_target, next_pts,
                            next_bar_x, next_bar_o, next_off_x, next_off_o
                        ))
        
        # 3. Regular moves on the board
        for p_from in range(1, self.NUM_POINTS + 1):
            if player_sign * temp_points[p_from] > 0: # Player has checkers on this point
                target_point = p_from - die_roll if player == self.PLAYER_X else p_from + die_roll

                if 1 <= target_point <= self.NUM_POINTS:
                    if player_sign * temp_points[target_point] >= -1: # Not blocked
                        next_pts = temp_points.copy()
                        next_bar_x, next_bar_o = p_x_bar, p_o_bar
                        next_off_x, next_off_o = p_x_off, p_o_off

                        next_pts[p_from] -= player_sign
                        if temp_points[target_point] == -player_sign: # Hit blot
                            next_pts[target_point] = player_sign
                            next_pts[opponent_actual_bar_index] += 1
                            if player == self.PLAYER_X: next_bar_o +=1
                            else: next_bar_x +=1
                        else:
                            next_pts[target_point] += player_sign
                        
                        possible_single_moves.append((
                            p_from, target_point, next_pts,
                            next_bar_x, next_bar_o, next_off_x, next_off_o
                        ))
        return possible_single_moves

    def _generate_recursive_plays(self, player, dice_list,
                                 current_pts_arr, bar_x, bar_o, off_x, off_o,
                                 moves_so_far, all_plays_list, memo):
        """
        Recursively generates all possible sequences of moves.
        A "play" is a list of (from_loc, to_loc) tuples.
        """
        state_tuple = (tuple(current_pts_arr), bar_x, bar_o, off_x, off_o, tuple(sorted(dice_list)))
        if not dice_list:
            if moves_so_far: # Only add non-empty move sequences
                all_plays_list.add(tuple(moves_so_far))
            return

        if state_tuple in memo:
            # If this state and dice combo has been processed, and it led to plays,
            # those plays would have been built on top of moves_so_far.
            # This memoization is tricky because of moves_so_far.
            # A simpler memo might just store if a state+dice can lead to *any* further moves.
            # For now, let's proceed without complex memoization here, relying on set for all_plays_list.
            pass


        made_a_move_with_current_dice_set = False
        unique_dice_in_list = sorted(list(set(dice_list))) # Try each unique die value

        for die in unique_dice_in_list:
            potential_next_steps = self._calculate_single_potential_moves(
                player, die, current_pts_arr, bar_x, bar_o, off_x, off_o
            )
            if potential_next_steps:
                made_a_move_with_current_dice_set = True
                
                temp_dice_list = list(dice_list) # Modifiable copy
                temp_dice_list.remove(die)

                for from_l, to_l, next_pts, n_bar_x, n_bar_o, n_off_x, n_off_o in potential_next_steps:
                    self._generate_recursive_plays(
                        player, temp_dice_list, next_pts,
                        n_bar_x, n_bar_o, n_off_x, n_off_o,
                        moves_so_far + [(from_l, to_l)], all_plays_list, memo
                    )
        
        if not made_a_move_with_current_dice_set and moves_so_far:
            # No more moves possible with the remaining dice from this state.
            # The current moves_so_far is a complete play.
            all_plays_list.add(tuple(moves_so_far))


    def get_legal_moves(self, player, dice_roll):
        """
        Generates all legal move sequences for the given player and dice roll.
        A move sequence is a list of (from_loc, to_loc) tuples.
        Example: dice_roll = (3, 5)
        Returns: [[(24, 21), (21, 16)], [(13, 10), (10, 5)], ...]
        """
        d1, d2 = dice_roll
        is_doubles = (d1 == d2)
        
        initial_dice = [d1, d1, d1, d1] if is_doubles else [d1, d2]
        
        all_plays_set = set() # Using a set to store tuples of moves to handle duplicates
        memo = {} # For memoization if we enhance it later

        # Initial state for recursion
        current_points_copy = self.points.copy()
        current_bar_x = self.points[self.BAR_X_INDEX]
        current_bar_o = self.points[self.BAR_O_INDEX]
        current_off_x = self.off_x
        current_off_o = self.off_o

        self._generate_recursive_plays(player, initial_dice,
                                       current_points_copy, current_bar_x, current_bar_o,
                                       current_off_x, current_off_o,
                                       [], all_plays_set, memo)
        
        # If not doubles, also try the other order of dice if they are different,
        # as it might allow playing both dice when one order doesn't.
        # The recursive function explores unique dice, so this might be redundant
        # if the recursion correctly handles trying all dice.
        # However, the "must use both dice" rule is subtle.
        # Let's ensure all paths are explored.
        if not is_doubles and d1 != d2:
             self._generate_recursive_plays(player, [d2, d1], # Try reversed dice
                                       current_points_copy, current_bar_x, current_bar_o,
                                       current_off_x, current_off_o,
                                       [], all_plays_set, memo)


        if not all_plays_set:
            return [] # No moves possible

        # Filter plays:
        # 1. Find the maximum number of dice used in any play
        max_dice_used = 0
        if all_plays_set:
             max_dice_used = max(len(play) for play in all_plays_set)

        # Keep only plays that use the maximum number of dice
        candidate_plays = [list(play) for play in all_plays_set if len(play) == max_dice_used]

        # 2. If not doubles, and only one die was used (max_dice_used == 1),
        #    and two dice were rolled, ensure the higher die was played if possible.
        num_dice_rolled = 4 if is_doubles else 2
        if not is_doubles and max_dice_used == 1 and num_dice_rolled == 2:
            higher_die = max(d1, d2)
            lower_die = min(d1, d2)

            # Check if higher die could have been played as a single move
            can_play_higher_die = bool(self._calculate_single_potential_moves(
                player, higher_die, current_points_copy,
                current_bar_x, current_bar_o, current_off_x, current_off_o
            ))

            if can_play_higher_die:
                # Filter to keep only plays that effectively used the higher die.
                # This means the single move in the play corresponds to the higher_die's effect.
                final_plays = []
                for play in candidate_plays: # play is [(from, to)]
                    move_from, move_to = play[0]
                    # This check is an approximation. A more robust way is to see if
                    # the (from,to) is among the _calculate_single_potential_moves for higher_die.
                    # For simplicity here, we assume if higher die was playable, any 1-move is it.
                    # This part needs careful validation or a more direct way to tag which die was used.
                    # Let's assume _generate_recursive_plays prioritizes longer sequences.
                    # If higher die was playable, it should have been part of some sequence.
                    # The current filtering for max_dice_used should handle this implicitly if
                    # _generate_recursive_plays correctly explores paths.
                    # The critical part is that if ONLY one die can be played, it MUST be the higher one if possible.
                    
                    # Re-evaluate: if max_dice_used is 1, and higher_die was playable,
                    # then all plays in candidate_plays *must* be plays of the higher die.
                    # If higher_die was NOT playable, then they must be plays of the lower_die.
                    
                    # Let's test if the single move in `play` could have been made by `higher_die`
                    is_this_play_by_higher_die = False
                    for h_from, h_to, _, _, _, _, _ in self._calculate_single_potential_moves(
                            player, higher_die, current_points_copy,
                            current_bar_x, current_bar_o, current_off_x, current_off_o):
                        if (move_from, move_to) == (h_from, h_to):
                            is_this_play_by_higher_die = True
                            break
                    if is_this_play_by_higher_die:
                        final_plays.append(play)
                
                if final_plays: # If any play matched the higher die
                    return final_plays
                else: # This case should ideally not happen if can_play_higher_die is true
                      # and candidate_plays is not empty. It implies the 1-move plays
                      # were by lower die, but higher was possible.
                      # Fallback to candidate_plays, but this indicates a logic gap.
                      # For now, if higher was possible, we expect candidate_plays to be ONLY higher die moves.
                      return candidate_plays


            # If higher die could not be played, then any 1-move play (using lower die) is fine.
            # candidate_plays would already contain these.
            return candidate_plays
        
        return candidate_plays


    def make_move(self, player, move_sequence):
        """
        Applies a given move sequence to the board.
        Assumes move_sequence is a legal move obtained from get_legal_moves.
        move_sequence is a list of (from_loc, to_loc) tuples.
        """
        if not move_sequence:
            return

        player_sign = 1 if player == self.PLAYER_X else -1
        player_bar_idx = self.BAR_X_INDEX if player == self.PLAYER_X else self.BAR_O_INDEX
        opponent_bar_idx = self.BAR_O_INDEX if player == self.PLAYER_X else self.BAR_X_INDEX
        player_off_attr = "off_x" if player == self.PLAYER_X else "off_o"
        
        player_bar_loc_const = self.BAR_X_INDEX if player == self.PLAYER_X else self.BAR_O_INDEX
        player_off_target_const = self.PLAYER_X_OFF_TARGET if player == self.PLAYER_X else self.PLAYER_O_OFF_TARGET

        for from_loc, to_loc in move_sequence:
            # 1. Decrement checker from source
            if from_loc == player_bar_loc_const:
                self.points[player_bar_idx] -= 1
            else: # From a point on the board
                self.points[from_loc] -= player_sign

            # 2. Increment checker at destination or bear off
            if to_loc == player_off_target_const: # Bearing off
                setattr(self, player_off_attr, getattr(self, player_off_attr) + 1)
            else: # Moving to a point
                # Check for hit
                if self.points[to_loc] == -player_sign: # Opponent's blot
                    self.points[to_loc] = player_sign # Player takes the point
                    self.points[opponent_bar_idx] += 1 # Opponent checker to bar
                else:
                    self.points[to_loc] += player_sign
        
        # Sanity check for bar counts (optional)
        # if self.points[self.BAR_X_INDEX] < 0 or self.points[self.BAR_O_INDEX] < 0 :
        #     raise ValueError("Bar count became negative after move.")


    def _get_char_for_display(self, point_value, display_row_from_base, max_rows=5):
        if point_value == 0: return ' '
        player_char = 'X' if point_value > 0 else 'O'
        count = abs(point_value)
        if count > display_row_from_base:
            if count > max_rows and display_row_from_base == max_rows - 1:
                # If more checkers than rows, show count on the last row
                return str(count % 10) if count > 9 else str(count) # Show last digit or full count
            return player_char
        return ' '

    def __str__(self):
        """Provides a string representation of the board, similar to GNU Backgammon."""
        lines = []
        max_display_rows = 5

        # Player O's information
        lines.append(f"Player O (O) Off: {self.off_o:2d}  Bar: {self.points[self.BAR_O_INDEX]:2d}   (moves 1->24)")
        lines.append("  +13-14-15-16-17-18------BAR------19-20-21-22-23-24--+")

        # Top half of the board (points 13-24)
        # Display checkers from top of the stack (row 0) down to base (row max_display_rows-1)
        for r_disp in range(max_display_rows -1, -1, -1): # display_row_from_top: 4 down to 0
            line = "  |"
            # Points 13-18 (O's outer board, X's outer board)
            for p_idx in range(13, 19):
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"
            
            o_bar_char = self._get_char_for_display(self.points[self.BAR_O_INDEX], r_disp, max_display_rows) if self.points[self.BAR_O_INDEX] > 0 else ' '
            line += f" {o_bar_char:^5} |" 

            # Points 19-24 (O's home board, X's starting area)
            for p_idx in range(19, 25):
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"
            lines.append(line)

        # Middle separator with textual bar counts
        lines.append(f"  |------------------| BAR O:{self.points[self.BAR_O_INDEX]:<2} X:{self.points[self.BAR_X_INDEX]:<2} |------------------|")

        # Bottom half of the board (points 12-1)
        # Display checkers from base of the stack (row 0) up to top (row max_display_rows-1)
        for r_disp in range(max_display_rows): # display_row_from_base: 0 up to 4
            line = "  |"
            # Points 12-7 (X's outer board, O's outer board)
            for p_idx in range(12, 6, -1):
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"

            x_bar_char = self._get_char_for_display(self.points[self.BAR_X_INDEX], r_disp, max_display_rows) if self.points[self.BAR_X_INDEX] > 0 else ' '
            line += f" {x_bar_char:^5} |" 
            
            # Points 6-1 (X's home board, O's starting area)
            for p_idx in range(6, 0, -1):
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"
            lines.append(line)

        lines.append("  +12-11-10--9--8--7-------BAR-------6--5--4--3--2--1--+")
        # Player X's information
        lines.append(f"Player X (X) Off: {self.off_x:2d}  Bar: {self.points[self.BAR_X_INDEX]:2d}   (moves 24->1)")
        
        return "\n".join(lines)

