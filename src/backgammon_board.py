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
    # These are 'locations' for from/to in a move tuple, chosen to not conflict with point indices 0-24
    PLAYER_X_OFF_TARGET = -1 # Moving TO X's off-board area
    PLAYER_O_OFF_TARGET = 25 # Moving TO O's off-board area

    NUM_CHECKERS_PER_PLAYER = 15 # Standard Backgammon has 15 checkers per player

    def __init__(self):
        """Initializes an empty board and then resets it to the standard starting position."""
        self.points = np.zeros(self.NUM_POINTS + 2, dtype=int)
        self.off_x = 0
        self.off_o = 0
        self.reset()

    def reset(self):
        """Resets the board to the standard starting position."""
        self.points = np.zeros(self.NUM_POINTS + 2, dtype=int)
        self.off_x = 0
        self.off_o = 0

        # Standard Backgammon initial setup (15 checkers per player)
        # Player X (positive) moves from 24 -> 1
        self.points[24] = 2
        self.points[13] = 5
        self.points[8]  = 3
        self.points[6]  = 5

        # Player O (negative) moves from 1 -> 24
        self.points[1]  = -2
        self.points[12] = -5
        self.points[17] = -3
        self.points[19] = -5

        # Verify total checkers after reset
        self._verify_checker_counts()

    def _verify_checker_counts(self):
        """Internal helper to assert correct number of checkers."""
        sum_x_checkers = self.points[self.BAR_X_INDEX] + \
                         np.sum(self.points[1:self.NUM_POINTS+1][self.points[1:self.NUM_POINTS+1] > 0]) + \
                         self.off_x
        sum_o_checkers = self.points[self.BAR_O_INDEX] + \
                         np.sum(np.abs(self.points[1:self.NUM_POINTS+1][self.points[1:self.NUM_POINTS+1] < 0])) + \
                         self.off_o
        assert sum_x_checkers == self.NUM_CHECKERS_PER_PLAYER, \
            f"Player X has {sum_x_checkers} checkers, expected {self.NUM_CHECKERS_PER_PLAYER}"
        assert sum_o_checkers == self.NUM_CHECKERS_PER_PLAYER, \
            f"Player O has {sum_o_checkers} checkers, expected {self.NUM_CHECKERS_PER_PLAYER}"


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

    def _get_current_board_tuple_state(self, current_points_arr, current_bar_x, current_bar_o, current_off_x, current_off_o):
        """Returns a hashable tuple representing the current board state for play identification."""
        return (
            tuple(current_points_arr),
            current_bar_x,
            current_bar_o,
            current_off_x,
            current_off_o
        )

    def _are_all_checkers_home(self, player, points_arr, player_bar_count):
        """
        Checks if all of a player's checkers are in their home board or off.
        Assumes points_arr includes bar counts at BAR_X_INDEX and BAR_O_INDEX.
        """
        if player_bar_count > 0:
            return False

        if player == self.PLAYER_X:
            # X's home board is 1-6. Checkers outside this range (7-24) are not home.
            for i in range(7, self.NUM_POINTS + 1):
                if points_arr[i] > 0:  # X checker outside home
                    return False
        else:  # Player O
            # O's home board is 19-24. Checkers outside this range (1-18) are not home.
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
        
        player_bar_loc = self.BAR_X_INDEX if player == self.PLAYER_X else self.BAR_O_INDEX
        player_off_target = self.PLAYER_X_OFF_TARGET if player == self.PLAYER_X else self.PLAYER_O_OFF_TARGET
        opponent_actual_bar_index = self.BAR_O_INDEX if player == self.PLAYER_X else self.BAR_X_INDEX
        player_sign = 1 if player == self.PLAYER_X else -1
        
        # Determine home board range for current player
        home_board_start, home_board_end = (1, 6) if player == self.PLAYER_X else (19, 24)

        # 1. Moves from the Bar (highest priority)
        if (player == self.PLAYER_X and p_x_bar > 0) or (player == self.PLAYER_O and p_o_bar > 0):
            target_point_from_bar = die_roll if player == self.PLAYER_X else (self.NUM_POINTS + 1 - die_roll)
            
            # Check if target point is valid (1-24)
            if 1 <= target_point_from_bar <= self.NUM_POINTS:
                # Check if target point is not blocked by opponent (<= 1 opponent checker)
                if player_sign * current_points_arr[target_point_from_bar] >= -1:
                    next_pts = current_points_arr.copy()
                    next_bar_x, next_bar_o = p_x_bar, p_o_bar
                    next_off_x, next_off_o = p_x_off, p_o_off

                    # Move checker from bar
                    if player == self.PLAYER_X: next_bar_x -= 1
                    else: next_bar_o -= 1
                    
                    # Check for hitting opponent's blot
                    if current_points_arr[target_point_from_bar] == -player_sign: # Opponent's blot
                        next_pts[target_point_from_bar] = player_sign # Player occupies point
                        if player == self.PLAYER_X: next_bar_o += 1 # Opponent's checker goes to their bar
                        else: next_bar_x += 1 # Opponent's checker goes to their bar
                    else:
                        next_pts[target_point_from_bar] += player_sign # Add checker to point
                    
                    possible_single_moves.append((
                        player_bar_loc, target_point_from_bar, next_pts,
                        next_bar_x, next_bar_o, next_off_x, next_off_o
                    ))
            return possible_single_moves # If there's a move from the bar, it must be taken

        # 2. Regular moves on the board and Bearing Off
        # Iterate over all possible starting points (from 1 to 24)
        points_to_check = range(self.NUM_POINTS, 0, -1) if player == self.PLAYER_X else range(1, self.NUM_POINTS + 1)
        
        for p_from in points_to_check:
            if player_sign * current_points_arr[p_from] > 0: # Player has checkers on this point
                target_point = p_from - die_roll if player == self.PLAYER_X else p_from + die_roll

                # Check for Bearing Off
                all_home = self._are_all_checkers_home(player, current_points_arr, p_x_bar if player == self.PLAYER_X else p_o_bar)
                if all_home and \
                   ((player == self.PLAYER_X and target_point < home_board_start) or \
                    (player == self.PLAYER_O and target_point > home_board_end)):
                    
                    # Bear off rules: exact roll, or overshoot from highest point in home board
                    can_bear_off = False
                    if player == self.PLAYER_X:
                        # Exact bear off (moving checker from point X by die Y lands on point 0 (off board))
                        if target_point == 0: 
                            can_bear_off = True
                        # Overshoot bear off: die roll exceeds target, and no checkers on higher points
                        elif target_point < 0: # Overshot the board
                            highest_pip_with_checker = -1
                            for p in range(home_board_end, home_board_start - 1, -1): # From 6 down to 1 (X's home)
                                if current_points_arr[p] > 0: # Found a checker
                                    highest_pip_with_checker = p
                                    break
                            if highest_pip_with_checker == p_from: # This is the highest checker (lowest pip value)
                                can_bear_off = True
                    else: # PLAYER_O
                        # Exact bear off (moving checker from point X by die Y lands on point 25 (off board))
                        if target_point == self.NUM_POINTS + 1: 
                            can_bear_off = True
                        # Overshoot bear off
                        elif target_point > self.NUM_POINTS + 1: # Overshot the board
                            lowest_pip_with_checker = -1 # For O, lowest pip means highest point index
                            for p in range(home_board_start, home_board_end + 1): # From 19 up to 24 (O's home)
                                if current_points_arr[p] < 0: # Found a checker
                                    lowest_pip_with_checker = p
                                    break
                            if lowest_pip_with_checker == p_from: # This is the lowest checker (highest pip value)
                                can_bear_off = True

                    if can_bear_off:
                        next_pts = current_points_arr.copy()
                        next_bar_x, next_bar_o = p_x_bar, p_o_bar
                        next_off_x, next_off_o = p_x_off, p_o_off

                        next_pts[p_from] -= player_sign
                        if player == self.PLAYER_X: next_off_x += 1
                        else: next_off_o += 1
                        
                        possible_single_moves.append((
                            p_from, player_off_target, next_pts,
                            next_bar_x, next_bar_o, next_off_x, next_off_o
                        ))
                    continue # Finished with this point, move to next
                
                # Check for Regular Move (within points 1-24)
                if 1 <= target_point <= self.NUM_POINTS:
                    if player_sign * current_points_arr[target_point] >= -1: # Not blocked by opponent
                        next_pts = current_points_arr.copy()
                        next_bar_x, next_bar_o = p_x_bar, p_o_bar
                        next_off_x, next_off_o = p_x_off, p_o_off

                        next_pts[p_from] -= player_sign
                        if current_points_arr[target_point] == -player_sign: # Hit opponent's blot
                            next_pts[target_point] = player_sign
                            if player == self.PLAYER_X: next_bar_o += 1 # Opponent's checker goes to bar
                            else: next_bar_x += 1 # Opponent's checker goes to bar
                        else:
                            next_pts[target_point] += player_sign
                        
                        possible_single_moves.append((
                            p_from, target_point, next_pts,
                            next_bar_x, next_bar_o, next_off_x, next_off_o
                        ))
        return possible_single_moves

    def _generate_recursive_plays(self, player, dice_list,
                                  current_pts_arr, bar_x, bar_o, off_x, off_o,
                                  moves_so_far, all_plays_info):
        """
        Recursively generates all possible sequences of moves along with their resulting board states.
        all_plays_info is a set of (tuple(move_sequence), tuple(final_board_state)).
        """
        # Base case: No dice left. This sequence of moves is complete.
        if not dice_list:
            if moves_so_far: # Only add non-empty move sequences
                final_board_state_tuple = self._get_current_board_tuple_state(current_pts_arr, bar_x, bar_o, off_x, off_o)
                all_plays_info.add((tuple(moves_so_far), final_board_state_tuple))
            return

        made_any_move_this_step = False
        # Try each unique die value from the remaining dice, prioritizing higher dice for exploring options
        unique_dice_in_list = sorted(list(set(dice_list)), reverse=True) 

        for die in unique_dice_in_list:
            potential_single_moves = self._calculate_single_potential_moves(
                player, die, current_pts_arr, bar_x, bar_o, off_x, off_o
            )

            if potential_single_moves:
                made_any_move_this_step = True
                temp_dice_list = list(dice_list)
                temp_dice_list.remove(die) # Consume this die for the current move

                for from_l, to_l, next_pts, n_bar_x, n_bar_o, n_off_x, n_off_o in potential_single_moves:
                    # Recursively explore from the new state with remaining dice
                    self._generate_recursive_plays(
                        player, temp_dice_list, next_pts,
                        n_bar_x, n_bar_o, n_off_x, n_off_o,
                        moves_so_far + [(from_l, to_l)], all_plays_info
                    )
        
        if not made_any_move_this_step:
            # If no move could be made with *any* of the remaining dice from this specific state,
            # then the current 'moves_so_far' constitutes a complete play up to this point.
            if moves_so_far: # Only add if actual moves were made in this sequence
                final_board_state_tuple = self._get_current_board_tuple_state(current_pts_arr, bar_x, bar_o, off_x, off_o)
                all_plays_info.add((tuple(moves_so_far), final_board_state_tuple))


    def get_legal_moves(self, player, dice_roll):
        """
        Generates all legal move sequences for the given player and dice roll.
        A move sequence is a list of (from_loc, to_loc) tuples.
        Returns: A list of (move_sequence, resulting_board_state_tuple)
        """
        d1, d2 = dice_roll
        is_doubles = (d1 == d2)

        # all_plays_info stores tuples: (move_sequence_tuple, final_board_state_tuple)
        all_plays_info = set()

        initial_dice_sets = []
        if is_doubles:
            initial_dice_sets.append([d1, d1, d1, d1]) # All four dice for doubles
        else:
            initial_dice_sets.append([d1, d2])
            if d1 != d2: # Only if dice are different, explore reversed order
                initial_dice_sets.append([d2, d1])

        current_points_copy = self.points.copy()
        current_bar_x = self.points[self.BAR_X_INDEX]
        current_bar_o = self.points[self.BAR_O_INDEX]
        current_off_x = self.off_x
        current_off_o = self.off_o

        for initial_dice in initial_dice_sets:
            self._generate_recursive_plays(player, initial_dice,
                                            current_points_copy, current_bar_x, current_bar_o,
                                            current_off_x, current_off_o,
                                            [], all_plays_info)
        
        if not all_plays_info:
            return [] # No moves possible

        # Step 1 & 2: Filter for maximum number of dice used
        max_dice_used = 0
        if all_plays_info:
            max_dice_used = max(len(play_seq) for play_seq, _ in all_plays_info)

        candidate_plays = [
            (list(play_seq), final_state) # Convert tuple back to list for external use
            for play_seq, final_state in all_plays_info if len(play_seq) == max_dice_used
        ]

        # Step 3: Handle the "must play higher die" rule for non-doubles, single-die plays
        num_dice_rolled = 4 if is_doubles else 2
        if not is_doubles and max_dice_used == 1 and num_dice_rolled == 2:
            higher_die = max(d1, d2)
            lower_die = min(d1, d2)

            # Check if higher die could have been played as a single move from the original board state
            higher_die_potential_moves = self._calculate_single_potential_moves(
                player, higher_die, current_points_copy,
                current_bar_x, current_bar_o, current_off_x, current_off_o
            )
            can_play_higher_die = bool(higher_die_potential_moves)

            if can_play_higher_die:
                # If higher die *can* be played, we must play it.
                # Filter candidate_plays to only include those made possible by higher_die.
                higher_die_move_tuples = {(fm, to) for fm, to, _,_,_,_,_ in higher_die_potential_moves}
                
                filtered_plays = []
                for play_seq, final_state in candidate_plays:
                    if len(play_seq) == 1: # Ensure it's a single move play
                        if tuple(play_seq[0]) in higher_die_move_tuples:
                            filtered_plays.append((play_seq, final_state))
                
                if filtered_plays:
                    return filtered_plays # These are the legal plays using the higher die
                else:
                    # This implies higher_die was possible, but no single-move play in candidate_plays matched it.
                    # This indicates an internal logic inconsistency if it happens.
                    return [] 
            else:
                # Higher die cannot be played. Now check if lower die can be played.
                lower_die_potential_moves = self._calculate_single_potential_moves(
                    player, lower_die, current_points_copy,
                    current_bar_x, current_bar_o, current_off_x, current_off_o
                )
                can_play_lower_die = bool(lower_die_potential_moves)
                
                if can_play_lower_die:
                    # Filter candidate_plays to only include those made possible by lower_die.
                    lower_die_move_tuples = {(fm, to) for fm, to, _,_,_,_,_ in lower_die_potential_moves}
                    
                    filtered_plays = []
                    for play_seq, final_state in candidate_plays:
                        if len(play_seq) == 1: # Ensure it's a single move play
                            if tuple(play_seq[0]) in lower_die_move_tuples:
                                filtered_plays.append((play_seq, final_state))
                    
                    return filtered_plays # These are the legal plays using the lower die
                else:
                    # Neither higher nor lower die can be played as a single move.
                    return [] # No moves possible.
        
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
        
        self._verify_checker_counts() # Sanity check after each move sequence


    def copy(self):
        """Returns a deep copy of the current board state."""
        new_board = BackgammonBoard()
        new_board.points = self.points.copy()
        new_board.off_x = self.off_x
        new_board.off_o = self.off_o
        return new_board


    def _get_char_for_display(self, point_value, display_row_from_base, max_rows=5):
        """Helper for __str__ to display checkers on points."""
        if point_value == 0: return ' '
        player_char = 'X' if point_value > 0 else 'O'
        count = abs(point_value)
        
        if count > max_rows: 
            # If more checkers than rows, show the count on the last row if it's the 5th checker or higher.
            # Otherwise, just show the player character.
            if display_row_from_base == max_rows - 1: # This is the top of the displayed stack
                return str(count) if count < 10 else player_char # Show actual count for 5-9, else 'X' or 'O'
            return player_char 
        elif count > display_row_from_base: 
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
        for r_disp in range(max_display_rows -1, -1, -1): # From top of stack down (display_row 4 down to 0)
            line = "  |"
            for p_idx in range(13, 19): # Left side (13-18)
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"
            
            # Bar for O
            o_bar_char = self._get_char_for_display(-self.points[self.BAR_O_INDEX], r_disp, max_display_rows) if self.points[self.BAR_O_INDEX] > 0 else ' ' 
            line += f" {o_bar_char:^5} |" 

            for p_idx in range(19, 25): # Right side (19-24)
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"
            lines.append(line)

        # Middle separator with textual bar counts
        lines.append(f"  |------------------| BAR O:{self.points[self.BAR_O_INDEX]:<2} X:{self.points[self.BAR_X_INDEX]:<2} |------------------|")

        # Bottom half of the board (points 12-1)
        for r_disp in range(max_display_rows): # From base of stack up (display_row 0 up to 4)
            line = "  |"
            for p_idx in range(12, 6, -1): # Left side (12-7)
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"

            # Bar for X
            x_bar_char = self._get_char_for_display(self.points[self.BAR_X_INDEX], r_disp, max_display_rows) if self.points[self.BAR_X_INDEX] > 0 else ' ' 
            line += f" {x_bar_char:^5} |" 
            
            for p_idx in range(6, 0, -1): # Right side (6-1)
                char = self._get_char_for_display(self.points[p_idx], r_disp, max_display_rows)
                line += f"{char:^3}|"
            lines.append(line)

        lines.append("  +12-11-10--9--8--7-------BAR-------6--5--4--3--2--1--+")
        # Player X's information
        lines.append(f"Player X (X) Off: {self.off_x:2d}  Bar: {self.points[self.BAR_X_INDEX]:2d}   (moves 24->1)")
        
        return "\n".join(lines)