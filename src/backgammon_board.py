import numpy as np

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
    BAR_X_INDEX = 0  # Index for X's checkers on their bar (hit by O)
    # Points 1-24 are indices 1-24 in self.points
    BAR_O_INDEX = 25 # Index for O's checkers on their bar (hit by X)

    # Standard initial number of checkers per player
    NUM_CHECKERS_PER_PLAYER = 15

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
        self.points[24] = 2  # Point 24 (X's starting point)
        self.points[13] = 5  # Point 13 (X's starting point)
        self.points[8]  = 3  # Point 8 (X's starting point)
        self.points[6]  = 5  # Point 6 (X's starting point)

        # Player O's initial checkers (negative numbers indicate O's checkers)
        # Player O moves from lower numbered points to higher numbered points.
        # O's home board is points 19-24.
        self.points[1]  = -2 # Point 1 (O's starting point, from X's perspective)
        self.points[12] = -5 # Point 12 (O's starting point)
        self.points[17] = -3 # Point 17 (O's starting point)
        self.points[19] = -5 # Point 19 (O's starting point)

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
            "points": self.points.tolist(),  # Convert numpy array to list for consistent output
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

    def _get_char_for_display(self, point_value, display_row_from_base, max_rows=5):
        """
        Helper function for __str__ to get the character for a cell in the board display.
        `display_row_from_base`: 0 is the checker at the base of the point, up to `max_rows-1`.
        """
        if point_value == 0:
            return ' '
        
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

