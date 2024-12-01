# draft 2
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import get_directions, random_move, count_capture, count_capture_dir, execute_move, check_endgame, get_valid_moves


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.depth = 3 
        self.time_limit = 1.9 
        self.start_time = None
        self.state_dict = {} # check if board state has been previously evaluated

    def step(self, chess_board, player, opponent):
        """
        - chess_board: a numpy array of shape (board_size, board_size)
          where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
          and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).
        """
        self.start_time = time.time()
        valid_moves = get_valid_moves(chess_board, player)
        best_move = None
        best_score = -float('inf')
        alpha = -float('inf') # best score for maximizer
        beta = float('inf') # best score for minimizer

        # search deeper when there are less moves remaining
        empty_spaces = np.sum(chess_board == 0)
        if empty_spaces < 15: 
            self.depth = 5
        elif empty_spaces < 30: 
            self.depth = 4
        else: 
            self.depth = 3

        for move in valid_moves:
            board_cpy = deepcopy(chess_board)
            execute_move(board_cpy, move, player)

            # run alpha-beta game search for every valid move 
            score = self.alpha_beta(board_cpy, self.depth, False, alpha, beta, player, opponent)
            if score > best_score:
                best_score = score
                best_move = move

            # return the best move so far if time runs out
            if time.time() - self.start_time > self.time_limit:
                break

        print("My AI's turn took", time.time() - self.start_time, "seconds.")
        return best_move

    def alpha_beta(self, board, depth, is_maximizing, alpha, beta, player, opponent):
        if time.time() - self.start_time > self.time_limit or depth == 0:
            return self.heuristics(board, player, opponent)

        key = board.tostring()
        if key in self.state_dict:
            return self.state_dict[key]

        valid_moves = get_valid_moves(board, player if is_maximizing else opponent)
        if not valid_moves:
            return self.heuristics(board, player, opponent)

        if is_maximizing:
            max_child = -float('inf')
            for move in valid_moves:
                board_cpy = deepcopy(board)
                execute_move(board_cpy, move, player)
                child = self.alpha_beta(board_cpy, depth - 1, False, alpha, beta, player, opponent)
                max_child = max(max_child, child)
                alpha = max(alpha, child)
                if beta <= alpha:
                    break # prune
            self.state_dict[key] = max_child
            return max_child
        else:
            min_child = float('inf')
            for move in valid_moves:
                board_cpy = deepcopy(board)
                execute_move(board_cpy, move, opponent)
                child = self.alpha_beta(board_cpy, depth - 1, True, alpha, beta, player, opponent)
                min_child = min(min_child, child)
                beta = min(beta, child)
                if beta <= alpha:
                    break # prune
            self.state_dict[key] = min_child
            return min_child

    def heuristics(self, board, player, opponent):
        M = board.shape[0] # size of the board
        M_idx = M - 1

        # prioritize corners
        # corner heuristic = (100 × player's corners) − (100 × opponent's corners)
        corners = [(0, 0), (0, M_idx), (M_idx, 0), (M_idx, M_idx)]
        corner_h = 0
        for corner in corners:
            if board[corner[0], corner[1]] == player:
                corner_h += 100
            elif board[corner[0], corner[1]] == opponent:
                corner_h -= 100

        # prioritize tiles that can't be flipped
        # unflippable heuristic = (5 × player's unflippable pieces) − (5 × opponent's unflippable pieces)
        unflip_h = 0
        for r in range(M):
            for c in range(M):
                if self.is_unflippable(board, r, c, player):
                    unflip_h += 5
                elif self.is_unflippable(board, r, c, opponent):
                    unflip_h -= 5

        # prioritize having more move options
        # mobility heuristic = (10 × player's moves) − (10 × opponent's moves)
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility_h = (player_moves - opponent_moves) * 10

        return corner_h + unflip_h + mobility_h

    # check if piece placement is unflippable 
    def is_unflippable(self, board, r, c, player):
        if board[r, c] != player:
            return False  
        
        # check all directions to see if the piece can be captured
        directions = get_directions()
        for dx, dy in directions:
            captured = count_capture_dir(board, (r, c), 3 - player, (dx, dy))
            if captured > 0:
                return False  
        return True 
