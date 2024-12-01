# my first agent
# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import get_directions, random_move, count_capture, execute_move, check_endgame, get_valid_moves



@register_agent("second_agent")
class SecondAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(SecondAgent, self).__init__()
    self.name = "SecondAgent"

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    """
    
    start_time = time.time()

    time_limit = 2.0
    depth_limit = 4 

    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
      return None 

    best_move = None
    best_score = float('-inf')
    alpha = float('-inf') 
    beta = float('inf')

    for move in valid_moves:
      board = deepcopy(chess_board)
      execute_move(board, move, player)

      # run alpha-beta game search for every valid move
      score = self.alpha_beta(board, depth_limit - 1, alpha, beta, False, player, opponent, start_time, time_limit)
     
      # update alpha and the best move if a better one is found
      if score > best_score:
        best_score = score
        best_move = move
        alpha = max(alpha, best_score)

      # don't go over time limit
      if time.time() - start_time > time_limit - 0.1:
        break

    time_taken = time.time() - start_time
    print("second_agent's turn took ", time_taken, "seconds.")
    return best_move

  def alpha_beta(self, board, depth, alpha, beta, max_player, player, opponent, start_time, time_limit):
    if depth == 0 or check_endgame(board, player, opponent)[0]:
      return self.corner_heuristic(board, player, opponent)

    if time.time() - start_time > time_limit - 0.1:
      return 0  # if time is running out

    if max_player:
      valid_moves = get_valid_moves(board, player)
    elif not max_player:
      valid_moves = get_valid_moves(board, opponent)
    
    if not valid_moves:
      return self.alpha_beta(board, depth - 1, alpha, beta, not max_player, player, opponent, start_time, time_limit)

    # max
    if max_player:
      max_eval = float('-inf')
      for move in valid_moves:
        board = deepcopy(board)
        execute_move(board, move, player)
        # recursive call 
        eval = self.alpha_beta(board, depth - 1, alpha, beta, False, player, opponent, start_time, time_limit)
        max_eval = max(max_eval, eval)
        alpha = max(alpha, eval)
        if beta <= alpha:
          break  # prune
      return max_eval
    # min
    else:
      min_eval = float('inf')
      for move in valid_moves:
        board = deepcopy(board)
        execute_move(board, move, opponent)
        # recursive call
        eval = self.alpha_beta(board, depth - 1, alpha, beta, True, player, opponent, start_time, time_limit)
        min_eval = min(min_eval, eval)
        beta = min(beta, eval)
        if beta <= alpha:
          break # prune
      return min_eval

  # try to occupy corners 
  def corner_heuristic(self, board, player, opponent):
    player_score = np.sum(board == player)
    opponent_score = np.sum(board == opponent)

    # heuristic: difference in piece counts + corner control
    M_idx = len(board) - 1
    corner_positions = [(0, 0), (0, M_idx), (M_idx, 0), (M_idx, M_idx)]
    corner_score = 0
    for r, c in corner_positions:
      # check if the corner is occupied
      if board[r, c] != 0:
          # if player, encourage them to occupy corners, add 10
          if board[r, c] == player:
              corner_score += 10
          # "penalize" player for no occupying corner, subtract 10
          else:
              corner_score -= 10
    return player_score - opponent_score + corner_score
