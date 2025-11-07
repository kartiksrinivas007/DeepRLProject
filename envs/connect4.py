
import dataclasses
from rich import print
from envs.types import *
import torch

@dataclasses.dataclass
class Connect4Env:
    board: Board
    player: Player
    done: Done
    reward: Reward

BOARD_STRING = """
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 1   2   3   4   5   6   7
"""

def print_board(board: Board):
    board_str = BOARD_STRING
    for i in reversed(range(board.shape[0])):
        for j in range(board.shape[1]):
            board_str = board_str.replace('?', '[green]X[/green]' if board[i, j] == 1 else '[red]O[/red]' if board[i, j] == -1 else ' ', 1)
    print(board_str)



def horizontals(board: Board) -> torch.Tensor:
    return torch.stack([
        board[i, j:j+4]
        for i in range(board.shape[0])
        for j in range(board.shape[1] - 3)
    ])

def verticals(board: Board) -> torch.Tensor:
    return torch.stack([
        board[i:i+4, j]
        for i in range(board.shape[0] - 3)
        for j in range(board.shape[1])
    ])

def diagonals(board: Board) -> torch.Tensor:
    return torch.stack([
        torch.diag(board[i:i+4, j:j+4])
        for i in range(board.shape[0] - 3)
        for j in range(board.shape[1] - 3)
    ])

def antidiagonals(board: Board) -> torch.Tensor:
    return torch.stack([
        torch.diag(torch.flip(board[i:i+4, j:j+4], dims = [1]))
        for i in range(board.shape[0] - 3)
        for j in range(board.shape[1] - 3)
    ])

def get_winner(board: Board) -> Player:
    all_lines = torch.concatenate((
        horizontals(board),
        verticals(board),
        diagonals(board),
        antidiagonals(board),
    ))

    
    # x_won and o_won are 1 if the player won, 0 otherwise
    x_won = torch.any(torch.all(all_lines == 1, dim=1)).to(torch.int8)
    o_won = torch.any(torch.all(all_lines == -1, dim=1)).to(torch.int8)
    # We consider the following cases:
    # - !x_won and !o_won -> 0 - 0 = 0 -> draw OR not finished
    # - x_won and !o_won -> 1 - 0 = 1 -> Player 1 (X) won
    # - !x_won and o_won -> 0 - 1 = -1 -> Player -1 (O) won
    # - x_won and o_won -> impossible, the game would have ended earlier
    return x_won - o_won

#TODO(kartiksrinivas): Assess whether the types are generic enough
def env_reset(_):
    return Connect4Env(
        board=torch.zeros((6, 7), dtype=torch.int8),
        player=torch.tensor(1, dtype=torch.int8),
        done=torch.tensor(False, dtype=torch.bool),
        reward=torch.tensor(0, dtype=torch.int8),
    )

def env_step(env: Connect4Env, action: Action) -> tuple[Connect4Env, Reward, Done]:
    col = action

    # Find the first empty row in the column.
    # If the column is full, this will be the top row.
    # torch.argmax will give the first occurence of the maximum
    row = torch.argmax(env.board[:, col] == 0)

    # If the column is full, the move is invalid.
    invalid_move = env.board[row, col] != 0

    # Place the player's piece in the board only if the move is valid and the game is not over.

    # --- PyTorch functional-style update ---
    # JAX's .at[...].set(...) is functional (out-of-place) and returns a new array.
    # PyTorch's indexing is in-place. To replicate the JAX behavior,
    # we must clone() the board first.
    board = env.board.clone()
    
    # Determine the value to place
    value_to_set = torch.where(env.done | invalid_move, 
                               env.board[row, col],   # If game over/invalid, keep original value
                               env.player)           # Otherwise, place the player's piece

    # Update the cloned board
    board[row, col] = value_to_set
    # ----------------------------------------

    # The reward is computed as follows:
    # (Assuming get_winner is a PyTorch-compatible function)
    reward = torch.where(env.done, 
                         torch.tensor(0),  # Use torch.tensor for constants
                         torch.where(invalid_move, 
                                     torch.tensor(-1), 
                                     get_winner(board) * env.player)
                        ).to(torch.int8)

    # We end the game if:
    # * the game was already over
    # * the move won or lost the game (reward != 0)
    # * the move was invalid
    # * the board is full (draw)
    done = env.done | (reward != 0) | invalid_move | torch.all(board[-1] != 0)

    env = Connect4Env(
        board=board,
        # switch player
        player=torch.where(done, env.player, -env.player),
        done=done,
        reward=reward, # reward is embedded inside the env
    )

    return env, reward, done