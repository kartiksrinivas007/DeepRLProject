
import dataclasses
from rich import print
from envs.types import *
import torch

@dataclasses.dataclass
class Env:
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
