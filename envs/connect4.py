
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
def env_reset(_) -> Connect4Env:
    return Connect4Env(
        board=torch.zeros((6, 7), dtype=torch.int8),
        player=torch.tensor(1, dtype=torch.int8),
        done=torch.tensor(False, dtype=torch.bool),
        reward=torch.tensor(0, dtype=torch.int8),
    )


def env_reset_dict(_) -> dict:
    return {
        "board" : torch.zeros((6, 7), dtype=torch.int8),
        "player": torch.tensor(1, dtype=torch.int8),
        "done": torch.tensor(False, dtype=torch.bool),
        "reward": torch.tensor(0, dtype=torch.int8),
    }

batched_get_winner = torch.compile(torch.vmap(get_winner, in_dims=0, out_dims=0))

def env_step_batch(env: dict, action: Action):
    """
    Batched functional step for Connect4.
    Each env in the batch is independent.

    Args:
        env: batch of environments, a dict-like or dataclass with fields:
              - board: (B, 6, 7) tensor
              - player: (B,) tensor
              - done: (B,) bool tensor
              - reward: (B,) tensor
        action: (B,) tensor of ints in [0, 6]

    Returns:
        next_env: updated env (same structure)
        reward: (B,) tensor
        done: (B,) tensor
    """
    board = env["board"]   # (B, 6, 7)
    player = env["player"]
    done = env["done"]

    B = board.shape[0]
    batch_idx = torch.arange(B)

    # --- Find the first empty row in each chosen column ---
    # For each batch i, we want board[i, :, col[i]]  → shape (B, 6)
    col_vals = board[batch_idx, :, action]
    
    # True where cell is empty
    empty_mask = col_vals == 0  # (B, 6)
    
    # torch.argmax picks first True (0s are ignored, 1st True is max along dim)
    # Reverse to find *bottom-most* empty cell if needed
    row = torch.argmax(empty_mask.float(), dim=1)  # (B,)

    # --- Check invalid moves (column full) ---
    invalid_move = col_vals[torch.arange(B), row] != 0  # (B,)

    # --- Clone board (functional) ---
    new_board = board.clone()

    # Prepare index tensors for batch scatter
    
    # Value to place (either unchanged or player's piece)
    value_to_set = torch.where(done | invalid_move,
                               new_board[batch_idx, row, action],
                               player)

    # Functional scatter update
    new_board[batch_idx, row, action] = value_to_set


    # --- Compute reward ---
    reward = torch.where(
        done,
        torch.tensor(0, dtype=torch.int8),
        torch.where(
            invalid_move,
            torch.tensor(-1, dtype=torch.int8),
            batched_get_winner(new_board) * player,
        ),
    )

    # --- Compute done ---
    board_full = torch.all(new_board[:, -1, :] != 0, dim=1)
    done = done | (reward != 0) | invalid_move | board_full

    # --- Switch player (only if not done) ---
    next_player = torch.where(done, player, -player)

    # --- Return new env ---
    next_env = {
        "board": new_board,
        "player": next_player,
        "done":done,
        "reward":reward,
    }

    return next_env, reward, done
