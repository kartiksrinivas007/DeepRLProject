import pytest
import torch
from envs.connect4 import *

    
def build_boards():
    cases = []
    num_cases = 2
    vals = [1, -1]
    for case in range(num_cases):
        # build a board
        board = torch.zeros((6 , 7), dtype=torch.int8)
        board[3:7, 2:6] = vals[case] # diagonals
        expected_winner = vals[case]
        cases.append((board, expected_winner))

    return cases


class TestConnect4:


    @pytest.mark.parametrize("input_board, expected_winner", build_boards())
    def test_basic_connect_4_env(self, input_board, expected_winner):
        winner = get_winner(input_board) # type:ignore
        assert torch.allclose(winner, torch.tensor(expected_winner, dtype=torch.int8))
        pass
    