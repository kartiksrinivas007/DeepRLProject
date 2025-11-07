from envs import connect4
import torch


if __name__ == "__main__":

        board = torch.randint(-1, 2, size = (6,7))
        connect4.print_board(board)
        connect4.get_winner(board)
        pass

    
    
