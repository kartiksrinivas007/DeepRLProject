from envs.connect4 import *

if __name__ == "__main__":

        board = torch.randint(-1, 2, size = (6,7))
        print_board(board)
        get_winner(board)
        pass