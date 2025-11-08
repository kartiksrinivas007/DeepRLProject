from envs import connect4
from dataclasses import asdict
from utils import compose
import torch


if __name__ == "__main__":

        board = torch.randint(-1, 2, size = (6,7))
        connect4.print_board(board)
        connect4.get_winner(board)
        
        # torch.compile this function to be fast
        env_creation_func = torch.compile(torch.vmap(compose(asdict, connect4.env_reset), in_dims=0))

        
        pass

    
    
