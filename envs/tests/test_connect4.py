import pytest
import torch
from envs.connect4 import *
from dataclasses import asdict
from utils import *
from torch.utils import benchmark

    
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
    
    
    def test_env_step_batched(self):
        
        NUM = 10000
        env = torch.vmap(env_reset_dict)(torch.arange(2))
        
        # take a step 
        next_env, reward, done = env_step_batch(env, action= torch.tensor([1, 2]))

        assert next_env["board"][0, 0, 1] == 1 and next_env["board"][1, 0, 2] == 1
    
    
    
    def test_connect4_compile_speed(self):
        NUM = 10000

        simple_op = torch.vmap(compose(asdict, env_reset))
        dict_op = torch.compile(torch.vmap(env_reset_dict))
        compiled_op = torch.compile(torch.vmap(compose(asdict, env_reset)))
        

        # Example test data
        test_data = torch.arange(NUM)

        print("\nWarming up compiled function...")
        # Warm-up all versions 
        _ = simple_op(test_data)
        _ = dict_op(test_data)
        _ = compiled_op(test_data)

        # 2. Create timers for eager and compiled versions
        t_eager = benchmark.Timer(
            stmt="simple_func(test_data)",
            globals={"simple_func": simple_op, "test_data": test_data},
        )

        t_dict = benchmark.Timer(
            stmt="compiled_func(test_data)",
            globals={"compiled_func": dict_op, "test_data": test_data},
        )

        t_compiled =  benchmark.Timer(
            stmt="compiled_func(test_data)",
            globals={"compiled_func": compiled_op, "test_data": test_data},
        )

        # 3. Run the benchmarks
        eager_result = t_eager.timeit(10)
        compiled_result = t_compiled.timeit(10)
        dict_result = t_dict.timeit(10)

        # 4. Print results
        eager_time_ms = eager_result.mean * 1000
        compiled_time_ms = compiled_result.mean * 1000
        dict_result_ms = dict_result.mean * 1000

        print(f"Non-compiled: {eager_time_ms:.3f} ms")
        print(f"Compiled: {compiled_time_ms:.3f} ms")
        print(f"Dict: {dict_result_ms:.3f} ms")
        pass
    
    
    

    