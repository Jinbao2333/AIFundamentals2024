def is_solvable(board):
    board_1d = board.split()
    board_1d = [int(x) if x != 'x' else 0 for x in board_1d]
    board_2d = [board_1d[i:i+3] for i in range(0, len(board_1d), 3)]

    target_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

    def dfs(current_state):
        stack = [(current_state, set())]
        
        while stack:
            current_state, visited = stack.pop()
            if current_state == target_state:
                return True
            
            visited.add(tuple(current_state))
            blank_index = current_state.index(0)
            row, col = divmod(blank_index, 3)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 3 and 0 <= new_col < 3:
                    new_state = current_state[:]
                    new_blank_index = new_row * 3 + new_col
                    new_state[blank_index], new_state[new_blank_index] = new_state[new_blank_index], new_state[blank_index]
                    if tuple(new_state) not in visited:
                        stack.append((new_state, visited))
        return False
    
    initial_state = board_2d
    
    return dfs(sum(initial_state, []))

input_board = input()
if is_solvable(input_board):
    print("1")
else:
    print("0")
