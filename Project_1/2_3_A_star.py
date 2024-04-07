import heapq

def count_inversions(board):
    inversion_count = 0
    board_1d = [x for x in board if x != 'x']
    for i in range(len(board_1d)):
        for j in range(i + 1, len(board_1d)):
            if board_1d[i] > board_1d[j]:
                inversion_count += 1
    return inversion_count

def generate_moves(board):
    moves = []
    for i in range(len(board)):
        if board[i] == 'x':
            blank_index = i
    blank_i, blank_j = divmod(blank_index, 3)
    for move, dir_name in [((0, 1), 'r'), ((0, -1), 'l'), ((1, 0), 'd'), ((-1, 0), 'u')]:
        new_i, new_j = blank_i + move[0], blank_j + move[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_blank_index = new_i * 3 + new_j
            new_board = board[:]
            new_board[blank_index], new_board[new_blank_index] = new_board[new_blank_index], new_board[blank_index]
            moves.append((new_board, dir_name))
    return moves

def heuristic_cost_estimate(board, target_state):
    count = 0
    for i in range(len(board)):
        if board[i] != target_state[i]:
            count += 1
    return count

def a_star(board, target_state):
    open_set = [(0 + heuristic_cost_estimate(board, target_state), board, "")]
    heapq.heapify(open_set)
    closed_set = set()
    while open_set:
        f, current_board, moves_so_far = heapq.heappop(open_set)
        if current_board == target_state:
            return moves_so_far
        if tuple(current_board) not in closed_set:
            closed_set.add(tuple(current_board))
            for next_board, move_dir in generate_moves(current_board):
                g = len(moves_so_far) + 1
                h = heuristic_cost_estimate(next_board, target_state)
                f = g + h
                heapq.heappush(open_set, (f, next_board, moves_so_far + move_dir))

    return "unsolvable"

target_state = ['1', '2', '3', '4', '5', '6', '7', '8', 'x']
input_board = input().split()
if count_inversions(input_board) % 2 == 0:
    steps = a_star(input_board, target_state)
    if steps != "unsolvable":
        print(steps)
    else:
        print("-1000")
else:
    print("unsolvable")
