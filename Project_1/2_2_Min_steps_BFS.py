from collections import deque
import numpy as np

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
    for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_i, new_j = blank_i + move[0], blank_j + move[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_blank_index = new_i * 3 + new_j
            new_board = board[:]
            new_board[blank_index], new_board[new_blank_index] = new_board[new_blank_index], new_board[blank_index]
            moves.append(new_board)
    return moves

def bfs(board, target_state):
    queue = deque([(board, 0)])
    visited = set()
    while queue:
        current_board, steps = queue.popleft()
        if current_board == target_state:
            return steps
        visited.add(tuple(current_board))
        for move in generate_moves(current_board):
            if tuple(move) not in visited:
                queue.append((move, steps + 1))
                visited.add(tuple(move))
    return -1

target_state = ['1', '2', '3', '4', '5', '6', '7', '8', 'x']
input_board = input().split()
if count_inversions(input_board) % 2 == 0:
    steps = bfs(input_board, target_state)
    if steps != -1:
        print(steps)
    else:
        print("-1000")
else:
    print("-1")