import heapq
import matplotlib.pyplot as plt

def heuristic(start, goal):
    # 计算曼哈顿距离作为启发式函数
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def MazeAstar(n, m, maze):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 每个单元格的最小距离
    dist = {(i, j): float('inf') for i in range(n) for j in range(m)}
    dist[(0, 0)] = 0

    # 优先队列 (f = g + h, cell)
    pq = [(0 + heuristic((0, 0), (m - 1, n - 1)), (0, 0))]

    explored_cells = [(0, 0)]

    parent = {}

    while pq:
        f, (row, col) = heapq.heappop(pq)
        g = f - heuristic((row, col), (m - 1, n - 1))

        # 检查是否到达右下角
        if row == n - 1 and col == m - 1:
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return g, path, explored_cells

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 检查新位置是否在迷宫内
            if 0 <= new_row < n and 0 <= new_col < m:
                new_dist = g + 1

                if new_dist < dist[(new_row, new_col)] and maze[new_row][new_col] == 0:
                    dist[(new_row, new_col)] = new_dist
                    heapq.heappush(pq, (new_dist + heuristic((new_row, new_col), (m - 1, n - 1)), (new_row, new_col)))
                    explored_cells.append((new_row, new_col))
                    parent[(new_row, new_col)] = (row, col)

    return -1, [], set()

def visualize_maze_with_path(maze, path, explored_cells):
    plt.figure(figsize=(len(maze[0]), len(maze)))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=8, color='red', linewidth=3)

    max_alpha = 0.8
    min_alpha = 0.2
    alpha_step = (max_alpha - min_alpha) / len(explored_cells)

    current_alpha = max_alpha
    for cell in explored_cells:
        plt.fill([cell[1] - 0.5, cell[1] + 0.5, cell[1] + 0.5, cell[1] - 0.5],
                 [cell[0] - 0.5, cell[0] - 0.5, cell[0] + 0.5, cell[0] + 0.5],
                 color = 'blue', alpha = current_alpha)
        current_alpha -= alpha_step

    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)

    plt.axis('on')
    plt.show()

n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

result, path, explored_cells = MazeAstar(n, m, maze)
print(result)
# print("Minimum number of moves:", result)
# print("Path:", path)
# print("Explored cells:", explored_cells)

visualize_maze_with_path(maze, path, explored_cells)
