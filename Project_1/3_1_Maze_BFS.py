from collections import deque
import matplotlib.pyplot as plt

def MazeBFS(n, m, maze):

    directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]

    visited = [(0, 0)]

    parent = {}

    queue = deque([(0, 0)])

    while queue:
        row, col = queue.popleft()

        if row == n - 1 and col == m - 1:
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return len(path) - 1, path, visited

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < m and maze[new_row][new_col] == 0 and (new_row, new_col) not in visited:
                visited.append((new_row, new_col))
                parent[(new_row, new_col)] = (row, col)
                queue.append((new_row, new_col))

    return -1, [], visited

def visualize_maze_with_path(maze, path, explored_cells):
    plt.figure(figsize=(9, 7))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')
    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)

    max_alpha = 1
    min_alpha = 0.4
    alpha_step = (max_alpha - min_alpha) / len(explored_cells)

    current_alpha = max_alpha
    for idx, cell in enumerate(explored_cells, 1):
        plt.fill([cell[1] - 0.5, cell[1] + 0.5, cell[1] + 0.5, cell[1] - 0.5],
                [cell[0] - 0.5, cell[0] - 0.5, cell[0] + 0.5, cell[0] + 0.5],
                color='#FC8C23', alpha=current_alpha)
        plt.text(cell[1], cell[0], str(idx), ha='center', va='center', fontsize=300/len(maze[0]), color='white', fontfamily='Bahnschrift')
        current_alpha -= alpha_step
        plt.pause(2/len(explored_cells))
        plt.draw()

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=300/len(maze[0]), color='#AC1F18', linewidth=300/len(maze[0]))

    plt.axis('on')
    plt.show()

n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

result, path, explored_cells = MazeBFS(n, m, maze)
print(result)
# print("Minimum number of moves:", result)
# print("Path:", path)
# print("Explored cells:", explored_cells)
visualize_maze_with_path(maze, path, explored_cells)