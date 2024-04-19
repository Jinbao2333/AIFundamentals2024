import matplotlib.pyplot as plt

def MazeDFS(n, m, maze):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    stack = [(0, 0)]
    parent = {(0, 0): None}
    explored_cells = []

    while stack:
        row, col = stack.pop()
        explored_cells.append((row, col))

        if row == n - 1 and col == m - 1:
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return len(path) - 1, path, explored_cells

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < m and maze[new_row][new_col] == 0 and (new_row, new_col) not in parent:
                parent[(new_row, new_col)] = (row, col)
                stack.append((new_row, new_col))

    return -1, [], explored_cells

def visualize_maze_with_path(maze, path, explored_cells):
    plt.figure(figsize=(len(maze[0]), len(maze)))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=24, color='#FFD111', linewidth=9)

    max_alpha = 0.8
    min_alpha = 0.2
    alpha_step = (max_alpha - min_alpha) / len(explored_cells)

    current_alpha = max_alpha
    for idx, cell in enumerate(explored_cells, 1):
        plt.fill([cell[1] - 0.5, cell[1] + 0.5, cell[1] + 0.5, cell[1] - 0.5],
                [cell[0] - 0.5, cell[0] - 0.5, cell[0] + 0.5, cell[0] + 0.5],
                color='#113285', alpha=current_alpha)
        plt.text(cell[1], cell[0], str(idx), ha='center', va='center', fontsize=20, color='white', fontfamily='Bahnschrift')
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

(result, path, explored_cells) = MazeDFS(n, m, maze)
print(result)
# print("Minimum number of moves:", result)
# print("Path:", path)
# print("Explored cells:", explored_cells)
visualize_maze_with_path(maze, path, explored_cells)
