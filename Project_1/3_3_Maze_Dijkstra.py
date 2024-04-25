import heapq
import matplotlib.pyplot as plt

def MazeDijkstra(n, m, maze):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    dist = {(i, j): float('inf') for i in range(n) for j in range(m)}
    dist[(0, 0)] = 0

    pq = [(0, (0, 0))]

    explored_cells = [(0, 0)]

    parent = {}

    while pq:
        d, (row, col) = heapq.heappop(pq)

        if row == n - 1 and col == m - 1:
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return d, path, explored_cells

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < m:
                new_dist = d + 1

                if new_dist < dist[(new_row, new_col)] and maze[new_row][new_col] == 0:
                    dist[(new_row, new_col)] = new_dist
                    heapq.heappush(pq, (new_dist, (new_row, new_col)))
                    explored_cells.append((new_row, new_col))
                    parent[(new_row, new_col)] = (row, col)

    return -1, [], set()

def visualize_maze_with_path(maze, path, explored_cells):
    plt.figure(figsize=(8, 6))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')
    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)

    max_alpha = 0.8
    min_alpha = 0.2
    alpha_step = (max_alpha - min_alpha) / len(explored_cells)

    current_alpha = max_alpha
    for idx, cell in enumerate(explored_cells, 1):
        plt.fill([cell[1] - 0.5, cell[1] + 0.5, cell[1] + 0.5, cell[1] - 0.5],
                [cell[0] - 0.5, cell[0] - 0.5, cell[0] + 0.5, cell[0] + 0.5],
                color='#986524', alpha=current_alpha)
        plt.text(cell[1], cell[0], str(idx), ha='center', va='center', fontsize=300/len(maze[0]), color='white', fontfamily='Bahnschrift')
        current_alpha -= alpha_step
        plt.pause(2/len(explored_cells))
        plt.draw()

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=300/len(maze[0]), color='#2177B8', linewidth=300/len(maze[0]))

    plt.axis('on')
    plt.show()

n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

result, path, explored_cells = MazeDijkstra(n, m, maze)
print(result)
# print("Minimum number of moves:", result)
# print("Path:", path)
# print("Explored cells:", explored_cells)

visualize_maze_with_path(maze, path, explored_cells)