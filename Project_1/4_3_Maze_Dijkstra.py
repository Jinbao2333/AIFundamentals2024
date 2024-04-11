import heapq
import matplotlib.pyplot as plt

def min_moves_to_bottom_right_dijkstra(n, m, maze):
    # Define directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Create a dictionary to store the minimum distance to each cell
    dist = {(i, j): float('inf') for i in range(n) for j in range(m)}
    dist[(0, 0)] = 0

    # Create a priority queue for Dijkstra's algorithm
    pq = [(0, (0, 0))]  # (distance, cell)

    # Create a set to store the explored cells
    explored_cells = set()

    # Create a dictionary to store the parent of each cell
    parent = {}

    while pq:
        d, (row, col) = heapq.heappop(pq)

        # Check if we have reached the bottom-right corner
        if row == n - 1 and col == m - 1:
            # Reconstruct the path
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return d, path, explored_cells

        # Explore all possible directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check if the new position is within the maze
            if 0 <= new_row < n and 0 <= new_col < m:
                # Calculate the new distance
                new_dist = d + 1  # Assuming all edges have unit weight

                # Update the distance if it's shorter
                if new_dist < dist[(new_row, new_col)] and maze[new_row][new_col] == 0:
                    dist[(new_row, new_col)] = new_dist
                    heapq.heappush(pq, (new_dist, (new_row, new_col)))
                    explored_cells.add((new_row, new_col))

    # If no path is found, return -1
    return -1, [], set()

def visualize_maze_with_path(maze, path, explored_cells):
    plt.figure(figsize=(len(maze[0]), len(maze)))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=8, color='red', linewidth=3)

    # Color the explored cells
    for cell in explored_cells:
        plt.fill([cell[1]-0.5, cell[1] + 0.5, cell[1] + 0.5, cell[1]-0.5], [cell[0]-0.5, cell[0]-0.5, cell[0] + 0.5, cell[0] + 0.5], color='blue', alpha=0.5)

    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)

    plt.axis('on')
    plt.show()

# Read input
n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

# Calculate minimum number of moves, the path, and the explored cells
result, path, explored_cells = min_moves_to_bottom_right_dijkstra(n, m, maze)
print("Minimum number of moves:", result)
print("Path:", path)
print("Explored cells:", explored_cells)

# Visualize maze with path and explored cells
visualize_maze_with_path(maze, path, explored_cells)
