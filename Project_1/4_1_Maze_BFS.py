from collections import deque
import matplotlib.pyplot as plt

def min_moves_to_bottom_right(n, m, maze):
    # Define directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Create a visited set to keep track of visited cells
    visited = set()

    # Create a dictionary to store the parent of each cell
    parent = {}

    # Create a queue for BFS
    queue = deque([(0, 0)])  # (row, column)

    while queue:
        row, col = queue.popleft()

        # Check if we have reached the bottom-right corner
        if row == n - 1 and col == m - 1:
            # Reconstruct the path
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return len(path) - 1, path, visited

        # Explore all possible directions
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check if the new position is within the maze and not visited yet
            if 0 <= new_row < n and 0 <= new_col < m and maze[new_row][new_col] == 0 and (new_row, new_col) not in visited:
                # Mark the new position as visited, record its parent, and enqueue it
                visited.add((new_row, new_col))
                parent[(new_row, new_col)] = (row, col)
                queue.append((new_row, new_col))

    # If no path is found, return -1
    return -1, [], visited

def visualize_maze_with_path(maze, path, explored_cells):
    plt.figure(figsize=(len(maze[0]), len(maze)))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=8, color='red', linewidth=3)

    # Color the explored cells
    for cell in explored_cells:
        plt.fill([cell[1]-0.5, cell[1] + 0.5, cell[1] + 0.5, cell[1]-0.5], [cell[0]-0.5, cell[0]-0.5, cell[0] + 0.5, cell[0] + 0.5], color = 'blue', alpha = 0.5)

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
result, path, explored_cells = min_moves_to_bottom_right(n, m, maze)
print("Minimum number of moves:", result)
print("Path:", path)
print("Explored cells:", explored_cells)
visualize_maze_with_path(maze, path, explored_cells)