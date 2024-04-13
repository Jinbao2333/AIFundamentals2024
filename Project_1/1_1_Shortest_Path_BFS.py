from collections import deque

def shortest_path(n, edges):
    graph = [[] for _ in range(n + 1)]
    for a, b in edges:
        graph[a].append(b)

    queue = deque([1])
    visited = [False] * (n + 1) 
    visited[1] = True
    distance = [float('inf')] * (n + 1)
    distance[1] = 0

    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                distance[neighbor] = distance[current] + 1

    if distance[n] != float('inf'):
        return distance[n]
    else:
        return -1

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

print(shortest_path(n, edges))
