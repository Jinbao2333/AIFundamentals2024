from collections import deque

def shortest_path(n, m, edges):
    graph = [[] for _ in range(n + 1)]
    for a, b in edges:
        graph[a].append(b)

    visited = [False] * (n + 1)
    distance = [float('inf')] * (n + 1)
    distance[1] = 0

    queue = deque([1])
    while queue:
        node = queue.popleft()
        if node == n:
            return distance[n]
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return -1

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

print(shortest_path(n, m, edges))