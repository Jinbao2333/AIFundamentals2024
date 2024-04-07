def naive_dijkstra(n, m, edges):
    h = [[] for _ in range(n + 1)]
    d = [float('inf')] * (n + 1)
    d[1] = 0
    vis = [False] * (n + 1)

    for u, v, w in edges:
        h[u].append((v, w))

    for _ in range(n):
        min_dist = float('inf')
        min_idx = -1
        for i in range(1, n + 1):
            if not vis[i] and d[i] < min_dist:
                min_dist = d[i]
                min_idx = i
        if min_idx == -1:
            break
        vis[min_idx] = True
        for v, w in h[min_idx]:
            if d[v] > d[min_idx] + w:
                d[v] = d[min_idx] + w

    if d[n] != float('inf'):
        return d[n]
    return -1

n, m = map(int, input().split())
edges = []
for _ in range(m):
    x, y, z = map(int, input().split())
    edges.append((x, y, z))

print(naive_dijkstra(n, m, edges))
