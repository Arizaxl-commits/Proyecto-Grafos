"""
stress_test.py

Script independiente para pruebas de rendimiento (sin interfaz web).
Ejecutar con:  python stress_test.py

Genera un reporte en consola y guarda los resultados en 'resultados_rendimiento.txt'.
"""

import time
import random
import functools
import networkx as nx



def dijkstra_test(graph, origin, destination):
    try:
        length = nx.dijkstra_path_length(graph, origin, destination, weight="weight")
        return length
    except Exception:
        return None


def has_euler_circuit(graph):
    if not nx.is_connected(graph):
        return False
    return all(d % 2 == 0 for _, d in graph.degree())


def hamiltonian_cycle_dp(graph, start):
    nodes = list(graph.nodes())
    n     = len(nodes)
    idx   = {node: i for i, node in enumerate(nodes)}
    s     = idx[start]
    INF   = float("inf")

    dist = [[INF] * n for _ in range(n)]
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1)
        dist[idx[u]][idx[v]] = w
        dist[idx[v]][idx[u]] = w

    FULL = (1 << n) - 1

    @functools.lru_cache(maxsize=None)
    def dp(mask, pos):
        if mask == FULL:
            return dist[pos][s], [s]
        best_cost = INF
        best_next = -1
        for nxt in range(n):
            if mask & (1 << nxt):
                continue
            if dist[pos][nxt] == INF:
                continue
            cost, _ = dp(mask | (1 << nxt), nxt)
            total   = dist[pos][nxt] + cost
            if total < best_cost:
                best_cost = total
                best_next = nxt
        if best_next == -1:
            return INF, []
        _, tail = dp(mask | (1 << best_next), best_next)
        return best_cost, [best_next] + tail

    dp.cache_clear()
    cost, _ = dp(1 << s, s)
    return cost if cost != INF else None


def random_complete_graph(n, seed=42):
    rng = random.Random(seed)
    G   = nx.Graph()
    names = [f"N{i}" for i in range(n)]
    G.add_nodes_from(names)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(names[i], names[j], weight=round(rng.uniform(1, 10), 2))
    return G


def run_tests():
    sizes   = [5, 10, 20]
    results = []

    header = f"{'Nodos':>6} | {'Dijkstra (ms)':>14} | {'Euler check (ms)':>17} | {'Hamiltoniano (s)':>17} | {'Ham. cache hits':>15}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for n in sizes:
        G     = random_complete_graph(n, seed=42)
        start = "N0"
        end   = f"N{n-1}"

        t0 = time.perf_counter()
        dijkstra_test(G, start, end)
        t_dij = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        has_euler_circuit(G)
        t_euler = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        hamiltonian_cycle_dp(G, start)
        t_ham = time.perf_counter() - t0

        t0 = time.perf_counter()
        hamiltonian_cycle_dp(G, start)
        t_ham_cached = time.perf_counter() - t0

        row = {
            "n":          n,
            "dijkstra_ms": round(t_dij, 4),
            "euler_ms":    round(t_euler, 4),
            "ham_s":       round(t_ham, 4),
            "ham_cached":  round(t_ham_cached * 1000, 4),
        }
        results.append(row)

        speedup = t_ham / t_ham_cached if t_ham_cached > 0 else float("inf")
        print(
            f"{n:>6} | {row['dijkstra_ms']:>14.4f} | {row['euler_ms']:>17.4f} | "
            f"{row['ham_s']:>17.4f} | {row['ham_cached']:>12.4f} ms  (×{speedup:.0f})"
        )

    print(sep)
    print("\nAnálisis de complejidad:")
    print("  Dijkstra       → O((V+E) log V)  — crecimiento casi constante")
    print("  Euler check    → O(V)             — crecimiento lineal")
    print("  Hamiltoniano   → O(2^n · n²)      — crecimiento EXPONENCIAL")
    print("\nCon caché (lru_cache) la segunda llamada es instantánea.")
    print("Sin caché, un grafo de 20 nodos podría tardar > 1 minuto.\n")

    with open("resultados_rendimiento.txt", "w", encoding="utf-8") as f:
        f.write("REPORTE DE PRUEBAS DE RENDIMIENTO\n")
        f.write("Proyecto: RutaBuga — Teoría de Grafos\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Nodos':>6} | {'Dijkstra (ms)':>14} | {'Euler check (ms)':>17} | {'Hamiltoniano (s)':>17}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['n']:>6} | {r['dijkstra_ms']:>14.4f} | {r['euler_ms']:>17.4f} | {r['ham_s']:>17.4f}\n")
        f.write("\nResultados guardados correctamente.\n")

    print("📄 Resultados guardados en 'resultados_rendimiento.txt'")


if __name__ == "__main__":
    run_tests()
