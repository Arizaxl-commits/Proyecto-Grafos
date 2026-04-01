import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import functools
from itertools import permutations
import sys

st.set_page_config(
    page_title="Aplicación de Grafos",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0b0f14;
    color: #e8e2d4;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }
code, .stCode { font-family: 'IBM Plex Mono', monospace; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111820;
    border-right: 1px solid #1e2a38;
}
section[data-testid="stSidebar"] * { color: #c9c2b4 !important; }

/* Buttons */
.stButton > button {
    background: #d4a843;
    color: #0b0f14;
    border: none;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.55rem 1.4rem;
    border-radius: 2px;
    transition: background 0.2s;
}
.stButton > button:hover { background: #e8c06a; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #111820;
    border: 1px solid #1e2a38;
    border-radius: 4px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #7a8fa6 !important; font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #d4a843 !important; font-family: 'IBM Plex Mono', monospace; }

/* Text inputs */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: #111820 !important;
    color: #e8e2d4 !important;
    border: 1px solid #1e2a38 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace;
}

/* Info / success / warning boxes */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #7a8fa6;
}
.stTabs [aria-selected="true"] { color: #d4a843 !important; border-bottom: 2px solid #d4a843; }

/* Expander */
.streamlit-expanderHeader { font-family: 'IBM Plex Mono', monospace; color: #7a8fa6 !important; }
</style>
""", unsafe_allow_html=True)


def dijkstra(graph: nx.Graph, origin: str, destination: str):
    """Camino más corto usando Dijkstra de NetworkX. O((V+E) log V)."""
    try:
        path   = nx.dijkstra_path(graph, origin, destination, weight="weight")
        length = nx.dijkstra_path_length(graph, origin, destination, weight="weight")
        return path, length
    except nx.NetworkXNoPath:
        return None, None
    except nx.NodeNotFound as e:
        return None, str(e)


def has_euler_circuit(graph: nx.Graph) -> bool:
    if not nx.is_connected(graph):
        return False
    return all(d % 2 == 0 for _, d in graph.degree())


def euler_circuit(graph: nx.Graph, start: str):
    if not has_euler_circuit(graph):
        return None, None
    circuit = list(nx.eulerian_circuit(graph, source=start))
    path    = [u for u, _ in circuit] + [circuit[-1][1]]
    total   = sum(graph[u][v].get("weight", 1) for u, v in circuit)
    return path, total

def hamiltonian_cycle_dp(graph: nx.Graph, start: str):
    nodes = list(graph.nodes())
    n     = len(nodes)
    if n < 2:
        return None, None

    idx   = {node: i for i, node in enumerate(nodes)}
    s     = idx[start]

    INF = float("inf")
    dist = [[INF] * n for _ in range(n)]
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1)
        dist[idx[u]][idx[v]] = w
        dist[idx[v]][idx[u]] = w

    FULL = (1 << n) - 1

    @functools.lru_cache(maxsize=None)
    def dp(mask: int, pos: int):
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
    cost, path_idx = dp(1 << s, s)
    if cost == INF:
        return None, None

    path = [nodes[s]] + [nodes[i] for i in path_idx]
    return path, cost



DEFAULT_EDGES = """A,B,4.0
A,C,2.0
A,E,9.0
B,D,5.0
B,E,3.0
C,D,8.0
C,E,10.0
D,E,2.0
D,F,6.0
E,F,4.0
E,G,7.0
F,G,3.0
F,H,5.0
G,H,9.0
G,A,6.0
H,B,8.0
H,C,7.0"""


def parse_edges(text: str):
    edges  = []
    errors = []
    for i, line in enumerate(text.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            errors.append(f"Línea {i}: se esperaban 3 campos separados por coma.")
            continue
        try:
            w = float(parts[2])
            edges.append((parts[0], parts[1], w))
        except ValueError:
            errors.append(f"Línea {i}: el peso '{parts[2]}' no es un número válido.")
    return edges, errors


def build_graph(edges):
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G



PALETTE = {
    "bg":      "#0b0f14",
    "node":    "#1e2a38",
    "node_hl": "#d4a843",
    "edge":    "#2e3f52",
    "edge_hl": "#d4a843",
    "text":    "#e8e2d4",
    "text_dk": "#0b0f14",
}

def _gps_pos(G: nx.Graph):
    import math
    nodes = sorted(G.nodes())
    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n - math.pi / 2
        pos[node] = (math.cos(angle), math.sin(angle))
    return pos


def draw_graph(G: nx.Graph, highlight_path=None, title=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    pos = _gps_pos(G)

    highlight_edges = set()
    if highlight_path and len(highlight_path) > 1:
        highlight_edges = {
            (highlight_path[i], highlight_path[i + 1])
            for i in range(len(highlight_path) - 1)
        }
        highlight_edges |= {(v, u) for u, v in highlight_edges}

    normal_edges = [(u, v) for u, v in G.edges() if (u, v) not in highlight_edges]
    hi_edges     = [(u, v) for u, v in G.edges() if (u, v) in highlight_edges]

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color=PALETTE["edge"],
                           width=1.5, ax=ax, alpha=0.7)
    if hi_edges:
        nx.draw_networkx_edges(G, pos, edgelist=hi_edges, edge_color=PALETTE["edge_hl"],
                               width=3.5, ax=ax, alpha=1.0)

    highlight_nodes = set(highlight_path) if highlight_path else set()
    normal_nodes    = [n for n in G.nodes() if n not in highlight_nodes]
    hi_nodes        = list(highlight_nodes)

    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes,
                           node_color=PALETTE["node"], node_size=600, ax=ax)
    if hi_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=hi_nodes,
                               node_color=PALETTE["node_hl"], node_size=800, ax=ax)

    labels = {n: n.split()[0] for n in G.nodes()}   # primera palabra para no saturar
    nx.draw_networkx_labels(G, pos, labels=labels, font_color=PALETTE["text"],
                            font_size=7, font_family="monospace", ax=ax)

    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color="#7a8fa6", font_size=6,
                                 font_family="monospace", ax=ax)

    ax.set_title(title, color=PALETTE["node_hl"],
                 fontsize=11, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


if "graph_text" not in st.session_state:
    st.session_state.graph_text = DEFAULT_EDGES
if "graph" not in st.session_state:
    edges, _ = parse_edges(DEFAULT_EDGES)
    st.session_state.graph = build_graph(edges)

with st.sidebar:
    st.markdown("## 🗺️ **Simulador De Rutas**")
    st.markdown("Aplicación de teoría de grafos para análisis de rutas")
    st.divider()

    st.markdown("### Definir Grafo")
    st.caption("Formato: `NodoA, NodoB, peso` (un par por línea)")

    graph_input = st.text_area(
        "Aristas",
        value=st.session_state.graph_text,
        height=280,
        key="graph_input_area",
    )

    if st.button("⚙ Cargar Grafo"):
        edges, errors = parse_edges(graph_input)
        if errors:
            for e in errors:
                st.error(e)
        elif len(edges) == 0:
            st.warning("No se encontraron aristas válidas.")
        else:
            st.session_state.graph_text = graph_input
            st.session_state.graph      = build_graph(edges)
            st.success(f"Grafo cargado: {st.session_state.graph.number_of_nodes()} nodos, "
                       f"{st.session_state.graph.number_of_edges()} aristas.")

    if st.button("↺ Restaurar ejemplo"):
        edges, _ = parse_edges(DEFAULT_EDGES)
        st.session_state.graph_text = DEFAULT_EDGES
        st.session_state.graph      = build_graph(edges)
        st.rerun()

    st.divider()
    G = st.session_state.graph
    st.metric("Nodos", G.number_of_nodes())
    st.metric("Aristas", G.number_of_edges())
    st.metric("¿Conexo?", "Sí" if nx.is_connected(G) else "No")


st.markdown("# Cálculo de Rutas Óptimas")
st.markdown(
    "Calcula rutas usando **Dijkstra**, **Circuito de Euler** y "
    "**Ciclo Hamiltoniano** (Held-Karp con DP)."
)

G = st.session_state.graph
nodes = sorted(list(G.nodes()))

tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Dijkstra — Ruta A→B",
    "♾ Euler — Recorrer Aristas",
    "🔷 Hamiltoniano — Visitar Nodos",
    "⚡ Pruebas de Rendimiento",
])

with tab1:
    st.markdown("### Camino más corto entre dos puntos")
    st.markdown(
        "**Algoritmo de Dijkstra** — Complejidad: **O((V+E) log V)** con cola de prioridad."
    )
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("Nodo origen", nodes, key="dij_origin")
    with col2:
        destination = st.selectbox("Nodo destino", nodes,
                                   index=min(1, len(nodes)-1), key="dij_dest")

    if st.button("Calcular ruta más corta", key="btn_dijkstra"):
        if origin == destination:
            st.warning("El origen y el destino son el mismo nodo.")
        else:
            t0 = time.perf_counter()
            path, length = dijkstra(G, origin, destination)
            elapsed = time.perf_counter() - t0

            if path is None:
                st.error(f"No existe camino entre **{origin}** y **{destination}**.")
            else:
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Distancia total", f"{length:.2f} km")
                col_m2.metric("Nodos en la ruta", len(path))
                col_m3.metric("Tiempo de cómputo", f"{elapsed*1000:.3f} ms")

                st.success("  →  ".join(path))
                fig = draw_graph(G, highlight_path=path,
                                 title=f"Dijkstra: {origin} → {destination}")
                st.pyplot(fig)

    with st.expander("📖 Justificación matemática"):
        st.markdown("""
**¿Por qué Dijkstra?**

Para encontrar el camino más corto entre dos vértices en un grafo ponderado con
pesos no negativos, Dijkstra es óptimo tanto en corrección como en eficiencia.

| Complejidad | Valor |
|---|---|
| Temporal | O((V + E) log V) con heap binario |
| Espacial | O(V) para distancias + O(V) para predecesores |

La invariante del algoritmo garantiza que, cuando un nodo sale de la cola de prioridad,
su distancia estimada ya es la distancia mínima real.  
Se prefiere sobre Bellman-Ford (O(VE)) porque los pesos de tráfico/distancia
nunca son negativos en este contexto.
        """)

with tab2:
    st.markdown("### Circuito de Euler — Recorrer cada calle exactamente una vez")
    st.markdown(
        "**Algoritmo de Hierholzer** — Complejidad: **O(E)**."
    )

    start_euler = st.selectbox("Nodo de inicio / fin", nodes, key="euler_start")

    if st.button("Calcular Circuito de Euler", key="btn_euler"):
        t0 = time.perf_counter()
        if not has_euler_circuit(G):
            elapsed = time.perf_counter() - t0
            st.error(
                "❌ El grafo **no posee** un Circuito de Euler. "
                "Para que exista, todos los nodos deben tener **grado par** "
                "y el grafo debe ser **conexo**."
            )
            odd_nodes = [n for n, d in G.degree() if d % 2 != 0]
            st.info(f"Nodos con grado impar: {', '.join(odd_nodes)}")
        else:
            path, total = euler_circuit(G, start_euler)
            elapsed = time.perf_counter() - t0

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Distancia total", f"{total:.2f} km")
            col_m2.metric("Aristas recorridas", G.number_of_edges())
            col_m3.metric("Tiempo de cómputo", f"{elapsed*1000:.3f} ms")

            st.success("  →  ".join(path))
            fig = draw_graph(G, highlight_path=path,
                             title=f"Circuito de Euler desde {start_euler}")
            st.pyplot(fig)

    with st.expander("📖 Justificación matemática"):
        st.markdown("""
**Teorema de Euler (1736)**

Un grafo conexo tiene un **circuito de Euler** si y solo si *todos* sus vértices
tienen grado par.  
Tiene un **sendero de Euler** (sin circuito) si y solo si tiene *exactamente dos*
vértices de grado impar.

**Algoritmo de Hierholzer** construye el circuito en O(E):
1. Inicia un camino desde el nodo fuente.
2. Cuando queda atrapado (sin aristas sin visitar), retrocede hasta encontrar
   un vértice con aristas disponibles e inicia un subcircuito.
3. Une los subcircuitos.

**Aplicación de Grafos**: útil para planificar rutas de recolección de residuos,
inspección de todas las calles, o reparto postal donde cada calle se recorre una vez.
        """)

with tab3:
    st.markdown("### Ciclo Hamiltoniano — Visitar cada intersección exactamente una vez")
    st.markdown(
        "**Held-Karp (DP + bitmask)** — Complejidad: **O(2ⁿ · n²)**. "
        "Recomendado hasta ~20 nodos."
    )

    n_nodes = G.number_of_nodes()
    if n_nodes > 20:
        st.warning(
            f"⚠ El grafo tiene **{n_nodes} nodos**. "
            "El algoritmo Held-Karp puede tardar varios segundos o minutos. "
            "Considera reducir el grafo."
        )

    start_ham = st.selectbox("Nodo de inicio / fin", nodes, key="ham_start")

    if st.button("Calcular Ciclo Hamiltoniano", key="btn_hamilton"):
        with st.spinner("Calculando... (puede demorar para grafos grandes)"):
            t0 = time.perf_counter()
            path, total = hamiltonian_cycle_dp(G, start_ham)
            elapsed = time.perf_counter() - t0

        if path is None:
            st.error("No se encontró un Ciclo Hamiltoniano en este grafo.")
        else:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Distancia total", f"{total:.2f} km")
            col_m2.metric("Nodos visitados", G.number_of_nodes())
            col_m3.metric("Tiempo de cómputo", f"{elapsed:.4f} s")

            st.success("  →  ".join(path))
            fig = draw_graph(G, highlight_path=path,
                             title=f"Ciclo Hamiltoniano desde {start_ham}")
            st.pyplot(fig)

    with st.expander("📖 Justificación matemática"):
        st.markdown("""
**¿Por qué Held-Karp y no fuerza bruta?**

| Método | Complejidad | 20 nodos |
|---|---|---|
| Fuerza bruta | O(n!) | ≈ 2.4 × 10¹⁸ ops |
| Held-Karp (DP) | O(2ⁿ · n²) | ≈ 4.2 × 10⁸ ops |

**Idea central de Held-Karp (1962)**:

Sea `dp[S][v]` = costo mínimo para visitar exactamente los nodos en el
conjunto *S* terminando en *v*, habiendo partido del nodo fuente.

Recurrencia:
```
dp[S][v] = min over u in S-{v} of (dp[S-{v}][u] + dist[u][v])
```

La respuesta es: `min over v != s of (dp[FULL][v] + dist[v][s])`

El uso de `@functools.lru_cache` (memoización) evita recalcular subproblemas,
reduciendo la complejidad de factorial a exponencial.

**NP-Completitud**: El problema de decisión del Hamiltoniano es NP-completo
(reducción desde 3-SAT). Held-Karp no lo resuelve en tiempo polinomial, pero sí
de forma exacta y eficiente dentro de la clase de algoritmos exponenciales.
        """)

with tab4:
    st.markdown("### Pruebas de Estrés")
    st.markdown(
        "Genera grafos aleatorios de **5, 10 y 20 nodos**, muestra cómo luce cada grafo "
        "con su ruta Hamiltoniana calculada, y explica por qué el tiempo crece."
    )

    import random
    import pandas as pd

    def random_complete_graph(n: int, seed: int = 0):
        rng = random.Random(seed)
        G_r = nx.Graph()
        node_names = [f"N{i}" for i in range(n)]
        G_r.add_nodes_from(node_names)
        for i in range(n):
            for j in range(i + 1, n):
                w = round(rng.uniform(1, 10), 2)
                G_r.add_edge(node_names[i], node_names[j], weight=w)
        return G_r

    def draw_test_graph(G_t, highlight_path, title, t_dij, t_ham, n):
        """Dibuja el grafo de prueba con la ruta Hamiltoniana resaltada y métricas."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                                 gridspec_kw={"width_ratios": [2, 1]})
        fig.patch.set_facecolor(PALETTE["bg"])

        ax_g = axes[0]
        ax_g.set_facecolor(PALETTE["bg"])
        pos_t = nx.spring_layout(G_t, seed=7, k=2.2)

        hi_edges = set()
        if highlight_path and len(highlight_path) > 1:
            for i in range(len(highlight_path) - 1):
                hi_edges.add((highlight_path[i], highlight_path[i+1]))
                hi_edges.add((highlight_path[i+1], highlight_path[i]))

        normal_e = [(u, v) for u, v in G_t.edges() if (u,v) not in hi_edges]
        hi_e     = [(u, v) for u, v in G_t.edges() if (u,v) in hi_edges]

        nx.draw_networkx_edges(G_t, pos_t, edgelist=normal_e,
                               edge_color=PALETTE["edge"], width=1.2, ax=ax_g, alpha=0.5)
        if hi_e:
            nx.draw_networkx_edges(G_t, pos_t, edgelist=hi_e,
                                   edge_color=PALETTE["edge_hl"], width=3, ax=ax_g)

        hi_nodes = set(highlight_path) if highlight_path else set()
        nx.draw_networkx_nodes(G_t, pos_t,
                               nodelist=[n2 for n2 in G_t.nodes() if n2 not in hi_nodes],
                               node_color=PALETTE["node"], node_size=500, ax=ax_g)
        if hi_nodes:
            nx.draw_networkx_nodes(G_t, pos_t, nodelist=list(hi_nodes),
                                   node_color=PALETTE["node_hl"], node_size=650, ax=ax_g)

        nx.draw_networkx_labels(G_t, pos_t, font_color=PALETTE["text"],
                                font_size=8, font_family="monospace", ax=ax_g)
        edge_labels = {(u,v): f"{d['weight']:.1f}" for u,v,d in G_t.edges(data=True)}
        nx.draw_networkx_edge_labels(G_t, pos_t, edge_labels=edge_labels,
                                     font_color="#7a8fa6", font_size=6,
                                     font_family="monospace", ax=ax_g)
        ax_g.set_title(title, color=PALETTE["node_hl"], fontsize=11,
                       fontweight="bold", pad=10)
        ax_g.axis("off")

        ax_m = axes[1]
        ax_m.set_facecolor("#0d1520")
        ax_m.axis("off")

        subproblemas = (2**n) * n
        aristas      = G_t.number_of_edges()

        lines = [
            ("NODOS",          f"{n}",                   PALETTE["node_hl"]),
            ("ARISTAS",        f"{aristas}",              PALETTE["node_hl"]),
            ("",               "",                        "#ffffff"),
            ("── DIJKSTRA ──", "",                        "#7a8fa6"),
            ("Tiempo",         f"{t_dij*1000:.3f} ms",    "#5bc4a0"),
            ("Complejidad",    f"O((V+E)·logV)",          "#c9c2b4"),
            ("Operaciones",    f"≈ {int(aristas * (n**0.5)):,}",  "#c9c2b4"),
            ("",               "",                        "#ffffff"),
            ("── HAMILTONIANO ──", "",                    "#7a8fa6"),
            ("Tiempo",         f"{t_ham:.4f} s",          "#e85d5d"),
            ("Complejidad",    f"O(2ⁿ·n²)",              "#c9c2b4"),
            ("Subproblemas",   f"≈ {subproblemas:,}",     "#e85d5d"),
            ("",               "",                        "#ffffff"),
            ("── POR QUÉ TARDÓ ──", "",                  "#7a8fa6"),
        ]

        if n == 5:
            explicacion = ["Con 5 nodos,", "solo 32 sub-", "problemas.", "Casi instant."]
        elif n == 10:
            explicacion = ["Con 10 nodos,", "1,024 sub-", "problemas.", "Muy rápido."]
        else:
            explicacion = ["Con 20 nodos,", "1,048,576 sub-", "problemas.", "¡Aquí tarda!"]

        for exp_line in explicacion:
            lines.append((exp_line, "", "#e8e2d4"))

        y = 0.97
        for label, val, color in lines:
            if label == "":
                y -= 0.03
                continue
            if val == "":
                ax_m.text(0.05, y, label, transform=ax_m.transAxes,
                          color=color, fontsize=8, fontfamily="monospace",
                          fontweight="bold")
            else:
                ax_m.text(0.05, y, label, transform=ax_m.transAxes,
                          color="#7a8fa6", fontsize=7.5, fontfamily="monospace")
                ax_m.text(0.97, y, val, transform=ax_m.transAxes,
                          color=color, fontsize=7.5, fontfamily="monospace",
                          ha="right", fontweight="bold")
            y -= 0.065

        plt.tight_layout(pad=1.5)
        return fig

    if st.button("▶ Ejecutar pruebas de rendimiento", key="btn_stress"):
        results  = []
        sizes    = [5, 10, 20]
        prog     = st.progress(0)
        status_box = st.empty()

        for k, n in enumerate(sizes):
            status_box.info(f"⏳ Probando con {n} nodos...")
            G_test = random_complete_graph(n, seed=42)
            start  = "N0"

            t0    = time.perf_counter()
            dpath, dlen = dijkstra(G_test, "N0", f"N{n-1}")
            t_dij = time.perf_counter() - t0

            t0     = time.perf_counter()
            has_euler_circuit(G_test)
            t_euler = time.perf_counter() - t0

            t0    = time.perf_counter()
            hpath, hcost = hamiltonian_cycle_dp(G_test, start)
            t_ham = time.perf_counter() - t0

            results.append({
                "Nodos":             n,
                "Aristas":           G_test.number_of_edges(),
                "Dijkstra (ms)":     round(t_dij * 1000, 4),
                "Euler check (ms)":  round(t_euler * 1000, 4),
                "Hamiltoniano (s)":  round(t_ham, 4),
                "Subproblemas Ham":  (2**n) * n,
            })

            st.markdown(f"---")
            st.markdown(f"#### Prueba con {n} nodos — {G_test.number_of_edges()} aristas")

            col_info1, col_info2, col_info3 = st.columns(3)
            col_info1.metric("⚡ Dijkstra",     f"{t_dij*1000:.3f} ms")
            col_info2.metric("🔷 Hamiltoniano", f"{t_ham:.4f} s")
            col_info3.metric("🧮 Subproblemas", f"{(2**n)*n:,}")

            fig_test = draw_test_graph(
                G_test, hpath,
                f"Grafo de {n} nodos — Ciclo Hamiltoniano resaltado",
                t_dij, t_ham, n
            )
            st.pyplot(fig_test)
            plt.close(fig_test)

            prog.progress((k + 1) / len(sizes))

        status_box.success("✅ Pruebas completadas.")

        st.markdown("---")
        st.markdown("#### Resumen comparativo")
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        fig_bar, axes_bar = plt.subplots(1, 2, figsize=(11, 4))
        fig_bar.patch.set_facecolor(PALETTE["bg"])
        for ax in axes_bar:
            ax.set_facecolor(PALETTE["bg"])
            ax.tick_params(colors=PALETTE["text"])
            for sp in ax.spines.values(): sp.set_color(PALETTE["edge"])
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.xaxis.label.set_color(PALETTE["text"])
            ax.title.set_color(PALETTE["node_hl"])

        axes_bar[0].bar([str(r["Nodos"]) for r in results],
                        [r["Dijkstra (ms)"] for r in results],
                        color=PALETTE["node_hl"], width=0.4)
        axes_bar[0].set_title("Dijkstra — tiempo real (ms)")
        axes_bar[0].set_xlabel("Nodos"); axes_bar[0].set_ylabel("ms")

        axes_bar[1].bar([str(r["Nodos"]) for r in results],
                        [r["Hamiltoniano (s)"] for r in results],
                        color="#e85d5d", width=0.4)
        axes_bar[1].set_title("Hamiltoniano — tiempo real (s)")
        axes_bar[1].set_xlabel("Nodos"); axes_bar[1].set_ylabel("segundos")

        plt.tight_layout()
        st.pyplot(fig_bar)

        st.markdown("""
**¿Por qué el Hamiltoniano tarda tanto más?**

| Nodos | Subproblemas Held-Karp | Dijkstra ops aprox. |
|---|---|---|
| 5  | 2⁵ × 5 = **160** | ~50 |
| 10 | 2¹⁰ × 10 = **10,240** | ~300 |
| 20 | 2²⁰ × 20 = **20,971,520** | ~1,800 |

Cada vez que se duplican los nodos, el Hamiltoniano no duplica su trabajo: lo **multiplica por 4 o más**.  
El caché `lru_cache` garantiza que cada subproblema se resuelve **una sola vez**, de lo contrario con 20 nodos sería inviable.
        """)