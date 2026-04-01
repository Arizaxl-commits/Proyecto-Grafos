# 🗺️ Rutas Óptimas con Teoría de Grafos

Aplicación web construida en **Python + Streamlit** que implementa tres algoritmos clásicos de teoría de grafos aplicados a las intersecciones.

---

## Algoritmos implementados

| Funcionalidad | Algoritmo | Complejidad |
|---|---|---|
| Ruta más corta A → B | Dijkstra | O((V+E) log V) |
| Recorrer todas las calles | Hierholzer (Euler) | O(E) |
| Visitar todos los nodos | Held-Karp (Hamiltoniano, DP) | O(2ⁿ · n²) |

---

## Requisitos

- Python 3.10 o superior
- pip

---

## Instalación y ejecución local

```bash
# 1. Clonar el repositorio
git clone https://github.com/Arizaxl-commits/Proyecto-Grafos.git
cd Proyecto-Grafos

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar la aplicación
streamlit run app.py
```

La app abre automáticamente en `http://localhost:8501`.

---

## Pruebas de rendimiento

Para ejecutar las pruebas de estrés de forma independiente (sin la interfaz web):

```bash
python stress_test.py
```

Genera un reporte en consola y guarda `resultados_rendimiento.txt`.

---

## Estructura del proyecto

```
Proyecto-Grafos/
├── app.py                  # Aplicación principal Streamlit
├── stress_test.py          # Script de pruebas de rendimiento
├── requirements.txt        # Dependencias
└── README.md               # Este archivo
```

---

## Uso de la aplicación

1. **Definir el grafo** en la barra lateral con el formato:
   ```
   NodoA, NodoB, peso
   ```
   Por ejemplo:
   ```
   Parque Principal, Catedral, 1.2
   Catedral, Alcaldía, 0.5
   ```
2. Hacer clic en **⚙ Cargar Grafo**.
3. Seleccionar la pestaña del algoritmo deseado y ejecutarlo.

### Pestaña 1 — Dijkstra
Selecciona origen y destino. Devuelve el camino más corto y lo resalta en el grafo.

### Pestaña 2 — Circuito de Euler
Encuentra un recorrido que pase por **cada arista exactamente una vez** y regrese al origen. Requiere que todos los nodos tengan grado par.

### Pestaña 3 — Ciclo Hamiltoniano
Encuentra el recorrido de **costo mínimo** que visita cada nodo exactamente una vez y vuelve al inicio. Usa programación dinámica con bitmask (Held-Karp) y `@lru_cache`.

### Pestaña 4 — Pruebas de Rendimiento
Genera grafos aleatorios de 5, 10 y 20 nodos y mide el tiempo de cada algoritmo.

---

## Notas sobre rendimiento

- **Dijkstra**: prácticamente instantáneo para cualquier tamaño razonable.
- **Euler**: lineal en el número de aristas; muy rápido.
- **Hamiltoniano**: crece exponencialmente. Para 20 nodos puede tardar varios segundos. Para > 25 nodos puede superar el minuto. El uso de `@functools.lru_cache` es esencial para mantenerlo viable en 20 nodos.

---

## Autores
Juan David Ariza Heredia 202551149-2724
Desarrollado para la asignatura de **Matemáticas Discretas 2 / Teoría de Grafos**.  

