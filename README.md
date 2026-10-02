# AI_esame

# 🚦 RoadEnv Pathfinding: Uniform Cost Search in OpenAI Gym

Implementazione di un agente intelligente per la ricerca del percorso a costo minimo all'interno di un ambiente a griglia custom (**RoadEnv-v0**) basato su **OpenAI Gym**, sviluppato per il laboratorio del corso di **Intelligenza Artificiale**.

Il progetto affronta un problema di ricerca deterministica con costi di transizione non uniformi (semafori singoli e doppi, ostacoli/muri) risolto tramite algoritmo di ricerca informata / a costo uniforme (**Uniform Cost Search - UCS**).

---

## 📌 Indice dei Contenuti

- [Descrizione del Problema](#-descrizione-del-problema)
- [Modello dell'Ambiente e Costi](#-modello-dellambiente-e-costi)
- [Algoritmo Risolutivo](#-algoritmo-risolutivo)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Requisiti e Installazione](#-requisiti-e-installazione)
- [Esecuzione ed Esperimenti](#-esecuzione-ed-esperimenti)
- [Metriche e Risultati](#-metriche-e-risultati)

---

## 🧭 Descrizione del Problema

Dato un ambiente stradale modellato come una griglia 2D $9 \times 9$:
* **Stato Iniziale**: Cella di partenza $S = (0, 0)$
* **Stato Obiettivo (Goal)**: Cella obiettivo $G = (8, 6)$
* **Spazio delle Azioni**: Movimento discreto nelle 4 direzioni cardinali:
  * `0: Left (L)`
  * `1: Right (R)`
  * `2: Up (U)`
  * `3: Down (D)`
  *(Le transizioni verso i muri `W` sono bloccate)*.

Le azioni sono completamente deterministiche. L'obiettivo consiste nell'individuare una sequenza di mosse da $S$ a $G$ che minimizzi il costo totale del percorso (*path cost*).

---

## 🚦 Modello dell'Ambiente e Costi

Ogni casella della matrice appartiene a una specifica tipologia con relative penalità di costo:

| Cella | Descrizione | Costo Transizione |
| :--- | :--- | :--- |
| `R` | Strada normale (*Road*) | **1** |
| `Ts` | Incrocio con **un semaforo** (*Single Traffic Light*) | **2** |
| `Tl` | Incrocio congestionato con **doppio semaforo** (*Double Traffic Light*) | **5** |
| `W` | Muro / Ostacolo non transitabile (*Wall*) | Non accessibile |
| `S` | Start $(0, 0)$ | - |
| `G` | Goal $(8, 6)$ | **1** |

### Mappa della Griglia ($9 \times 9$)

```text
[['S' 'R' 'W' 'W' 'W' 'W' 'R' 'W' 'W']
 ['W' 'Ts' 'R' 'R' 'R' 'R' 'Tl' 'R' 'R']
 ['W' 'R' 'W' 'W' 'W' 'W' 'R' 'W' 'W']
 ['R' 'Ts' 'R' 'Ts' 'R' 'R' 'Ts' 'W' 'W']
 ['W' 'W' 'W' 'R' 'W' 'W' 'R' 'Ts' 'R']
 ['W' 'R' 'R' 'Tl' 'W' 'W' 'W' 'R' 'W']
 ['W' 'R' 'W' 'R' 'Ts' 'R' 'R' 'Tl' 'R']
 ['W' 'R' 'W' 'W' 'R' 'W' 'W' 'R' 'W']
 ['R' 'Ts' 'R' 'R' 'Tl' 'R' 'G' 'Ts' 'R']]
```

---

## 🧠 Algoritmo Risolutivo

Poiché le transizioni presentano costi variabili ($c \in \{1, 2, 5\}$), la semplice ricerca in ampiezza (*Breadth-First Search - BFS*) non garantisce l'ottimalità della soluzione.

È stato adottato un approccio **Uniform Cost Search (Dijkstra)** con coda con priorità (`PriorityQueue`):
* **Frontiera**: I nodi vengono estratti in ordine di costo progressivo $g(n) = \text{path\_cost}$.
* **Insieme Esplorati (`explored`)**: Memorizza gli stati già visitati per prevenire cicli e riespansioni ridondanti.
* **Ottimalità**: Garantisce il raggiungimento del percorso a costo minimo globale per qualsiasi funzione di costo non negativa.

---

## 📂 Struttura del Progetto

```text
.
├── AI_exam_ex_3.ipynb            # Jupyter Notebook principale con soluzione e validazione
├── images/
│   ├── road_env.jpg              # Rappresentazione visiva della mappa di gioco
│   └── smaller_road_env.jpg      # Grafica dell'ambiente a risoluzione ridotta
└── tools/
    ├── AI_lab.yml                # Configurazione ambiente Conda
    ├── envs/
    │   ├── road_env.py           # Registrazione e dinamiche custom di RoadEnv-v0
    │   └── obsgrid_env.py        # Ambiente base a griglia con ostacoli
    ├── gym/                      # Framework OpenAI Gym integrato per il laboratorio
    └── utils/
        └── ai_lab_functions.py   # Strutture dati ausiliarie (Node, PriorityQueue, build_path)
```

---

## 💻 Requisiti e Installazione

Il progetto utilizza l'ambiente Conda configurato tramite `AI_lab.yml`.

### 1. Creazione dell'ambiente Conda

```bash
conda env create -f tools/AI_lab.yml
conda activate AI_lab
```

### 2. Pacchetti principali inclusi

* Python 3.7+ / 3.11+
* `gym`
* `numpy`
* `tqdm`
* `jupyter` / `ipykernel`

---

## 🚀 Esecuzione ed Esperimenti

Avviare Jupyter Notebook o JupyterLab:

```bash
jupyter notebook AI_exam_ex_3.ipynb
```

### Snippet di Risoluzione

```python
import gym
import tools.envs
from utils.ai_lab_functions import PriorityQueue, Node, build_path

def my_solution(environment):
    queue = PriorityQueue()
    explored = set()

    start_node = Node(environment.startstate, None)
    queue.add(start_node)

    time_cost = 0
    memory_cost = 0

    while not queue.is_empty():
        node = queue.remove()
        explored.add(node)
        time_cost += 1

        if node.state == environment.goalstate:
            return build_path(node), time_cost, memory_cost

        for action in range(environment.action_space.n):
            child_state = environment.sample(node.state, action)
            cost = 1
            if environment.grid[child_state] == 'Tl':
                cost = 5
            elif environment.grid[child_state] == 'Ts':
                cost = 2

            child = Node(child_state, node, node.pathcost + cost, node.pathcost + cost)
            
            if child_state not in explored and child_state not in queue:
                queue.add(child)
                explored.add(child_state)
                memory_cost = max(memory_cost, len(queue) + len(explored))

    return None, time_cost, memory_cost
```

---

## 📊 Metriche e Risultati

La validazione eseguita all'interno del notebook produce i seguenti risultati:

* **Tempo di Esecuzione**: $\sim 0.0091\text{ s}$
* **Nodi Esplorati (Time Cost)**: 47
* **Picco Nodi in Memoria (Space Cost)**: 95
* **Esito**: `Your solution is correct!`

### Percorso Ottimo Estratto:
```text
(0, 1) -> (1, 1) -> (2, 1) -> (3, 1) -> (3, 2) -> (3, 3) -> 
(3, 4) -> (3, 5) -> (3, 6) -> (4, 6) -> (4, 7) -> (5, 7) -> 
(6, 7) -> (7, 7) -> (8, 7) -> (8, 6)
```
