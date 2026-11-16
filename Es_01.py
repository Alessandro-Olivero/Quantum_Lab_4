"""Esercizio 1: formulazione QUBO del Max-Cut e simulated annealing.

Per ogni nodo i, x_i = 0 indica S1 e x_i = 1 indica S2.
Il peso del taglio e' C(x) = sum_(i,j) w_ij*(x_i + x_j - 2*x_i*x_j).
Il QUBO da minimizzare e' f(x) = -C(x).
"""

from pathlib import Path

from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo


# Legge il grafo fornito dalla consegna. Il percorso e' relativo allo script.
# La prima riga contiene il numero di nodi; le altre contengono i, j e peso.
input_file = Path(__file__).resolve().parent / "Lab4_input_files/Lab4/max_cut.txt"
with input_file.open(encoding="utf-8") as file:
    n = int(file.readline())
    edges = []
    for line in file:
        i, j, weight = line.split()
        edges.append((int(i), int(j), float(weight)))

# Crea una variabile binaria simbolica per ciascun nodo del grafo.
# Il valore del bit determina a quale dei due sottoinsiemi appartiene il nodo.
x = [boolean_var(i) for i in range(n)]

# Costruisce la formulazione generica sommando il contributo di ogni arco.
# Per bit uguali, 2*x_i*x_j - x_i - x_j vale 0: l'arco non e' tagliato.
# Per bit diversi vale -1: l'arco contribuisce con -weight all'energia.
# Minimizzare l'energia equivale quindi a massimizzare il peso del taglio.
# Non servono vincoli: ogni assegnazione binaria definisce una partizione.
objective = QUBO()
for i, j, weight in edges:
    objective += weight * (2 * x[i] * x[j] - x[i] - x[j])

# Risolve il QUBO tramite simulated annealing con 100 ricerche.
# Durante il raffreddamento si riduce la probabilita' di accettare mosse
# che aumentano l'energia. Il metodo cerca un minimo, senza certificarlo.
# Il seed fissa la casualita' per rendere riproducibile questa esecuzione.
results = anneal_qubo(objective, num_anneals=100, seed=42)

# Seleziona la soluzione con energia minima tra quelle trovate.
# Un nodo assente dal QUBO (ad esempio isolato) puo' stare in entrambi gli
# insiemi: get(i, 0) lo assegna convenzionalmente a S1.
state = {i: results.best.state.get(i, 0) for i in range(n)}

# Traduce i bit nei due sottoinsiemi e ricava il peso da C(x) = -f(x).
print("S1:", [i for i in range(n) if state[i] == 0])
print("S2:", [i for i in range(n) if state[i] == 1])
print(f"Peso del taglio: {-results.best.value:g}")
