"""Esercizio 2: formulazione QUBO del graph coloring e simulated annealing.

x_ic = 1 se il nodo i ha il colore c, altrimenti x_ic = 0.
Ogni nodo deve avere un solo colore e nodi adiacenti devono avere colori diversi.
Per variabili binarie, la penalita' per x_ic + x_jc <= 1 e' x_ic*x_jc.
Il QUBO da minimizzare e':
    f(x) = P * sum_i (1 - sum_c x_ic)**2
         + P * sum_(i,j) sum_c x_ic*x_jc.
"""

from pathlib import Path

from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo


# Legge il grafo fornito dalla consegna. Il percorso e' relativo allo script.
# La prima riga contiene numero di nodi e numero di colori disponibili.
# Le righe successive contengono gli estremi degli archi non orientati.
input_file = Path(__file__).resolve().parent / "Lab4_input_files/Lab4/graph_coloring.txt"
with input_file.open(encoding="utf-8") as file:
    n, num_colors = map(int, file.readline().split())
    edges = []
    for line in file:
        i, j = map(int, line.split())
        edges.append((i, j))

# Crea una variabile binaria simbolica per ogni coppia (nodo, colore).
# x[i][c] vale 1 quando il nodo i riceve il colore c, 0 altrimenti.
# I colori vengono numerati da 0 a num_colors-1.
x = [[boolean_var((i, c)) for c in range(num_colors)] for i in range(n)]

# L'obiettivo contiene soltanto penalita' non negative. Qualsiasi P > 0
# rende una colorazione valida un minimo a energia zero; scegliamo P = 1.
# Il numero di colori e' gia' fornito, quindi non occorre minimizzarlo.
P = 1
objective = QUBO()

# Primo vincolo: ogni nodo deve avere esattamente un colore.
# Se la somma dei bit del nodo vale 1, la penalita' e' nulla;
# se non viene scelto alcun colore o se ne vengono scelti piu' di uno,
# il quadrato produce una penalita' positiva.
# qubovert applica x_ic**2 = x_ic, ottenendo un polinomio quadratico.
for i in range(n):
    objective += P * (1 - sum(x[i])) ** 2

# Secondo vincolo: per ogni arco e colore, x_ic + x_jc <= 1.
# Il prodotto dei due bit vale 1 soltanto nel caso vietato (1, 1),
# cioe' quando entrambi gli estremi hanno lo stesso colore.
# Nei casi ammessi (0, 0), (0, 1) e (1, 0) il prodotto vale 0.
# Questa penalita' e' gia' quadratica e non richiede variabili ausiliarie.
for i, j in edges:
    for c in range(num_colors):
        objective += P * x[i][c] * x[j][c]

# Risolve il QUBO con 100 ricerche di simulated annealing.
# Durante il raffreddamento diminuisce la probabilita' di accettare mosse
# che aumentano l'energia. Il seed fissa la casualita' dell'esecuzione.
results = anneal_qubo(objective, num_anneals=100, seed=42)

# best seleziona lo stato con energia minima tra le ricerche effettuate.
# Le etichette (i, c) identificano i bit attivi e quindi il colore del nodo.
state = results.best.state
for i in range(n):
    colors = [c for c in range(num_colors) if state[(i, c)] == 1]
    print(f"Nodo {i}: colori {colors}")

# Energia zero significa che tutti i vincoli di colorazione sono soddisfatti.
print(f"Energia QUBO: {results.best.value:g}")
