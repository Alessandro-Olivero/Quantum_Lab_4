from pathlib import Path
import random

import pandas as pd
from qubovert import boolean_var
from qubovert.sim import anneal_qubo

# Lettura delle affinità. Il file è nella stessa cartella dello script.
df = pd.read_csv(Path(__file__).resolve().parent / "Lab4_input_files/Lab4/objects.txt", header=0, index_col=0)
A = df.values
N = A.shape[0]   # numero di ospiti (20)
T = 4            # numero di tavoli
S = 5            # posti per tavolo

# x[i][t] = 1 se l'ospite i siede al tavolo t
x = [[boolean_var(("x", i, t)) for t in range(T)] for i in range(N)]

# Obiettivo: massimizzare l'affinità tra ospiti allo stesso tavolo.
# Il segno meno trasforma la massimizzazione in minimizzazione.
# Con i < j ogni coppia è contata una volta sola (A è simmetrica).
obiettivo = -sum(A[i][j] * x[i][t] * x[j][t]
                 for i in range(N) for j in range(i + 1, N) for t in range(T))

# Vincolo 1: ogni ospite siede esattamente a un tavolo
vincolo_ospiti = sum((sum(x[i][t] for t in range(T)) - 1) ** 2 for i in range(N))

# Vincolo 2: ogni tavolo ha esattamente S ospiti
vincolo_tavoli = sum((sum(x[i][t] for i in range(N)) - S) ** 2 for t in range(T))

# Penalità: maggiore della somma delle affinità positive (scelta sicura).
# Provare a ridurla (P/2, P/4, ...) se la qualità è bassa.
P = sum(A[i][j] for i in range(N) for j in range(i + 1, N) if A[i][j] > 0) + 1

modello = obiettivo + P * (vincolo_ospiti + vincolo_tavoli)

results = anneal_qubo(modello, num_anneals=100, anneal_duration=10000,
                      temperature_range=(P, 0.1), seed=42)

state = results.best.state

# Lettura della soluzione: lista di ospiti per ogni tavolo
tavoli = {t: [i for i in range(N) if state.get(("x", i, t), 0) == 1] for t in range(T)}

# Verifica di validità
tavoli_per_ospite = [sum(state.get(("x", i, t), 0) for t in range(T)) for i in range(N)]
ospiti_ok = all(n == 1 for n in tavoli_per_ospite)
tavoli_ok = all(len(g) == S for g in tavoli.values())
valida = ospiti_ok and tavoli_ok


def affinita(assegnazione):
    """Affinità totale: somma di A[i][j] per le coppie allo stesso tavolo."""
    return sum(A[i][j] for g in assegnazione.values()
               for a, i in enumerate(g) for j in g[a + 1:])


print("P =", P)
for t, g in tavoli.items():
    print(f"Tavolo {t}: {g}")
print("Ogni ospite ha un solo tavolo:", ospiti_ok)
print("Ogni tavolo ha", S, "ospiti:", tavoli_ok)
print("Soluzione valida:", valida)
if valida:
    print("Affinità totale ottenuta:", affinita(tavoli))

# Benchmark di qualità: assegnazioni casuali valide
random.seed(0)
punteggi = []
for _ in range(1000):
    ordine = list(range(N))
    random.shuffle(ordine)
    casuale = {t: ordine[t * S:(t + 1) * S] for t in range(T)}
    punteggi.append(affinita(casuale))
print("Casuale: media =", sum(punteggi) / len(punteggi), ", migliore =", max(punteggi))