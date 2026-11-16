from pathlib import Path

from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo

# Legge il grafo fornito dalla consegna. Il percorso e' relativo allo script.
# La prima riga contiene numero di oggetti e peso massimo.
# Le righe successive contengono gli oggetti con la loro preferenza e peso.
input_file = Path(__file__).resolve().parent / "Lab4_input_files/Lab4/objects.txt"
with input_file.open(encoding="utf-8") as file:
    num_objects, max_weight = map(int, file.readline().split())
    object = []
    for line in file:
        i, j = map(int, line.split())
        object.append((i, j)) # dentro la lista object, ogni elemento è una tupla (preferenza, peso) dell'oggetto i

# Crea una variabile binaria simbolica per ogni oggetto.
# x[i] vale 1 quando l'oggetto i è stato preso, 0 altrimenti.
x = [boolean_var(i) for i in range(num_objects)]

modello = QUBO()

# calcolo la preferenza totale degli oggetti presi
# x[i] inidica se l'oggetto i è stato preso o meno
# object[i][0] indica la preferenza dell'oggetto i
for i in range(num_objects):
    modello += x[i] * object[i][0]

for i in range(num_objects):
    modello+= x[i]*object[i][1]

modello += s - max_weight 