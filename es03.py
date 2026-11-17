"""Esercizio 3: formulazione QUBO del bagaglio e simulated annealing.

x_i = 1 se l'oggetto i viene inserito nel bagaglio, altrimenti x_i = 0.
Si massimizza sum_i p_i*x_i rispettando sum_i w_i*x_i <= Wmax.
Si introduce lo slack s >= 0: sum_i w_i*x_i + s = Wmax.
Per i pesi interi del problema, si usa la codifica unaria:
    s = sum_k s_k, con Wmax variabili binarie s_k.
Il QUBO da minimizzare e':
    f(x, s) = -sum_i p_i*x_i + P*(sum_i w_i*x_i + s - Wmax)**2.
"""

from pathlib import Path

from qubovert import boolean_var
from qubovert.sim import anneal_qubo


# Legge i dati forniti dalla consegna. Il percorso e' relativo allo script.
# La prima riga contiene numero di oggetti e peso massimo del bagaglio.
# Le altre righe contengono punteggio di preferenza e peso di ogni oggetto.
input_file = Path(__file__).resolve().parent / "Lab4_input_files/Lab4/objects.txt"
with input_file.open(encoding="utf-8") as file:
    n, max_weight = map(int, file.readline().split())
    preferences = []
    weights = []
    for line in file:
        preference, weight = map(int, line.split())
        preferences.append(preference)
        weights.append(weight)

# Una variabile binaria per oggetto: 1 significa portarlo, 0 lasciarlo.
# Le etichette distinguono le variabili degli oggetti da quelle dello slack.
x = [boolean_var(("x", i)) for i in range(n)]

# Lo slack rappresenta la capacita' inutilizzata del bagaglio.
# La somma di Wmax bit puo' assumere tutti i valori interi da 0 a Wmax:
# con limite 10 kg servono qui 10 bit, ciascuno associato a 1 kg.
# Ogni selezione ammissibile puo' quindi annullare la penalita' scegliendo
# s = Wmax - peso_totale. Se il peso supera Wmax, nessuno slack non negativo
# puo' soddisfare l'uguaglianza.
slack_bits = [boolean_var(("s", k)) for k in range(max_weight)]
slack = sum(slack_bits)

# Costruisce simbolicamente punteggio e peso della selezione.
# Il segno meno sul punteggio trasforma la massimizzazione in minimizzazione.
total_preference = sum(preferences[i] * x[i] for i in range(n))
total_weight = sum(weights[i] * x[i] for i in range(n))

# Con i punteggi positivi forniti, la loro somma limita il guadagno possibile.
# I pesi sono interi: una violazione dell'uguaglianza ha quadrato almeno 1.
# Scegliendo P maggiore della somma dei punteggi, nessun guadagno puo'
# compensare una violazione. Una selezione ammissibile ha penalita' zero.
P = sum(preferences) + 1
objective = -total_preference + P * (total_weight + slack - max_weight) ** 2

# Il quadrato di un'espressione lineare e' al massimo quadratico.
# qubovert semplifica i quadrati dei bit usando x_i**2 = x_i.
# Risolve il QUBO con 1000 ricerche di simulated annealing, a seed fissato.
# La temperatura scende da 100 a 0.1 in 10000 passi: inizialmente permette
# di superare le barriere dovute alla penalita', poi favorisce energie basse.
results = anneal_qubo(
    objective, num_anneals=1000, anneal_duration=10000,
    temperature_range=(100, 0.1), seed=42,
)

# best seleziona lo stato con energia minima tra quelli trovati.
# Gli indici degli oggetti seguono l'ordine del file, partendo da 0.
state = results.best.state
selected = [i for i in range(n) if state[("x", i)] == 1]
print("Oggetti selezionati:", selected)
print("Peso totale:", sum(weights[i] for i in selected), "kg")
print("Preferenza totale:", sum(preferences[i] for i in selected))
